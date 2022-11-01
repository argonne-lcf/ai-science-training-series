"""
ai4sci/trainer.py

Implements Trainer object.
"""
from __future__ import absolute_import, annotations, division, print_function
import os
import logging

# this limits the amount of memory used
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
PARALLEL_THREADS = 128
PREFETCH_BUFFER_SIZE = 8
os.environ['OMP_NUM_THREADS'] = str(PARALLEL_THREADS)
NUM_PARALLEL_READERS = PARALLEL_THREADS

import time

from typing import Optional

import tensorflow as tf
import horovod.tensorflow as hvd

from omegaconf import DictConfig
from pathlib import Path
from ai4sci.network import ResNet34, ConvNet

HERE = Path(os.path.abspath(__file__)).parent
CHECKPOINT_DIR = HERE.joinpath('checkpoints')


log = logging.getLogger(__name__)
tf.autograph.set_verbosity(0)

RANK = hvd.rank()
SIZE = hvd.size()
LOCAL_RANK = hvd.local_rank()

Tensor = tf.Tensor
Model = tf.keras.models.Model
TF_FLOAT = tf.keras.backend.floatx()


def get_data(data_dir: os.PathLike) -> dict[str, tf.data.Dataset]:
    dpath = Path(data_dir).joinpath(f'mnist-{hvd.rank()}.npz')
    (xtrain, ytrain), (xtest, ytest) = (
        tf.keras.datasets.mnist.load_data(dpath.as_posix())
    )
    train_dset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(xtrain[..., tf.newaxis] / 255.0, TF_FLOAT),
         tf.cast(ytrain, tf.int64))
    )
    test_dset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(xtest[..., tf.newaxis] / 255.0, TF_FLOAT),
         tf.cast(ytest, tf.int64))
    )
    return {'train': train_dset, 'test': test_dset}


def metric_average(x: Tensor):
    assert x is not None and isinstance(x, Tensor)
    return x if SIZE == 1 else hvd.allreduce(x, average=True)


def calc_acc(logits, labels):
    pred = tf.argmax(logits, axis=1)
    correct = tf.cast(pred, TF_FLOAT) == tf.cast(labels, TF_FLOAT)
    return tf.reduce_mean(tf.cast(correct, TF_FLOAT))


def calc_loss(logits, labels):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
    return tf.reduce_mean(loss)


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.rank = RANK
        self._is_chief = (RANK == 0) and (LOCAL_RANK == 0)
        self.model = self.build_model()
        self.ntrain_samples = int(1281167)
        self.ntest_samples = int(50000)
        self.optimizer = tf.optimizers.Adam(cfg.lr_init * SIZE)
        self.loss_fn = calc_loss
        from ai4sci.network import prepare_data_loader
        train_dset, test_dset = prepare_data_loader(self.cfg.batch_size)
        self.datasets = {'train': train_dset, 'test': test_dset}
        # Setup checkpoint manager only from one process
        self.ckpt_dir = None
        self.checkpoint = None
        self.manager = None
        if self._is_chief:
            self.ckpt_dir = Path(CHECKPOINT_DIR).as_posix()
            self.checkpoint = tf.train.Checkpoint(
                model=self.model,
                optimizer=self.optimizer
            )
            self.manager = tf.train.CheckpointManager(
                self.checkpoint,
                max_to_keep=5,
                directory=self.ckpt_dir
            )

    def build_model(self) -> Model:
        # return ConvNet()
        return ResNet34()

    def save_checkpoint(self) -> None:
        if self._is_chief and self.manager is not None:
            self.manager.save()

    @tf.function
    def train_step(
            self,
            data: Tensor,
            target: Tensor,
            first_batch: Optional[bool] = False,
    ) -> tuple[Tensor, Tensor]:
        with tf.GradientTape() as tape:
            logits = self.model(data, training=True)
            loss = self.loss_fn(logits=logits, labels=target)
            pred = tf.cast(tf.math.argmax(logits, axis=1), target.dtype)
            acc = tf.math.reduce_sum(
                tf.cast(tf.math.equal(pred, target), TF_FLOAT)
            )

        # Horovod: add Horovod DistributedGradientTape
        tape = hvd.DistributedGradientTape(tape)
        grads = tape.gradient(loss, self.model.trainable_variables)
        updates = zip(grads, self.model.trainable_variables)
        self.optimizer.apply_gradients(updates)
        # Horovod: Broadcast initial variable states from rank 0 to all other
        # processes. This is necessary to ensure consistent initialization of
        # all workers when training is started with random weights or restored
        # from a checkpoint

        # NOTE: Broadcast should be done after the first gradient step
        # to ensure optimizer initialization
        if first_batch:
            hvd.broadcast_variables(self.model.variables, root_rank=0)
            hvd.broadcast_variables(self.optimizer.variables(), root_rank=0)

        return loss, acc

    def train_epoch(
            self,
            epoch: Optional[int] = None
    ) -> dict:
        epoch = 0 if epoch is None else epoch
        steps_per_epoch = int(self.ntrain_samples // self.cfg.batch_size // SIZE)
        batch = self.datasets['train'].take(steps_per_epoch)

        # nstep = self.ntrain_samples // SIZE // self.cfg.batch_size
        # batch = self.datasets['train'].take(nstep)
        metrics = {}
        training_acc = 0.0
        running_loss = 0.0
        for bidx, (data, target) in enumerate(batch):
            t0 = time.time()
            loss, acc = self.train_step(  # type:ignore
                data,
                target,
                first_batch=(bidx == 0 and epoch == 0)
            )
            training_acc += acc
            running_loss += loss.numpy()
            if RANK == 0 and (bidx % self.cfg.logfreq == 0):
                metrics = {
                    'epoch': epoch,
                    'dt': time.time() - t0,
                    'running_loss': running_loss / (self.ntrain_samples // SIZE),
                    'batch_loss': loss / self.cfg.batch_size,
                    'acc': training_acc / (self.ntrain_samples // SIZE),
                    'batch_acc': acc / self.cfg.batch_size,
                }

                pre = [
                    f'[{RANK}]',
                    (
                        f'[{epoch}/{self.cfg.epochs}:'
                        f' {bidx * len(data)} / {self.ntrain_samples // SIZE}'
                        f' ({100. * bidx / steps_per_epoch:.0f}%)]'
                    ),
                ]
                log.info(' '.join([
                    *pre, *[f'{k}={v:.4f}' for k, v in metrics.items()]
                ]))
        running_loss = running_loss / self.ntrain_samples
        training_acc = training_acc / self.ntrain_samples
        training_acc = hvd.allreduce(training_acc, average=True)
        loss_avg = hvd.allreduce(running_loss, average=True)

        return {'loss': loss_avg, 'acc': training_acc}

    def test(self) -> Tensor:
        steps_test = int(self.ntest_samples // self.cfg.batch_size // SIZE)
        test_batch = self.datasets['test'].take(steps_test)
        total = 0
        correct = 0
        for data, target in test_batch:
            logits = self.model(data, training=False)
            pred = tf.cast(tf.math.argmax(logits, axis=1), target.dtype)
            total += target.shape[0]
            correct += tf.reduce_sum(
                tf.cast(tf.math.equal(pred, target), TF_FLOAT)
            )

        return correct / tf.constant(total, dtype=TF_FLOAT)
