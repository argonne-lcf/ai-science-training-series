"""
main.py

Simple example demonstrating how to use Horovod with Tensorflow for data
parallel distributed training.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging

import time

import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import hydra

from omegaconf import DictConfig

log = logging.getLogger(__name__)
tf.autograph.set_verbosity(0)

hvd.init()
RANK = hvd.rank()
SIZE = hvd.size()
LOCAL_RANK = hvd.local_rank()
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(
        gpus[hvd.local_rank()],
        'GPU'
    )


from ai4sci.trainer import Trainer

Tensor = tf.Tensor
Model = tf.keras.models.Model
TF_FLOAT = tf.keras.backend.floatx()


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> None:
    epoch_times = []
    start = time.time()
    trainer = Trainer(cfg)
    for epoch in range(cfg.epochs):
        t0 = time.time()
        metrics = trainer.train_epoch(epoch)
        epoch_times.append(time.time() - t0)

        if epoch % cfg.logfreq == 0 and RANK == 0:
            acc = trainer.test()
            astr = f'[TEST] Accuracy: {acc:.0f}%'
            sepstr = '-' * len(astr)
            log.info(sepstr)
            log.info(astr)
            log.info(sepstr)
            summary = '  '.join([
                '[TRAIN]',
                f'loss={metrics["loss"]:.4f}',
                f'acc={metrics["acc"] * tf.constant(100., TF_FLOAT):.0f}%'
            ])
            #log.info((sep := '-' * len(summary)))
            log.info(summary)
            #log.info(sep)

            trainer.save_checkpoint()

    log.info(f'Total training time: {time.time() - start} seconds')
    nepochs = min(len(epoch_times), 5)
    log.info(
        f'Average time per epoch in the last 5: {np.mean(epoch_times[-nepochs])}'
    )


if __name__ == '__main__':
    main()
