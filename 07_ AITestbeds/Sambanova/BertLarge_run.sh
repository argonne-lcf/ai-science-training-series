#! /bin/bash 
set -e 
export SOFTWARE_HOME=/opt
LOGDIR=`date +%m%d%y.%H`
if [ "$1" ] ; then
LOGDIR=$1
fi
MODEL_NAME="BertLarge"
OUTPUT_PATH=/data/ANL/results/$(hostname)/${USER}/${LOGDIR}/${MODEL_NAME}.out
echo "Using ${OUTPUT_PATH} for output"
mkdir -p /data/ANL/results/$(hostname)/${USER}/${LOGDIR}
export SOFTWARE_HOME=/opt

LOGDIR=`date +%m%d%y.%H`
if [ "$1" ] ; then
LOGDIR=$1
fi
MODEL_NAME="BertLarge"
OUTPUT_PATH=/data/ANL/results/$(hostname)/${USER}/${LOGDIR}/${MODEL_NAME}.out
echo "Using ${OUTPUT_PATH} for output"
mkdir -p /data/ANL/results/$(hostname)/${USER}/${LOGDIR}
export SOFTWARE_HOME=/opt

ACTIVATE=/opt/sambaflow/apps/nlp/transformers_on_rdu/venv/bin/activate
#######################
# Edit these variables.
#######################
export OMP_NUM_THREADS=18
#export CCL_TIMEOUT=3600
#######################
# Start script timer
SECONDS=0
# Temp file location
DIRECTORY=$$
#OUTDIR=${HOME}/${DIRECTORY}
OUTDIR=${HOME}/${MODEL_NAME}

source ${ACTIVATE}
cd ${HOME}
echo "Model: " ${MODEL_NAME} >> ${OUTPUT_PATH} 2>&1
echo "Date: " $(date +%m/%d/%y) >> ${OUTPUT_PATH} 2>&1
echo "Time: " $(date +%H:%M) >> ${OUTPUT_PATH} 2>&1
apt list --installed sambaflow >> ${OUTPUT_PATH} 2>&1
if [ ! -d ${OUTDIR} ] ; then
  mkdir ${OUTDIR}
fi
cd ${OUTDIR}
#######################
echo "Machine State Before: " >> ${OUTPUT_PATH} 2>&1
/opt/sambaflow/bin/snfadm -l inventory >> ${OUTPUT_PATH} 2>&1
#######################
if [ ! -e ${OUTDIR}/bertlrg/bertlrg.pef ] ; then
  echo "${OUTDIR}/bertlrg/bertlrg.pef does not exist, exiting" >> ${OUTPUT_PATH} 2>&1
  exit 1
fi
#######################
echo "RUN" >> ${OUTPUT_PATH} 2>&1
export SAMBA_CCL_ASYNC_ALLREDUCE=0
export SAMBA_CCL_HIERARCHICAL_ALLREDUCE=0
env >> ${OUTPUT_PATH} 2>&1
FI_VERBS_IFACE=eth
#CONVFUNC_DEBUG_RUN=1
#SAMBA_SEED=256
#DISALLOW_VISUALIZE=True
#OMP_NUM_THREADS=8
  #COMMAND="/usr/local/bin/srun --mpi=pmi2 python /opt/sambaflow/apps/nlp/transformers_on_rdu/transformers_hook.py run --config_name /opt/sambaflow/apps/nlp/transformers_on_rdu/modules/configs/mlm_24layer_ml_perf_config.json --tokenizer_name bert-large-uncased --module_name mlm_ns --task_name mlm_ns --max_seq_length 128 -b 256 --output_dir=${OUTDIR}/hf_output --overwrite_output_dir --do_train  --per_device_train_batch_size  256  --data_dir /data/ANL/wikicorpus_en --cache ${OUTDIR}/cache/ --max_predictions_per_seq 20  --warmup_steps 12500 --max_steps 250000 --steps_this_run 5005 --logging_steps 1 --weight_decay 0.01 --learning_rate 0.000175  --non_split_head --dense_adam --data-parallel --reduce-on-rdu --adam_beta2 0.98 --max_grad_norm_clip 1.0 --min_throughput 570000  --max_throughput 620000 --skip_checkpoint  --pef=${OUTDIR}/bertlrg/bertlrg.pef --log-level error --validate_stat_perf --validate_tying_plus_embed_train "
  COMMAND="/usr/local/bin/srun --mpi=pmi2 python /opt/sambaflow/apps/nlp/transformers_on_rdu/transformers_hook.py run --config_name /opt/sambaflow/apps/nlp/transformers_on_rdu/modules/configs/mlm_24layer_ml_perf_config.json --tokenizer_name bert-large-uncased --module_name mlm_ns --task_name mlm_ns --max_seq_length 128 -b 256 --output_dir=${OUTDIR}/hf_output --overwrite_output_dir --do_train  --per_device_train_batch_size  256  --data_dir /data/ANL/wikicorpus_en --cache ${OUTDIR}/cache/ --max_predictions_per_seq 20  --warmup_steps 12500 --max_steps 250000 --steps_this_run 5005 --logging_steps 1 --weight_decay 0.01 --learning_rate 0.000175  --non_split_head --dense_adam --data-parallel --reduce-on-rdu --adam_beta2 0.98 --max_grad_norm_clip 1.0 --min_throughput 560000  --max_throughput 620000 --skip_checkpoint  --pef=${OUTDIR}/bertlrg/bertlrg.pef --log-level error --validate_stat_perf --validate_tying_plus_embed_train "
echo "RUN COMMAND: $COMMAND" >> ${OUTPUT_PATH} 2>&1
eval $COMMAND >> ${OUTPUT_PATH} 2>&1

#######################
echo "Machine state After: " >> ${OUTPUT_PATH} 2>&1
/opt/sambaflow/bin/snfadm -l inventory >> ${OUTPUT_PATH} 2>&1
echo "Duration: " $SECONDS >> ${OUTPUT_PATH} 2>&1