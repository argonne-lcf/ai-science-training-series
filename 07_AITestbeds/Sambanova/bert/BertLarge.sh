#! /bin/bash 
set -e 
export SOFTWARE_HOME=/opt
LOGDIR=`date +%m%d%y.%H`
if [ "$1" ] ; then
LOGDIR=$1
fi
MODEL_NAME="BertLarge"
OUTPUT_PATH=${PWD}/${LOGDIR}/${MODEL_NAME}.out
echo "Using ${OUTPUT_PATH} for output"
mkdir -p ${PWD}/${LOGDIR}
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
PROJ_DIR=${PWD}

cd ${HOME}
echo "Model: " ${MODEL_NAME} > ${OUTPUT_PATH} 2>&1
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
if [ -e ${OUTDIR}/bertlrg/bertlrg.pef ] ; then
#rm ${OUTDIR}/bertlrg/bertlrg.pef
 echo "${OUTDIR}/bertlrg/bertlrg.pef exists"
fi
if [ ! -e ${OUTDIR}/bertlrg/bertlrg.pef ] ; then
  export SN_NUM_THREADS=32
  echo "COMPILE START AT ${SECONDS}" >> ${OUTPUT_PATH} 2>&1
COMMAND="python /opt/sambaflow/apps/nlp/transformers_on_rdu/transformers_hook.py compile --model_name_or_path bert-large-uncased --tokenizer_name bert-large-uncased --module_name mlm_ns --task_name mlm_ns  --max_seq_length 128 --per_device_train_batch_size 256 -b 256 --output_dir=${OUTDIR}/hf_output --overwrite_output_dir --cache_dir ${OUTDIR}/cache --compiler-configs-file /opt/sambaflow/apps/nlp/transformers_on_rdu/human_decisions/compiler_configs/compiler_configs_bertlarge_sc_mlm_ml_perf_fullfeature_macv2_gm.json --mac-human-decision /opt/sambaflow/apps/nlp/transformers_on_rdu/human_decisions/mac_overrides/bertlarge_sc_training_mlm_ml_perf_fullfeature_macv2.json --mac-v2 --non_split_head --dense_adam --data-parallel -ws 2 --weight_decay 0.01 --max_grad_norm_clip 1.0 --adam_beta2 0.98 --num-tiles 4 --pef-name=bertlrg --output-folder=${OUTDIR} --log-level error --disable-strict-conversion"
  echo "COMPILE COMMAND: $COMMAND" >> ${OUTPUT_PATH} 2>&1
  eval $COMMAND >> ${OUTPUT_PATH} 2>&1
  echo "COMPILE END AT ${SECONDS}" >> ${OUTPUT_PATH} 2>&1
fi
#######################
echo "RUN" >> ${OUTPUT_PATH} 2>&1
env >> ${OUTPUT_PATH} 2>&1
/usr/local/bin/sbatch --output=${HOME}/slurm-%A.out --ntasks 16 --gres=rdu:8 --ntasks-per-node 16  --nodes 1 --nodelist $(hostname) --cpus-per-task=8  ${PROJ_DIR}/BertLarge_run.sh $1 >> ${OUTPUT_PATH} 2>&1

#######################
echo "Machine state After: " >> ${OUTPUT_PATH} 2>&1
/opt/sambaflow/bin/snfadm -l inventory >> ${OUTPUT_PATH} 2>&1
echo "Duration: " $SECONDS >> ${OUTPUT_PATH} 2>&1
