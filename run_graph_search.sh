#!/bin/bash


DATASET=$1
METHOD=$2
TRMODE=$3
NOTE=$4
TOPK=$5
ActorPolicy=$6
Joint=$7
GraphEmbDim=${8}
GraphDim=${9}
NoHFeat=${10}
GraphTeacherOpt=${11}
NumMP=${12}
Fraction=${13}
SepQuery=${14}
AllQuery=${15}
CurPosH=${16}
PreTrain=${17}
PreEpc=${18}
GP0=${19}
GP1=${20}
GP2=${21}
GP3=${22}
GP4=${23}
GP5=${24}
GP6=${25}
GA=${26}
GAT=${27}
SM=${28}
SHARE=${29}
CA=${30}
MVS=${31}
MEL=${32}


GPConfig=${GP0}-${GP1}-${GP2}-${GP3}-${GP4}-${GP5}-${GP6}
ROOTDIR="./test/graph_search/${DATASET}/${METHOD}/${NOTE}/"
SUBDIR="train_tp-k-${TOPK}_grph-dim-${GraphDim}_no-h-feat-${NoHFeat}_grph-tcher-opt-${GraphTeacherOpt}_num-mp-${NumMP}_frac-${Fraction}_sep-q-${SepQuery}_all-q-${AllQuery}_CurPosH-${CurPosH}_${PreTrain}_${PreEpc}_grph-pling-${GPConfig}_${GA}_${GAT}_${SHARE}_${CA}_${MVS}_${MEL}_${NOVIS}_${NOLANG}"

CPTDIR="${ROOTDIR}/checkpoints/${SUBDIR}/"
RESULTDIR="${ROOTDIR}/results/${SUBDIR}/"
TBDIR="${ROOTDIR}/tensorboard_logs/${SUBDIR}/"
LOG="${ROOTDIR}/logs/${SUBDIR}.txt.`date +'%Y-%m-%d_%H-%M-%S'`.log"
mkdir -p "${ROOTDIR}/logs"
mkdir -p ${CPTDIR}
mkdir -p ${RESULTDIR}
mkdir -p ${TBDIR}

exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

hostname

CUDA_VISIBLE_DEVICES=0 python tasks/R2R-pano/main_graph_search.py \
    --exp_name ${NOTE} \
    --batch_size 28 \
    --img_fc_use_angle 1 \
    --img_feat_input_dim 2176 \
    --img_fc_dim 1024 \
    --rnn_hidden_size 512 \
    --eval_every_epochs 5  \
    --train_iters_epoch 200  \
    --use_ignore_index 1 \
    --arch ${METHOD} \
    --fix_action_ended 0 \
    --results_dir ${RESULTDIR} \
    --checkpoint_dir ${CPTDIR} \
    --log_dir ${TBDIR} \
    --dataset_name ${DATASET} \
    --training_mode ${TRMODE} \
    --GSA_top_K ${TOPK} \
    --actor_policy ${ActorPolicy} \
    --joint_training ${Joint} \
    --graph_emb_dim ${GraphEmbDim} \
    --graph_dim ${GraphDim} \
    --no_h_feat ${NoHFeat} \
    --graph_teacher_option ${GraphTeacherOpt} \
    --max_mp_steps ${NumMP} \
    --data_fraction ${Fraction} \
    --separate_query ${SepQuery} \
    --use_all_query ${AllQuery} \
    --use_cur_pos_h ${CurPosH} \
    --use_pretraining ${PreTrain} \
    --pretraining_epochs ${PreEpc} \
    --use_graph_pooling ${GP0} \
    --pooling_num_node  ${GP1} \
    --pooling_graph_dim ${GP2} \
    --planner_graph_dim ${GP3} \
    --pooling_mp_steps  ${GP4} \
    --normalize_pooling ${GP5} \
    --pooling_channels  ${GP6} \
    --graph_attention ${GA} \
    --graph_attention_type ${GAT} \
    --feedback_training ${SM} \
    --mlp_share ${SHARE} \
    --ctx_attend ${CA} \
    --moving_window_size ${MVS} \
    --max_episode_len ${MEL}

echo 'Done'

# bash run_neuralese_agent.sh R2R cogrounding test_proxy_neuralese
