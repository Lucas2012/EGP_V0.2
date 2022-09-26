#!/bin/bash

actor_policy='feature'
train_mode='GSA_Graph_supervised_pure_scratch'

joint=1
no_h_feat=1
graph_emb_dim=-1

fraction=1

sample_method='sample2action'

use_all_query=0

dataset=R2R

sep_query=0

use_pretraining=0
pretraining_epochs=-1
use_cur_pos_h=1

top_k=16

graph_teacher_option=follow_gt

num_mp=3

#use_graph_pooling
GP0=1
#pooling_num_node
GP1=6
#pooling_graph_dim
GP2=32
#planner_graph_dim
GP3=256
#pooling_mp_steps
GP4=3
#normalize_pooling
GP5='softmax'
#pooling_channels
GP6=1

use_attention=0
attention_type='none'

#graph_teacher_option=default_teacher
ctx_attend=0
SHARE=1

MVS=0

MEL=10

graph_dim=256
top_k=16
GP6=3
num_mp=3


LOG='test_run.log'
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

bash run_graph_search.sh ${dataset} cogrounding ${train_mode} VLN_GSA_supervised_pure_scratch_${sample_method}_pooling-graph_${fraction} ${top_k} ${actor_policy} ${joint} ${graph_emb_dim} ${graph_dim} ${no_h_feat} ${graph_teacher_option} ${num_mp} ${fraction} ${sep_query} ${use_all_query} ${use_cur_pos_h} ${use_pretraining} ${pretraining_epochs} ${GP0} ${GP1} ${GP2} ${GP3} ${GP4} ${GP5} ${GP6} ${use_attention} ${attention_type} ${sample_method} ${SHARE} ${ctx_attend} ${MVS} ${MEL}
