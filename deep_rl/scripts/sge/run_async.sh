#!/bin/bash -l
#job queue
#$ -q all.q
#grid engine ouput
#$ -o $HOME/async.out
#$ -e $HOME/async.err
#job name
#$ -N async
#array jobs (here: number of experiment trials)
#$ -t 1-10
#use currently defined shell variables
#$ -V


######CONFIG##########################
DATA_DIR="/data2/logs"
SRC_DIR="$HOME/deep_q_rl"
N_AGENTS=3
N_EPOCHS=100
#####################################
#add source to path
PTHON_PATH="$PYTHON_PATH:$SRC_DIR/deep_rl"

#assure log dir exists
BASE_DIR="$DATA_DIR/async/run$SGE_TASK_ID"
mkdir -p $BASE_DIR

PORT=$((50000 + $SGE_TASK_ID))

cd "$SRC_DIR/deep_rl/scripts"

python run_nips.py --save-path="$BASE_DIR" --log_level INFO --mode async --deterministic false --num_agents "$N_AGENTS" --epochs "$N_EPOCHS"  --param-port "$PORT" --update-frequency 50
