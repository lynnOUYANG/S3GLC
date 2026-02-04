#!/usr/bin/env bash
# set -euo pipefail

# LOG_DIR="logs/alpha_beta"
# mkdir -p "${LOG_DIR}"

# BETAS=(0.01 0.2 0.4 0.5 0.6 0.8 0.99)
# ALPHAS=(0.1 0.5 0.8 1 2 4 6)

# declare -A CMD_MAP
# CMD_MAP[ENZYMES]="python main.py --lr 0.0001 --hidden-dim 16 --num-gc-layers 4 --batch-size 512 --epochs 20 --iter 5 --seed 42 --DS ENZYMES --r 20 --d 8"
# CMD_MAP[AIDS]="python main.py --lr 0.0001 --hidden-dim 16 --num-gc-layers 4 --batch-size 512 --epochs 20 --iter 5 --seed 42 --DS AIDS --r 10 --d 4"
# CMD_MAP[BZR]="python main.py --lr 0.0001 --hidden-dim 16 --num-gc-layers 4 --batch-size 512 --epochs 20 --iter 5 --seed 42 --DS BZR --r 8 --d 2"
# CMD_MAP[MUTAG]="python main.py --lr 0.0001 --hidden-dim 16 --num-gc-layers 4 --batch-size 512 --epochs 20 --iter 5 --seed 42 --DS MUTAG --r 8 --d 4"
# CMD_MAP[PTC-MM]="python main.py --lr 0.0001 --hidden-dim 16 --num-gc-layers 4 --batch-size 256 --epochs 20 --iter 5 --seed 42 --DS PTC-MM --r 6 --d 2"
# python main.py --lr 0.0001 --hidden-dim 16 --num-gc-layers 4 --batch-size 256 --epochs 20 --iter 5 --seed 42 --DS DD --r 12 --d 4

python main.py --lr 0.002 --hidden-dim 16 --num-gc-layers 4 --batch-size 512 --epochs 20 --iter 5 --seed 42 --DS ENZYMES --r 16 --d 8 --gpu 1 
python main.py --lr 0.001 --hidden-dim 16 --num-gc-layers 4 --batch-size 256 --epochs 20 --iter 5 --seed 42 --DS AIDS --r 10 --d 5 --gpu 1
python main.py --lr 0.0008 --hidden-dim 16 --num-gc-layers 4 --batch-size 512 --epochs 20 --iter 5 --seed 42 --DS BZR --r 6 --d 4 --gpu 1
python main.py --lr 0.0001 --hidden-dim 16 --num-gc-layers 4 --batch-size 512 --epochs 20 --iter 5 --seed 42 --DS MUTAG --r 8 --d 4 --gpu 1
python main.py --lr 0.001 --hidden-dim 16 --num-gc-layers 4 --batch-size 512 --epochs 20 --iter 5 --seed 42 --DS PTC_MR --r 5 --d 3 --gpu 1
python main.py --lr 0.002 --hidden-dim 16 --num-gc-layers 4 --batch-size 512 --epochs 20 --iter 5 --seed 42 --DS DD --r 12 --d 4 --gpu 1
python main.py --lr 0.002 --hidden-dim 16 --num-gc-layers 4 --batch-size 256 --epochs 20 --iter 5 --seed 42 --DS PTC_MM --r 5 --d 3 --gpu 1

# for ds in ENZYMES AIDS BZR MUTAG "PTC-MM"; do
#   base_cmd="${CMD_MAP[$ds]}"
#   for beta in "${BETAS[@]}"; do
#     for alpha in "${ALPHAS[@]}"; do
#       echo "Running ${ds} beta=${beta} alpha=${alpha}"
#       log_file="${LOG_DIR}/${ds}_beta${beta}_alpha${alpha}.txt"
#       ${base_cmd} --beta "${beta}" --alpha "${alpha}" 2>&1 | tee "${log_file}"
#     done
#   done
# done

#!/usr/bin/env bash
# set -e

# BASE_CMD="srun -p short --gres=gpu:a100:1 python main.py \
#   --lr 0.0001 --hidden-dim 32 --num-gc-layers 4 --batch-size 4096 \
#   --epochs 20 --iter 5 --seed 42 --DS MNIST --r 40 --d 20"

# BETA_LIST=(0.01 )
# ALPHA_LIST=(0.02)

# for beta in "${BETA_LIST[@]}"; do
#   for alpha in "${ALPHA_LIST[@]}"; do
#     echo "Running with beta=${beta}, alpha=${alpha}"
#     ${BASE_CMD} --beta "${beta}" --alpha "${alpha}"
#   done
# done

# srun -p short --gres=gpu:a100:1 python main.py --lr 0.0001 --hidden-dim 32 --num-gc-layers 4 --batch-size 4096 --epochs 20 --iter 5 --seed 42 --DS MNIST --r 40 --d 12
# srun -p short --gres=gpu:a100:1 python main.py --lr 0.0001 --hidden-dim 32 --num-gc-layers 4 --batch-size 4096 --epochs 20 --iter 5 --seed 42 --DS CIFAR10 --r 35 --d 20

# srun -p short --gres=gpu:a100:1 python main.py --lr 0.0001 --hidden-dim 32 --num-gc-layers 4 --batch-size 4096 --epochs 20 --iter 5 --seed 42 --DS REDDIT-12K --r 40 --d 20
# srun -p short --gres=gpu:a100:1 python main.py --lr 0.0001 --hidden-dim 32 --num-gc-layers 4 --batch-size 4096 --epochs 20 --iter 5 --seed 42 --DS COLLAB --r 40 --d 20

# srun -p short --gres=gpu:a100:1 python main.py --lr 0.0001 --hidden-dim 32 --num-gc-layers 4 --batch-size 4096 --epochs 20 --iter 5 --seed 42 --DS COLLAB --r 30 --d 20
# srun -p short --gres=gpu:a100:1 python main.py --lr 0.0001 --hidden-dim 32 --num-gc-layers 4 --batch-size 4096 --epochs 20 --iter 5 --seed 42 --DS COLLAB --r 40 --d 20
# # srun -p short --gres=gpu:a100:1 python main.py --lr 0.0001 --hidden-dim 32 --num-gc-layers 4 --batch-size 4096 --epochs 20 --iter 5 --seed 42 --DS COLLAB --r 50 --d 20

# srun -p short --gres=gpu:a100:1 python main.py --lr 0.0001 --hidden-dim 32 --num-gc-layers 4 --batch-size 4096 --epochs 20 --iter 5 --seed 42 --DS COLLAB --r 70 --d 20
# srun -p short --gres=gpu:a100:1 python main.py --lr 0.0001 --hidden-dim 32 --num-gc-layers 4 --batch-size 4096 --epochs 20 --iter 5 --seed 42 --DS COLLAB --r 80 --d 20

srun -p short --gres=gpu:a100:1 python main.py --lr 0.001 --hidden-dim 32 --num-gc-layers 4 --batch-size 4096 --epochs 20 --iter 5 --seed 42 --DS COLLAB --r 40 --d 20
srun -p short --gres=gpu:a100:1 python main.py --lr 0.001 --hidden-dim 32 --num-gc-layers 4 --batch-size 4096 --epochs 20 --iter 5 --seed 42 --DS REDDIT-12K --r 70 --d 20
srun -p short --gres=gpu:a100:1 python main.py --lr 0.001 --hidden-dim 32 --num-gc-layers 4 --batch-size 4096 --epochs 20 --iter 5 --seed 42 --DS PPA --r 60 --d 20



# python main.py --lr 0.001 --hidden-dim 16 --num-gc-layers 4 --batch-size 512 --epochs 20 --iter 5 --seed 42 --DS ENZYMES --r 16 --d 7 --gpu 1
# python main.py --lr 0.001 --hidden-dim 16 --num-gc-layers 4 --batch-size 512 --epochs 20 --iter 5 --seed 42 --DS ENZYMES --r 16 --d 8 --gpu 1
# python main.py --lr 0.001 --hidden-dim 16 --num-gc-layers 4 --batch-size 512 --epochs 20 --iter 5 --seed 42 --DS ENZYMES --r 16 --d 9 --gpu 1
# python main.py --lr 0.001 --hidden-dim 16 --num-gc-layers 4 --batch-size 512 --epochs 20 --iter 5 --seed 42 --DS ENZYMES --r 16 --d 10 --gpu 1
# python main.py --lr 0.001 --hidden-dim 16 --num-gc-layers 4 --batch-size 512 --epochs 20 --iter 5 --seed 42 --DS ENZYMES --r 16 --d 11 --gpu 1
# python main.py --lr 0.001 --hidden-dim 16 --num-gc-layers 4 --batch-size 512 --epochs 20 --iter 5 --seed 42 --DS ENZYMES --r 16 --d 12 --gpu 1

