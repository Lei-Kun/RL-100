#!/usr/bin/env bash
set -euo pipefail

# Two-stage launcher for 2D policy:
# 1) Run a single DDP stage-1 pass to produce BC / IQL / dynamics artifacts.
# 2) Fan out BPPO sweep jobs, each in its own timestamped run_dir while sharing
#    the same stage-1 artifacts and a locked global best under the root run dir.
#
# Example:
#   bash scripts/train_policy_image_unet_two_stage_flow.sh rl100 adroit_door_medium 0112 100 4

DEBUG=False
save_ckpt=True

alg_name=${1:?alg_name is required}
task_name=${2:?task_name is required}
addition_info=${3:?addition_info is required}
seed=${4:?seed is required}
NUM_GPUS=${5:-4}

if [ -z "${PRE_IMAGE_NORM:-}" ]; then
    if [ "${task_name}" = "peg_2d" ]; then
        PRE_IMAGE_NORM=False
    else
        PRE_IMAGE_NORM=True
    fi
fi

config_name='rl100_2d_flow'
exp_name=${task_name}-${alg_name}-flow-${addition_info}

# Stage switches
RUN_STAGE1=${RUN_STAGE1:-true}
RUN_SWEEP=${RUN_SWEEP:-true}
# Stage-2 sweep jobs are one process per GPU. Running them through torchrun's
# elastic rendezvous adds a failure mode without adding distributed training.
SWEEP_ENTRY=${SWEEP_ENTRY:-train_ddp.py}
SWEEP_USE_TORCHRUN=${SWEEP_USE_TORCHRUN:-false}

# Sweep grid
LR_VALUES=${LR_VALUES:-"1e-6 1.42e-6 2.66e-6 8.66e-6"}
ROLLOUT_VALUES=${ROLLOUT_VALUES:-"3 5 10 12"}
CLIP_STD_MAX_VALUES=${CLIP_STD_MAX_VALUES:-"null"}
FLOW_NOISE_LEVEL_VALUES=${FLOW_NOISE_LEVEL_VALUES:-"0.7"}

# Stage-1 should only materialize BC / IQL / dynamics. With bppo_steps=0 it still
# enters finetune_dp3(), but performs only the initial evaluation / save path.
STAGE1_BPPO_STEPS=${STAGE1_BPPO_STEPS:-0}
STAGE1_BPPO_LR=${STAGE1_BPPO_LR:-1e-5}
STAGE1_ROLLOUT_LENGTH=${STAGE1_ROLLOUT_LENGTH:-10}
STAGE1_CLIP_STD_MAX=${STAGE1_CLIP_STD_MAX:-0.1}
STAGE1_FLOW_NOISE_LEVEL=${STAGE1_FLOW_NOISE_LEVEL:-0.7}
STAGE1_EVAL_EPISODES=${STAGE1_EVAL_EPISODES:-30}
STAGE1_EVAL_TIMES=${STAGE1_EVAL_TIMES:-30}

BASE_MASTER_PORT=${MASTER_PORT:-29518}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

port_is_open() {
    local port=$1
    (echo >"/dev/tcp/127.0.0.1/${port}") >/dev/null 2>&1
}

find_free_port() {
    local start_port=$1
    local candidate=$start_port
    while port_is_open "${candidate}"; do
        candidate=$((candidate + 1))
    done
    echo "${candidate}"
}

if [ "$NUM_GPUS" -gt 1 ]; then
    gpu_list=$(bash scripts/find_gpus.sh "$NUM_GPUS")
    if [ $? -ne 0 ]; then
        echo "Failed to find $NUM_GPUS available GPUs" >&2
        exit 1
    fi
else
    gpu_id=$(bash scripts/find_gpu.sh)
    gpu_list=$gpu_id
fi

if [ "$DEBUG" = True ]; then
    wandb_mode=offline
else
    wandb_mode=offline
fi

MASTER_PORT=$(find_free_port "${BASE_MASTER_PORT}")
if [ "${MASTER_PORT}" != "${BASE_MASTER_PORT}" ]; then
    printf 'MASTER_PORT %s is busy, using %s instead\n' "${BASE_MASTER_PORT}" "${MASTER_PORT}"
fi

cd RL-100

export HYDRA_FULL_ERROR=1
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=${gpu_list}
export MUJOCO_EGL_DEVICE_ID=$(echo "${gpu_list}" | cut -d',' -f1)
export CUDA_LAUNCH_BLOCKING=1

act='mish'
encoder_type='resnet'
model='skipnet'
encoder_tag="${encoder_type}"

root_run_dir="data/outputs_2d_flow_two_stage/${exp_name}_seed${seed}/${act}/${encoder_tag}/${model}"
stage1_run_dir="${root_run_dir}"
sweep_root_dir="${root_run_dir}"

mkdir -p "${sweep_root_dir}"

get_common_params() {
    local run_dir=$1
    local lr=$2
    local rollout_length=$3
    local clip_std_max=$4
    local flow_noise_level=$5

    echo "task=${task_name} \
        hydra.run.dir=${run_dir} \
        training.debug=$DEBUG \
        training.seed=${seed} \
        training.device=cuda:0 \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        checkpoint.save_ckpt=${save_ckpt} \
        unio4.bppo_lr=${lr} \
        unio4.rollout_length=${rollout_length} \
        policy._target_=rl_100.policy.rl100_2d.RL1002D \
        policy.ddim_noise_scheduler.num_train_timesteps=100 \
        training.resume=True \
        use_action_embed=False \
        horizon=3 \
        n_action_steps=1 \
        n_obs_steps=3 \
        ft_all_actions=False \
        unio4.bppo_steps=6000 \
        num_inference_steps=10 \
        flow_inference_steps=10 \
        flow_sde_type='cps' \
        flow_noise_level=${flow_noise_level} \
        flow_sde_window_size=0 \
        flow_logit_normal_sampling=False \
        flow_noise_on_final_step=True \
        flow_cps_logprob_mode='gaussian' \
        only_bc=True \
        offline=True \
        use_agent_pos=True \
        policy.use_visual=True \
        policy.model=$model \
        policy.act=$act \
        policy.mlp_policy_depth=3 \
        feature_type='2D' \
        policy.scheduler_type='flow' \
        encoder_output_dim=64 \
        policy.down_dims='[256,512,1024]' \
        task.env_runner.eval_episodes=30 \
        task.env_runner.env_num=1 \
        ++task.env_runner.with_pointcloud=False \
        policy.use_aug=True \
        policy.img_shape=[3,224,224] \
        task.dataset.pre_image_norm=${PRE_IMAGE_NORM} \
        use_recon=True \
        use_vib=True \
        dynamics_type='diffusion' \
        training.num_epochs=800 \
        training.num_critic_epochs=600 \
        dynamics.dynamics_max_epochs=350 \
        dataloader.batch_size=32 \
        val_dataloader.batch_size=32 \
        ppo.enable_ratio_logging=true \
        ppo.ratio_log_every_updates=10 \
        ppo.ratio_plot_on_final_flush=true \
        optimizer.lr=2e-4 \
        critic.q_lr=2e-4 \
        critic.v_lr=2e-4 \
        dynamics.dynamics_lr=4.4e-4 \
        clip_std_max=${clip_std_max} \
        encoder_type='resnet' \
        encoders.resnet.share_rgb_model=False \
        encoders.resnet.rgb_model.weights='r3m' \
        distill_phase=null \
        distill2mean=True \
        distill_loss_type='action_same_noise' \
        encoders.resnet.recon_loss_weight=0.05 \
        encoders.resnet.kl_beta=5e-4 \
        offline_use_aug=False \
        kl_annealing=False"
}

stage1_complete() {
    local run_dir=$1
    if [ ! -f "${run_dir}/checkpoints/latest.ckpt" ]; then
        return 1
    fi
    if [ ! -f "${run_dir}/Q_bc_20.pt" ]; then
        return 1
    fi
    if [ ! -f "${run_dir}/value_20.pt" ]; then
        return 1
    fi
    if ! compgen -G "${run_dir}/saved_models*/dynamics.pth" > /dev/null; then
        return 1
    fi
    return 0
}

run_stage1() {
    local params
    params=$(get_common_params "${stage1_run_dir}" "${STAGE1_BPPO_LR}" "${STAGE1_ROLLOUT_LENGTH}" "${STAGE1_CLIP_STD_MAX}" "${STAGE1_FLOW_NOISE_LEVEL}")

    echo "=== Stage 1: materialize BC / IQL / dynamics artifacts ==="
    echo "run_dir=${stage1_run_dir}"
    torchrun --nproc_per_node="${NUM_GPUS}" \
             --master_port="${MASTER_PORT}" \
             train_ddp.py --config-name="${config_name}.yaml" \
             ${params} \
             unio4.bppo_steps="${STAGE1_BPPO_STEPS}" \
             unio4.eval_times="${STAGE1_EVAL_TIMES}" \
             task.env_runner.eval_episodes="${STAGE1_EVAL_EPISODES}"
}

run_sweep_job() {
    local combo_idx=$1
    local gpu=$2
    local lr=$3
    local rollout_length=$4
    local clip_std_max=$5
    local flow_noise_level=$6

    local timestamp
    local sweep_run_dir
    local params

    timestamp=$(date +"%Y-%m-%d-%H-%M-%S")
    sweep_run_dir="${sweep_root_dir}/${timestamp}-combo_${combo_idx}-lr_${lr}_rollout_${rollout_length}_clip_${clip_std_max}_flow_${flow_noise_level}"
    mkdir -p "${sweep_run_dir}"
    params=$(get_common_params "${sweep_run_dir}" "${lr}" "${rollout_length}" "${clip_std_max}" "${flow_noise_level}")

    echo "[GPU ${gpu}] Starting sweep job: lr=${lr}, rollout=${rollout_length}, clip_std_max=${clip_std_max}, flow_noise_level=${flow_noise_level}"
    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export MUJOCO_EGL_DEVICE_ID=${gpu}
        if [ "${SWEEP_USE_TORCHRUN}" = "true" ] || [ "${SWEEP_USE_TORCHRUN}" = "True" ]; then
            exec torchrun --nproc_per_node=1 \
                 --master_port="$((MASTER_PORT + combo_idx + 1))" \
                 train_ddp.py --config-name="${config_name}.yaml" \
                 ${params} \
                 +unio4.stage1_resume_dir="${stage1_run_dir}" \
                 +unio4.global_best_dir="${root_run_dir}/best"
        else
            exec "${PYTHON:-python}" "${SWEEP_ENTRY}" --config-name="${config_name}.yaml" \
                 ${params} \
                 +unio4.stage1_resume_dir="${stage1_run_dir}" \
                 +unio4.global_best_dir="${root_run_dir}/best"
        fi
    )
}

if [ "${RUN_STAGE1}" = true ] || [ "${RUN_STAGE1}" = "True" ]; then
    if stage1_complete "${stage1_run_dir}"; then
        echo "=== Stage 1 artifacts already exist, skipping stage 1 ==="
        echo "stage1_run_dir=${stage1_run_dir}"
    else
        run_stage1
    fi
fi

if ! stage1_complete "${stage1_run_dir}"; then
    echo "Stage 1 artifacts are incomplete under ${stage1_run_dir}" >&2
    exit 1
fi

if [ "${RUN_SWEEP}" = true ] || [ "${RUN_SWEEP}" = "True" ]; then
    echo "=== Stage 2: BPPO sweep ==="
    IFS=',' read -ra GPU_ARRAY <<< "${gpu_list}"
    num_available_gpus=${#GPU_ARRAY[@]}

    declare -a param_combinations=()
    for lr in ${LR_VALUES}; do
        for rollout_length in ${ROLLOUT_VALUES}; do
            for clip_std_max in ${CLIP_STD_MAX_VALUES}; do
                for flow_noise_level in ${FLOW_NOISE_LEVEL_VALUES}; do
                    param_combinations+=("${lr}:${rollout_length}:${clip_std_max}:${flow_noise_level}")
                done
            done
        done
    done

    echo "total parameter combinations: ${#param_combinations[@]}"
    echo "available GPUs: ${num_available_gpus} (${gpu_list})"

    declare -a batch_pids=()
    batch_slot=0
    combo_idx=0
    sweep_failed=0

    wait_for_batch() {
        local pid
        local status
        for pid in "${batch_pids[@]}"; do
            set +e
            wait "${pid}"
            status=$?
            set -e
            if [ "${status}" -ne 0 ]; then
                echo "Sweep job pid=${pid} failed with exit code ${status}" >&2
                sweep_failed=1
            fi
        done
        batch_pids=()
    }

    for combo in "${param_combinations[@]}"; do
        IFS=':' read -r lr rollout_length clip_std_max flow_noise_level <<< "${combo}"
        gpu=${GPU_ARRAY[$batch_slot]}

        run_sweep_job "${combo_idx}" "${gpu}" "${lr}" "${rollout_length}" "${clip_std_max}" "${flow_noise_level}" &
        batch_pids+=($!)

        combo_idx=$((combo_idx + 1))
        batch_slot=$((batch_slot + 1))

        if [ "${batch_slot}" -eq "${num_available_gpus}" ]; then
            wait_for_batch
            batch_slot=0
        fi
    done

    if [ "${#batch_pids[@]}" -gt 0 ]; then
        wait_for_batch
    fi

    if [ "${sweep_failed}" -ne 0 ]; then
        echo "One or more BPPO sweep jobs failed" >&2
        exit 1
    fi

    echo "=== Two-stage sweep completed ==="
fi
