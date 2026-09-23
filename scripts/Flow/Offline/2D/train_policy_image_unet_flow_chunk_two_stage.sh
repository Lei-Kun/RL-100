#!/usr/bin/env bash
set -euo pipefail

# Two-stage launcher for 2D chunk policy:
# 1) Run one stage-1 pass to produce BC / IQL / dynamics artifacts.
# 2) Fan out BPPO sweep jobs, each in its own timestamped run_dir while sharing
#    the same stage-1 artifacts and a global best under the root run dir.
#
# Example:
#   bash scripts/train_policy_image_unet_chunk_two_stage.sh rl100 adroit_door_medium 0112 100 1

DEBUG=${DEBUG:-False}
save_ckpt=${save_ckpt:-True}

alg_name=${1:?alg_name is required}
task_name=${2:?task_name is required}
addition_info=${3:?addition_info is required}
seed=${4:?seed is required}
NUM_GPUS=${5:-1}

if [ "${task_name}" = "peg_2d" ]; then
    PRE_IMAGE_NORM=${PRE_IMAGE_NORM:-False}
    BATCH_SIZE=${BATCH_SIZE:-32}
    VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-32}
else
    PRE_IMAGE_NORM=${PRE_IMAGE_NORM:-True}
    BATCH_SIZE=${BATCH_SIZE:-512}
    VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-512}
fi

config_name=${config_name:-'rl100_2d_flow'}
exp_name=${task_name}-${alg_name}-${addition_info}

RUN_STAGE1=${RUN_STAGE1:-true}
RUN_SWEEP=${RUN_SWEEP:-true}
STAGE1_ENTRY=${STAGE1_ENTRY:-train_ddp.py}
# Stage-2 sweep jobs are one process per GPU. Running them through torchrun's
# elastic rendezvous adds a failure mode without adding distributed training.
SWEEP_ENTRY=${SWEEP_ENTRY:-${STAGE1_ENTRY}}
SWEEP_USE_TORCHRUN=${SWEEP_USE_TORCHRUN:-false}

GPU_LIST=${GPU_LIST:-""}
# Direct-run defaults:
# compare the current main line against the Plan B chunk-level vdelta scalar advantage
LR_VALUES=${LR_VALUES:-"1e-6 1.42e-6 2.83e-6"}
ROLLOUT_VALUES=${ROLLOUT_VALUES:-"3 5 10"}
CLIP_STD_MAX_VALUES=${CLIP_STD_MAX_VALUES:-"0.1 null"}
CHUNK_ADV_CLIP_VALUES=${CHUNK_ADV_CLIP_VALUES:-"null"}
CHUNK_LOSS_MODE_COMBOS=${CHUNK_LOSS_MODE_COMBOS:-"scalar:scalar_iql"}
CHUNK_VDELTA_GAE_N_ROLLOUT=${CHUNK_VDELTA_GAE_N_ROLLOUT:-3}
CHUNK_VDELTA_GAE_LAMBDA=${CHUNK_VDELTA_GAE_LAMBDA:-0.95}
CHUNK_VDELTA_GAE_CHUNK_SOURCE=${CHUNK_VDELTA_GAE_CHUNK_SOURCE:-repeat_first}
USE_CONV_ACTION_EMBED=${USE_CONV_ACTION_EMBED:-False}
USE_ACTION_EMBED=${USE_ACTION_EMBED:-False}
Q_HIDDEN_DIM=${Q_HIDDEN_DIM:-1024}
V_HIDDEN_DIM=${V_HIDDEN_DIM:-512}
DYNAMICS_HIDDEN_DIMS=${DYNAMICS_HIDDEN_DIMS:-"[1024,1024,512,512]"}
DYNAMICS_WEIGHT_DECAY=${DYNAMICS_WEIGHT_DECAY:-"[2.5e-5,5.0e-5,7.5e-5,7.5e-5,1.0e-4]"}
DYNAMICS_MAX_EPOCHS_SINCE_UPDATE=${DYNAMICS_MAX_EPOCHS_SINCE_UPDATE:-50}
CRITIC_ACTION_SCALE_NORM=${CRITIC_ACTION_SCALE_NORM:-False}
DYNAMICS_ACTION_SCALE_NORM=${DYNAMICS_ACTION_SCALE_NORM:-False}
CRITIC_Q_LAYER_NORM_VALUES=${CRITIC_Q_LAYER_NORM_VALUES:-"True"}
if [ "${USE_CONV_ACTION_EMBED}" = "True" ]; then
    CRITIC_ACTION_EMBED_LAYER_NORM_VALUES=${CRITIC_ACTION_EMBED_LAYER_NORM_VALUES:-"False"}
    DYNAMICS_ACTION_EMBED_LAYER_NORM_VALUES=${DYNAMICS_ACTION_EMBED_LAYER_NORM_VALUES:-"False"}
else
    CRITIC_ACTION_EMBED_LAYER_NORM_VALUES=${CRITIC_ACTION_EMBED_LAYER_NORM_VALUES:-"False"}
    DYNAMICS_ACTION_EMBED_LAYER_NORM_VALUES=${DYNAMICS_ACTION_EMBED_LAYER_NORM_VALUES:-"False"}
fi

# Chunk geometry
N_OBS_STEPS=${N_OBS_STEPS:-3}
N_ACTION_STEPS=${N_ACTION_STEPS:-16}
HORIZON=${HORIZON:-$((N_ACTION_STEPS + N_OBS_STEPS - 1))}

# Stride overrides for offline chunk boundary experiments
CRITIC_STRIDE=${CRITIC_STRIDE:-${N_ACTION_STEPS}}
FINETUNE_STRIDE=${FINETUNE_STRIDE:-${N_ACTION_STEPS}}

# Conv1d AE action encoder settings
CONV_HIDDEN_DIMS=${CONV_HIDDEN_DIMS:-"[128,256]"}
CONV_LATENT_CZ=${CONV_LATENT_CZ:-32}
CONV_KERNEL_SIZE=${CONV_KERNEL_SIZE:-5}
CONV_N_GROUPS=${CONV_N_GROUPS:-8}
ACTION_RECON_BETA=${ACTION_RECON_BETA:-0.5}

STAGE1_CRITIC_Q_LAYER_NORM=$(echo ${CRITIC_Q_LAYER_NORM_VALUES} | awk '{print $1}')
STAGE1_CRITIC_ACTION_EMBED_LAYER_NORM=$(echo ${CRITIC_ACTION_EMBED_LAYER_NORM_VALUES} | awk '{print $1}')
STAGE1_DYNAMICS_ACTION_EMBED_LAYER_NORM=$(echo ${DYNAMICS_ACTION_EMBED_LAYER_NORM_VALUES} | awk '{print $1}')

STAGE1_BPPO_STEPS=${STAGE1_BPPO_STEPS:-0}
STAGE1_BPPO_LR=${STAGE1_BPPO_LR:-1e-6}
STAGE1_ROLLOUT_LENGTH=${STAGE1_ROLLOUT_LENGTH:-3}
STAGE1_CLIP_STD_MAX=${STAGE1_CLIP_STD_MAX:-0.1}
STAGE1_EVAL_TIMES=${STAGE1_EVAL_TIMES:-1}
STAGE1_EVAL_EPISODES=${STAGE1_EVAL_EPISODES:-5}

MASTER_PORT=${MASTER_PORT:-29518}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -n "${GPU_LIST}" ]; then
    gpu_list=${GPU_LIST}
else
    if [ "${NUM_GPUS}" -gt 1 ]; then
        gpu_list=$(bash scripts/find_gpus.sh "${NUM_GPUS}")
        if [ $? -ne 0 ]; then
            echo "Failed to find ${NUM_GPUS} available GPUs" >&2
            exit 1
        fi
    else
        gpu_id=$(bash scripts/find_gpu.sh)
        gpu_list=${gpu_id}
    fi
fi

echo "gpu ids (to use): ${gpu_list}"
echo "resolved strides: critic=${CRITIC_STRIDE}, finetune=${FINETUNE_STRIDE}, n_action_steps=${N_ACTION_STEPS}"

if [ "${DEBUG}" = True ]; then
    wandb_mode=offline
else
    wandb_mode=offline
fi

cd RL-100

export HYDRA_FULL_ERROR=1
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=${gpu_list}
export MUJOCO_EGL_DEVICE_ID=$(echo "${gpu_list}" | cut -d',' -f1)
export CUDA_LAUNCH_BLOCKING=1

act=${act:-'mish'}
model=${model:-'dp3'}
encoder_type=${encoder_type:-'resnet'}

root_run_dir="data/outputs_2d_flow_chunk/${exp_name}_seed${seed}/${act}/${encoder_type}/${model}"
stage1_run_dir="${root_run_dir}"
sweep_root_dir="${root_run_dir}"

# Stride-aware critic artifact directory
critic_artifact_dir="${root_run_dir}/critic_c${CRITIC_STRIDE}_f${FINETUNE_STRIDE}"

mkdir -p "${root_run_dir}"
mkdir -p "${critic_artifact_dir}"

get_common_params() {
    local run_dir=$1
    local lr=$2
    local rollout_length=$3
    local clip_std_max=$4
    local device=$5
    local chunk_adv_clip=${6:-null}
    local offline_chunk_ratio_mode=${7:-scalar}
    local offline_chunk_adv_mode=${8:-scalar_iql}
    local critic_q_layer_norm=${9:-False}
    local critic_action_embed_layer_norm=${10:-False}
    local dynamics_action_embed_layer_norm=${11:-False}

    echo "task=${task_name} \
        hydra.run.dir=${run_dir} \
        training.debug=${DEBUG} \
        training.seed=${seed} \
        training.device=${device} \
        exp_name=${exp_name} \
        logging.mode=${wandb_mode} \
        checkpoint.save_ckpt=${save_ckpt} \
        unio4.bppo_lr=${lr} \
        unio4.rollout_length=${rollout_length} \
        clip_std_max=${clip_std_max} \
        policy._target_=rl_100.policy.rl100_2d.RL1002D \
        policy.ddim_noise_scheduler.num_train_timesteps=100 \
        training.resume=True \
        use_action_embed=${USE_ACTION_EMBED} \
        horizon=${HORIZON} \
        n_action_steps=${N_ACTION_STEPS} \
        n_obs_steps=${N_OBS_STEPS} \
        ft_all_actions=False \
        num_inference_steps=10 \
        flow_inference_steps=10 \
        flow_sde_type='cps' \
        flow_noise_level=0.7 \
        flow_sde_window_size=0 \
        flow_logit_normal_sampling=False \
        flow_noise_on_final_step=True \
        flow_cps_logprob_mode='gaussian' \
        unio4.bppo_steps=6000 \
        offline=True \
        use_agent_pos=True \
        policy.use_visual=True \
        policy.model=${model} \
        policy.act=${act} \
        policy.mlp_policy_depth=3 \
        feature_type='2D' \
        policy.scheduler_type='flow' \
        encoder_output_dim=64 \
        policy.down_dims=[256,512,1024] \
        task.env_runner.eval_episodes=30 \
        task.env_runner.env_num=1 \
        ++task.env_runner.with_pointcloud=False \
        policy.use_aug=True \
        critic.omega=0.9 \
        critic.gamma=0.997 \
        policy.img_shape=[3,224,224] \
        task.dataset.pre_image_norm=${PRE_IMAGE_NORM} \
        ++task.critic_dataset.pre_image_norm=${PRE_IMAGE_NORM} \
        ++task.finetune_dataset.pre_image_norm=${PRE_IMAGE_NORM} \
        use_recon=True \
        use_vib=True \
        dynamics_type='diffusion' \
        dynamics.prediction_mode='full' \
        training.num_epochs=800 \
        training.num_critic_epochs=600 \
        dynamics.dynamics_max_epochs=350 \
        dataloader.batch_size=${BATCH_SIZE} \
        val_dataloader.batch_size=${VAL_BATCH_SIZE} \
        ppo.enable_ratio_logging=true \
        ppo.ratio_log_every_updates=10 \
        ppo.ratio_plot_on_final_flush=true \
        optimizer.lr=2e-4 \
        critic.q_lr=2e-4 \
        critic.v_lr=2e-4 \
        dynamics.dynamics_lr=4.4e-4 \
        encoder_type='resnet' \
        encoders.resnet.share_rgb_model=False \
        encoders.resnet.rgb_model.weights='r3m' \
        distill_phase=null \
        distill2mean=True \
        distill_loss_type='action_same_noise' \
        encoders.resnet.recon_loss_weight=0.05 \
        kl_annealing=False \
        offline_use_aug=False \
        encoders.resnet.kl_beta=5e-4 \
        chunk_as_single_action=True \
        bppo_chunk_level_ratio=True \
        chunk_adv_clip=${chunk_adv_clip} \
        offline_chunk_ratio_mode=${offline_chunk_ratio_mode} \
        offline_chunk_adv_mode=${offline_chunk_adv_mode} \
        critic.q_layer_norm=${critic_q_layer_norm} \
        critic.action_embed_layer_norm=${critic_action_embed_layer_norm} \
        critic.action_scale_norm=${CRITIC_ACTION_SCALE_NORM} \
        critic.q_hidden_dim=${Q_HIDDEN_DIM} \
        critic.v_hidden_dim=${V_HIDDEN_DIM} \
        dynamics.action_embed_layer_norm=${dynamics_action_embed_layer_norm} \
        dynamics.action_scale_norm=${DYNAMICS_ACTION_SCALE_NORM} \
        dynamics.dynamics_hidden_dims=${DYNAMICS_HIDDEN_DIMS} \
        dynamics.dynamics_weight_decay=${DYNAMICS_WEIGHT_DECAY} \
        predict_r=True \
        task.critic_dataset.sequence_stride=${CRITIC_STRIDE} \
        task.finetune_dataset.sequence_stride=${FINETUNE_STRIDE} \
        use_conv_action_embed=${USE_CONV_ACTION_EMBED} \
        conv_hidden_dims=${CONV_HIDDEN_DIMS} \
        conv_latent_cz=${CONV_LATENT_CZ} \
        conv_kernel_size=${CONV_KERNEL_SIZE} \
        conv_n_groups=${CONV_N_GROUPS} \
        action_recon_beta=${ACTION_RECON_BETA} \
        chunk_vdelta_gae_n_rollout=${CHUNK_VDELTA_GAE_N_ROLLOUT} \
        chunk_vdelta_gae_lambda=${CHUNK_VDELTA_GAE_LAMBDA} \
        chunk_vdelta_gae_chunk_source=${CHUNK_VDELTA_GAE_CHUNK_SOURCE}"
}

base_stage1_complete() {
    # Check BC checkpoint + dynamics (shared across stride settings)
    local run_dir=$1
    if [ ! -f "${run_dir}/checkpoints/latest.ckpt" ]; then
        return 1
    fi
    if ! compgen -G "${run_dir}/saved_models*/dynamics.pth" > /dev/null; then
        return 1
    fi
    return 0
}

critic_stage1_complete() {
    # Check critic/value artifacts (stride-specific)
    local critic_dir=$1
    if [ ! -f "${critic_dir}/Q_bc_20.pt" ]; then
        return 1
    fi
    if [ ! -f "${critic_dir}/value_20.pt" ]; then
        return 1
    fi
    return 0
}

stage1_complete() {
    # Both base artifacts and stride-specific critic artifacts must exist
    local run_dir=$1
    local critic_dir=$2
    base_stage1_complete "${run_dir}" && critic_stage1_complete "${critic_dir}"
}

run_stage1() {
    local params
    local stage1_chunk_adv_clip
    local stage1_mode_combo
    local stage1_ratio_mode
    local stage1_adv_mode

    stage1_chunk_adv_clip=$(echo ${CHUNK_ADV_CLIP_VALUES} | awk '{print $1}')
    stage1_mode_combo=$(echo ${CHUNK_LOSS_MODE_COMBOS} | awk '{print $1}')
    IFS=':' read -r stage1_ratio_mode stage1_adv_mode <<< "${stage1_mode_combo}"

    params=$(get_common_params "${stage1_run_dir}" "${STAGE1_BPPO_LR}" "${STAGE1_ROLLOUT_LENGTH}" "${STAGE1_CLIP_STD_MAX}" "cuda:0" "${stage1_chunk_adv_clip}" "${stage1_ratio_mode}" "${stage1_adv_mode}" "${STAGE1_CRITIC_Q_LAYER_NORM}" "${STAGE1_CRITIC_ACTION_EMBED_LAYER_NORM}" "${STAGE1_DYNAMICS_ACTION_EMBED_LAYER_NORM}")
    local primary_gpu
    local stage1_gpu_count

    primary_gpu="$(echo "${gpu_list}" | cut -d',' -f1)"
    IFS=',' read -ra STAGE1_GPU_ARRAY <<< "${gpu_list}"
    stage1_gpu_count=${#STAGE1_GPU_ARRAY[@]}

    echo "=== Stage 1: materialize BC / IQL / dynamics artifacts ==="
    echo "stage1_run_dir=${stage1_run_dir}"
    echo "stage1_entry=${STAGE1_ENTRY}"
    echo "critic_artifact_dir=${critic_artifact_dir}"
    echo "stage1 critic/dynamics knobs: q_ln=${STAGE1_CRITIC_Q_LAYER_NORM}, action_ln=${STAGE1_CRITIC_ACTION_EMBED_LAYER_NORM}, dyn_action_ln=${STAGE1_DYNAMICS_ACTION_EMBED_LAYER_NORM}, ratio_mode=${stage1_ratio_mode}, adv_mode=${stage1_adv_mode}"
    if [ "${STAGE1_ENTRY}" = "train_ddp.py" ]; then
        echo "stage1_gpu_count=${stage1_gpu_count} (${gpu_list})"
        export CUDA_VISIBLE_DEVICES="${gpu_list}"
        torchrun --standalone --nproc_per_node="${stage1_gpu_count}" "${STAGE1_ENTRY}" \
            --config-name="${config_name}.yaml" \
            ${params} \
            unio4.bppo_steps="${STAGE1_BPPO_STEPS}" \
            unio4.eval_times="${STAGE1_EVAL_TIMES}" \
            task.env_runner.eval_episodes="${STAGE1_EVAL_EPISODES}" \
            +unio4.critic_artifact_dir="${critic_artifact_dir}"
    else
        export CUDA_VISIBLE_DEVICES="${primary_gpu}"
        "${PYTHON:-python}" "${STAGE1_ENTRY}" --config-name="${config_name}.yaml" \
            ${params} \
            unio4.bppo_steps="${STAGE1_BPPO_STEPS}" \
            unio4.eval_times="${STAGE1_EVAL_TIMES}" \
            task.env_runner.eval_episodes="${STAGE1_EVAL_EPISODES}" \
            +unio4.critic_artifact_dir="${critic_artifact_dir}"
    fi
}

run_sweep_job() {
    local gpu=$1
    local lr=$2
    local rollout_length=$3
    local clip_std_max=$4
    local chunk_adv_clip=$5
    local offline_chunk_ratio_mode=$6
    local offline_chunk_adv_mode=$7
    local critic_q_layer_norm=$8
    local critic_action_embed_layer_norm=$9
    local dynamics_action_embed_layer_norm=${10}

    local timestamp
    timestamp=$(date +"%Y-%m-%d-%H-%M-%S")
    local sweep_run_dir="${sweep_root_dir}/${timestamp}-lr_${lr}_rollout_${rollout_length}_clip_${clip_std_max}_advclip_${chunk_adv_clip}_rmode_${offline_chunk_ratio_mode}_amode_${offline_chunk_adv_mode}_qln_${critic_q_layer_norm}_aln_${critic_action_embed_layer_norm}_dln_${dynamics_action_embed_layer_norm}"

    local params
    mkdir -p "${sweep_run_dir}"
    params=$(get_common_params "${sweep_run_dir}" "${lr}" "${rollout_length}" "${clip_std_max}" "cuda:0" "${chunk_adv_clip}" "${offline_chunk_ratio_mode}" "${offline_chunk_adv_mode}" "${critic_q_layer_norm}" "${critic_action_embed_layer_norm}" "${dynamics_action_embed_layer_norm}")

    echo "[GPU ${gpu}] Starting sweep job: lr=${lr}, rollout=${rollout_length}, clip_std_max=${clip_std_max}, chunk_adv_clip=${chunk_adv_clip}, ratio_mode=${offline_chunk_ratio_mode}, adv_mode=${offline_chunk_adv_mode}, q_ln=${critic_q_layer_norm}, action_ln=${critic_action_embed_layer_norm}, dyn_action_ln=${dynamics_action_embed_layer_norm}"
    local predict_r_override=""
    if [ "${offline_chunk_adv_mode}" = "per_step_vdelta" ] || [ "${offline_chunk_adv_mode}" = "chunk_vdelta_scalar" ] || [ "${offline_chunk_adv_mode}" = "chunk_vdelta_gae" ]; then
        predict_r_override="predict_r=True"
    fi
    (
        export CUDA_VISIBLE_DEVICES=${gpu}
        export MUJOCO_EGL_DEVICE_ID=${gpu}
        if [ "${SWEEP_USE_TORCHRUN}" = "true" ] || [ "${SWEEP_USE_TORCHRUN}" = "True" ]; then
            exec torchrun --standalone --nproc_per_node=1 "${STAGE1_ENTRY}" \
                --config-name="${config_name}.yaml" \
                ${params} \
                +unio4.stage1_resume_dir="${stage1_run_dir}" \
                +unio4.critic_artifact_dir="${critic_artifact_dir}" \
                +unio4.global_best_dir="${root_run_dir}/best" \
                ${predict_r_override}
        else
            exec "${PYTHON:-python}" "${SWEEP_ENTRY}" --config-name="${config_name}.yaml" \
                ${params} \
                +unio4.stage1_resume_dir="${stage1_run_dir}" \
                +unio4.critic_artifact_dir="${critic_artifact_dir}" \
                +unio4.global_best_dir="${root_run_dir}/best" \
                ${predict_r_override}
        fi
    )
}

if [ "${RUN_STAGE1}" = true ] || [ "${RUN_STAGE1}" = "True" ]; then
    if stage1_complete "${stage1_run_dir}" "${critic_artifact_dir}"; then
        echo "=== Stage 1 artifacts already exist, skipping stage 1 ==="
        echo "stage1_run_dir=${stage1_run_dir}"
        echo "critic_artifact_dir=${critic_artifact_dir}"
    else
        run_stage1
    fi
fi

if ! stage1_complete "${stage1_run_dir}" "${critic_artifact_dir}"; then
    echo "Stage 1 artifacts are incomplete under ${stage1_run_dir} / ${critic_artifact_dir}" >&2
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
                for chunk_adv_clip in ${CHUNK_ADV_CLIP_VALUES}; do
                    for mode_combo in ${CHUNK_LOSS_MODE_COMBOS}; do
                        IFS=':' read -r offline_chunk_ratio_mode offline_chunk_adv_mode <<< "${mode_combo}"
                        for critic_q_layer_norm in ${CRITIC_Q_LAYER_NORM_VALUES}; do
                            for critic_action_embed_layer_norm in ${CRITIC_ACTION_EMBED_LAYER_NORM_VALUES}; do
                                for dynamics_action_embed_layer_norm in ${DYNAMICS_ACTION_EMBED_LAYER_NORM_VALUES}; do
                                    param_combinations+=("${lr}:${rollout_length}:${clip_std_max}:${chunk_adv_clip}:${mode_combo}:${critic_q_layer_norm}:${critic_action_embed_layer_norm}:${dynamics_action_embed_layer_norm}")
                                done
                            done
                        done
                    done
                done
            done
        done
    done

    echo "total parameter combinations: ${#param_combinations[@]}"
    echo "available GPUs: ${num_available_gpus} (${gpu_list})"

    declare -a batch_pids=()
    batch_slot=0
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
        IFS=':' read -r lr rollout_length clip_std_max chunk_adv_clip offline_chunk_ratio_mode offline_chunk_adv_mode critic_q_layer_norm critic_action_embed_layer_norm dynamics_action_embed_layer_norm <<< "${combo}"
        gpu=${GPU_ARRAY[$batch_slot]}

        run_sweep_job "${gpu}" "${lr}" "${rollout_length}" "${clip_std_max}" "${chunk_adv_clip}" "${offline_chunk_ratio_mode}" "${offline_chunk_adv_mode}" "${critic_q_layer_norm}" "${critic_action_embed_layer_norm}" "${dynamics_action_embed_layer_norm}" &
        batch_pids+=($!)
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

    echo "=== 2D chunk two-stage sweep completed ==="
fi
