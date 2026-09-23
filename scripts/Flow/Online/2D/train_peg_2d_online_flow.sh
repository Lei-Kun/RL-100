#!/usr/bin/env bash
set -euo pipefail

# DDP distill + real-robot Peg online Flow training with an RL1002D image policy.
# The offline stage must use the same geometry: horizon=3, n_obs_steps=3,
# n_action_steps=1.
#
# Example:
# ARTIFACT_DIR=data/outputs_2d_flow_two_stage/peg_2d-rl100-flow-bc_seed42/mish/resnet/skipnet \
# PEG_2D_ZARR=/data/hdd2/xarm_teleop/data/data_peg.zarr \
# bash scripts/Flow/Online/2D/train_peg_2d_online_flow.sh rl100 peg_2d bc 42

DEBUG=${DEBUG:-False}
save_ckpt=${save_ckpt:-True}

alg_name=${1:?alg_name is required}
task_name=${2:-peg_2d}
addition_info=${3:?addition_info is required}
seed=${4:?seed is required}

config_name='rl100_2d_flow'
exp_name=${task_name}-${alg_name}-flow-${addition_info}

act=${act:-mish}
model=${model:-skipnet}
encoder_type=${encoder_type:-resnet}

DISTILL_GPUS=${DISTILL_GPUS:-}
DISTILL_GPU_COUNT=${DISTILL_GPU_COUNT:-4}
ONLINE_GPU=${ONLINE_GPU:-${GPU_ID:-}}

N_OBS_STEPS=${N_OBS_STEPS:-3}
N_ACTION_STEPS=${N_ACTION_STEPS:-1}
HORIZON=${HORIZON:-3}

if [ "$((HORIZON - N_OBS_STEPS + 1))" -ne "${N_ACTION_STEPS}" ]; then
    echo "Expected horizon - n_obs_steps + 1 == n_action_steps" >&2
    exit 1
fi

PEG_2D_ZARR=${PEG_2D_ZARR:-/data/hdd2/xarm_teleop/data/data_peg.zarr}
stage1_run_dir=${ARTIFACT_DIR:-data/outputs_2d_flow_two_stage/${exp_name}_seed${seed}/${act}/${encoder_type}/${model}}
critic_artifact_dir=${CRITIC_ARTIFACT_DIR:-${stage1_run_dir}}
run_dir=${RUN_DIR:-${stage1_run_dir}}

if [ ! -d "${PEG_2D_ZARR}" ]; then
    echo "RGB Peg dataset not found: ${PEG_2D_ZARR}" >&2
    echo "The dataset must contain data/external_img and data/wrist_img." >&2
    exit 1
fi

export PEG_2D_ZARR
python - <<'PY'
import os
import numpy as np
import zarr

path = os.environ['PEG_2D_ZARR']
root = zarr.open(path, mode='r')
required = {
    'external_img', 'wrist_img', 'state', 'action', 'point_cloud',
    'reward', 'done', 'timeout', 'return',
}
data = root['data']
missing = sorted(required.difference(data.keys()))
if missing:
    raise SystemExit(f"RGB Peg dataset is missing keys: {missing}")
n_frames = len(data['action'])
bad_lengths = {key: len(data[key]) for key in required if len(data[key]) != n_frames}
if bad_lengths:
    raise SystemExit(f"RGB Peg dataset has inconsistent lengths: expected {n_frames}, got {bad_lengths}")
episode_ends = np.asarray(root['meta']['episode_ends'])
if not len(episode_ends) or episode_ends[-1] != n_frames or np.any(np.diff(episode_ends) <= 0):
    raise SystemExit(f"Invalid episode_ends for {n_frames} frames: {episode_ends}")
for key in ('external_img', 'wrist_img'):
    array = data[key]
    if array.ndim != 4 or tuple(array.shape[1:]) != (224, 224, 3):
        raise SystemExit(f"data/{key} must have shape (N,224,224,3), got {array.shape}")
    sample = np.asarray(array[:min(len(array), 8)])
    if array.dtype != np.uint8 or sample.min() < 0 or sample.max() > 255:
        raise SystemExit(
            f"data/{key} must be uint8 RGB in [0,255], got "
            f"dtype={array.dtype}, range=[{sample.min()}, {sample.max()}]"
        )
PY

artifact_check=${stage1_run_dir}
if [[ "${artifact_check}" != /* ]]; then
    artifact_check="RL-100/${artifact_check}"
fi
if [ ! -f "${artifact_check}/checkpoints/latest.ckpt" ]; then
    echo "Offline 2D stage-1 checkpoint not found: ${artifact_check}/checkpoints/latest.ckpt" >&2
    echo "Run scripts/Flow/Offline/2D/train_policy_image_unet_two_stage_flow.sh for task=${task_name} first." >&2
    exit 1
fi

distilled_check=${artifact_check}/best/last/distilled_model.pt
if [ -z "${DISTILL_GPUS}" ] && [ ! -f "${distilled_check}" ]; then
    DISTILL_GPUS=$(bash scripts/find_gpus.sh "${DISTILL_GPU_COUNT}")
elif [ -z "${DISTILL_GPUS}" ]; then
    DISTILL_GPUS=${ONLINE_GPU:-$(bash scripts/find_gpu.sh)}
fi
if [ -z "${ONLINE_GPU}" ]; then
    ONLINE_GPU=$(echo "${DISTILL_GPUS}" | cut -d',' -f1)
fi
IFS=',' read -r -a distill_gpu_ids <<< "${DISTILL_GPUS}"
distill_gpu_count=${#distill_gpu_ids[@]}

echo -e "\033[33mdistill gpus: ${DISTILL_GPUS} (count=${distill_gpu_count})\033[0m"
echo -e "\033[33monline gpu: ${ONLINE_GPU}\033[0m"
echo -e "\033[33mRGB dataset: ${PEG_2D_ZARR}\033[0m"
echo -e "\033[33mstage1 artifacts: ${stage1_run_dir}\033[0m"

cd RL-100

export HYDRA_FULL_ERROR=1
export HF_ENDPOINT="https://hf-mirror.com"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

COMMON_ARGS=(
    task=${task_name}
    hydra.run.dir=${run_dir}
    training.debug=${DEBUG}
    training.seed=${seed}
    exp_name=${exp_name}
    logging.mode=offline
    checkpoint.save_ckpt=${save_ckpt}
    training.resume=True
    policy._target_=rl_100.policy.rl100_2d.RL1002D
    horizon=${HORIZON}
    n_action_steps=${N_ACTION_STEPS}
    n_obs_steps=${N_OBS_STEPS}
    ft_all_actions=False
    use_action_embed=False
    num_inference_steps=10
    flow_inference_steps=10
    flow_distill_inference_steps=1
    flow_distill_teacher_steps=10
    flow_sde_type=cps
    flow_cps_logprob_mode=gaussian
    flow_noise_level=0.7
    flow_sde_window_size=0
    flow_logit_normal_sampling=False
    flow_noise_on_final_step=True
    policy.scheduler_type=flow
    policy.model=${model}
    policy.act=${act}
    policy.encoder_output_dim=64
    "policy.down_dims=[256,512,1024]"
    policy.mlp_policy_depth=3
    policy.use_visual=True
    policy.use_aug=False
    "policy.img_shape=[3,224,224]"
    policy.joint_opt_encoder=False
    feature_type=2D
    use_agent_pos=True
    encoder_type=${encoder_type}
    encoders.resnet.share_rgb_model=False
    encoders.resnet.rgb_model.weights=r3m
    online=True
    offline=False
    +unio4.stage1_resume_dir=${stage1_run_dir}
    +unio4.critic_artifact_dir=${critic_artifact_dir}
    +unio4.global_best_dir=${stage1_run_dir}/best
    unio4.bppo_lr=3e-6
    unio4.rollout_length=30
    unio4.bppo_steps=5000
    unio4.eval_times=1
    critic.load_pretrain=True
    ppo.lr_a=4.4e-6
    ppo.K_epochs=5
    ppo.lr_c=3e-4
    ppo.mini_batch_size=128
    ppo.batch_size=128
    ++ppo.use_vec_env_online=False
    ++ppo.train_env_num=1
    ++ppo.eval_env_num=1
    ppo.fix_encoder=False
    ppo.share_encoder=False
    ppo.max_train_steps=1000000
    ppo.save_online_cp=True
    ppo.online_cp_save_freq=1
    ppo.load_online_cp=False
    ppo.iql_ft=False
    ppo.idql_eval=False
    ppo.online_iql_recon=True
    ppo.fix_iql_encoder=False
    ppo.is_share_iql_encoder=False
    ppo.iql_encoder_update_with=q
    ppo.per_step_recon=False
    ppo.recon=False
    ppo.value_recon=False
    ppo.force_stochastic_online=True
    task.env_runner.env_num=1
    task.env_runner.eval_episodes=1
    task.env_runner.with_image=True
    task.env_runner.with_pointcloud=False
    task.norm_dataset.zarr_path=${PEG_2D_ZARR}
    task.dataset.zarr_path=${PEG_2D_ZARR}
    task.critic_dataset.zarr_path=${PEG_2D_ZARR}
    task.finetune_dataset.zarr_path=${PEG_2D_ZARR}
    task.scale_dataset.zarr_path=${PEG_2D_ZARR}
    distill_phase=online
    distill_loss_type=action_same_noise
    distill2mean=True
    update_phase=step
    load_bc=False
    clip_std_max=0.1
    use_vib=True
    use_recon=True
    dynamics_type=diffusion
    encoders.resnet.kl_beta=0.05
    encoders.resnet.recon_loss_weight=0.1
    dataloader.num_workers=0
    val_dataloader.num_workers=0
)

offline_distilled_path=${stage1_run_dir}/best/last/distilled_model.pt
if [ ! -f "${offline_distilled_path}" ]; then
    echo -e "\033[33moffline distilled model missing; starting DDP distill\033[0m"
    export CUDA_VISIBLE_DEVICES=${DISTILL_GPUS}
    unset CUDA_LAUNCH_BLOCKING
    export MUJOCO_EGL_DEVICE_ID=0
    export EGL_DEVICE_ID=0
    DISTILL_CMD=(
        torchrun
        --standalone
        --nproc_per_node=${distill_gpu_count}
        train_ddp.py
        --config-name=${config_name}.yaml
        training.device=cuda:0
        task.env_runner.fake_env=True
    )
    DISTILL_CMD+=("${COMMON_ARGS[@]}")
    "${DISTILL_CMD[@]}"
else
    echo -e "\033[32mfound offline distilled model; skipping DDP distill\033[0m"
fi

if [ ! -f "${offline_distilled_path}" ]; then
    echo "DDP distill did not create ${offline_distilled_path}" >&2
    exit 1
fi

echo -e "\033[33mstarting real-robot online training on GPU ${ONLINE_GPU}\033[0m"
export CUDA_VISIBLE_DEVICES=${ONLINE_GPU}
export CUDA_LAUNCH_BLOCKING=1
export MUJOCO_EGL_DEVICE_ID=0
export EGL_DEVICE_ID=0
ONLINE_CMD=(
    python
    train_real.py
    --config-name=${config_name}.yaml
    training.device=cuda:0
    task.env_runner.fake_env=False
)
ONLINE_CMD+=("${COMMON_ARGS[@]}")
"${ONLINE_CMD[@]}"
