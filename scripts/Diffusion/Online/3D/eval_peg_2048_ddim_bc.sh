# Evaluate the 2048-point Peg DDIM/BC checkpoint on the real robot.
# Example:
# bash scripts/Diffusion/Online/3D/eval_peg_2048_ddim_bc.sh rl100 peg_2048 bc_eval 42
# To use another checkpoint:
# EVAL_CKPT=/absolute/path/to/latest.ckpt bash scripts/Diffusion/Online/3D/eval_peg_2048_ddim_bc.sh rl100 peg_2048 bc_eval 42

DEBUG=False
save_ckpt=False

alg_name=${1}
task_name=${2}
config_name='rl100_3d_epsilon'
addition_info=${3}
seed=${4}
ft_seed=${seed}
train_env_num=${5:-1}
exp_name=${task_name}-${alg_name}-${addition_info}

gpu_id=$(bash scripts/find_gpu.sh)
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=offline
    echo -e "\033[33mEval mode\033[0m"
fi

cd RL-100

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}
export CUDA_LAUNCH_BLOCKING=1
export MUJOCO_EGL_DEVICE_ID=${gpu_id}
export EGL_DEVICE_ID=${gpu_id}

act=${act:-'mish'}
encoder_type=${encoder_type:-'dp3vib'}
model=${model:-'dp3'}

N_OBS_STEPS=${N_OBS_STEPS:-3}
N_ACTION_STEPS=${N_ACTION_STEPS:-16}
HORIZON=${HORIZON:-$((N_ACTION_STEPS + N_OBS_STEPS - 1))}

# train_real.py expects <stage1_run_dir>/checkpoints/latest.ckpt.  Keep the
# original checkpoint untouched and expose it through a private symlink.
EVAL_CKPT=${EVAL_CKPT:-data/latest.ckpt}
stage1_run_dir=${EVAL_RUN_DIR:-data/eval_peg_2048_ddim_bc}
run_dir="${stage1_run_dir}"
eval_ckpt=$(readlink -f "${EVAL_CKPT}")
ckpt_link="${stage1_run_dir}/checkpoints/latest.ckpt"

if [ ! -f "${eval_ckpt}" ]; then
    echo "Checkpoint not found: ${EVAL_CKPT}" >&2
    exit 1
fi
if [ -e "${ckpt_link}" ] && [ ! -L "${ckpt_link}" ]; then
    echo "Refusing to replace non-symlink checkpoint: ${ckpt_link}" >&2
    exit 1
fi
mkdir -p "$(dirname "${ckpt_link}")"
ln -sfn "${eval_ckpt}" "${ckpt_link}"

echo -e "\033[33mcheckpoint: ${eval_ckpt}\033[0m"
echo -e "\033[33meval directory: ${stage1_run_dir}\033[0m"

python train_real.py --config-name=${config_name}.yaml \
    task=${task_name} \
    hydra.run.dir=${run_dir} \
    training.debug=$DEBUG \
    training.seed=${ft_seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    checkpoint.save_ckpt=${save_ckpt} \
    training.resume=True \
    eval=True \
    +use_checkpoint_normalizer=True \
    online=False \
    offline=False \
    policy._target_=rl_100.policy.rl100_3d.RL1003D \
    policy.ddim_noise_scheduler.num_train_timesteps=100 \
    policy.cm_noise_scheduler.num_train_timesteps=100 \
    horizon=${HORIZON} \
    n_action_steps=${N_ACTION_STEPS} \
    n_obs_steps=${N_OBS_STEPS} \
    num_inference_steps=10 \
    policy.model=${model} \
    policy.encoder_type=${encoder_type} \
    policy.act=${act} \
    policy.encoder_output_dim=64 \
    policy.diffusion_step_embed_dim=256 \
    policy.down_dims="[256,512,1024]" \
    policy.scheduler_type='ddim' \
    policy.use_vib=True \
    policy.use_recon=True \
    policy.beta_kl=1e-4 \
    policy.img_shape=[3,84,84] \
    policy.use_agent_pos=True \
    task.env_runner.env_num=1 \
    task.env_runner.eval_episodes=10 \
    task.env_runner.fake_env=False \
    ++ppo.use_vec_env_online=False \
    ++ppo.eval_env_num=${train_env_num} \
    critic.is_iql=False \
    critic.load_pretrain=False \
    chunk_as_single_action=True \
    gamma=0.99 \
    dataloader.num_workers=0 \
    val_dataloader.num_workers=0
