# Peg online RL without distillation.
# Based directly on train_policy_online_flow_chunk.sh.
# Example:
# bash scripts/Flow/Online/3D/train_peg_online_rl.sh rl100 peg peg_1024 42



DEBUG=False
save_ckpt=True

alg_name=${1}
task_name=${2}
config_name='rl100_3d_flow'
addition_info=${3}
seed=${4}
ft_seed=${seed}
train_env_num=${5:-1}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"


gpu_id=$(bash scripts/find_gpu.sh)
# gpu_id=${5}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"


if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=offline
    echo -e "\033[33mTrain mode\033[0m"
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
CRITIC_STRIDE=${CRITIC_STRIDE:-${N_ACTION_STEPS}}
FINETUNE_STRIDE=${FINETUNE_STRIDE:-${N_ACTION_STEPS}}
USE_ACTION_EMBED=${USE_ACTION_EMBED:-False}
USE_CONV_ACTION_EMBED=${USE_CONV_ACTION_EMBED:-False}
Q_HIDDEN_DIM=${Q_HIDDEN_DIM:-1024}
V_HIDDEN_DIM=${V_HIDDEN_DIM:-512}
DYNAMICS_HIDDEN_DIMS=${DYNAMICS_HIDDEN_DIMS:-"[1024,1024,512,512]"}
DYNAMICS_WEIGHT_DECAY=${DYNAMICS_WEIGHT_DECAY:-"[2.5e-5,5.0e-5,7.5e-5,7.5e-5,1.0e-4]"}
CRITIC_Q_LAYER_NORM=${CRITIC_Q_LAYER_NORM:-True}
CRITIC_ACTION_EMBED_LAYER_NORM=${CRITIC_ACTION_EMBED_LAYER_NORM:-False}
CRITIC_ACTION_SCALE_NORM=${CRITIC_ACTION_SCALE_NORM:-False}
DYNAMICS_ACTION_EMBED_LAYER_NORM=${DYNAMICS_ACTION_EMBED_LAYER_NORM:-False}
DYNAMICS_ACTION_SCALE_NORM=${DYNAMICS_ACTION_SCALE_NORM:-False}
CONV_HIDDEN_DIMS=${CONV_HIDDEN_DIMS:-"[128,256]"}
CONV_LATENT_CZ=${CONV_LATENT_CZ:-32}
CONV_KERNEL_SIZE=${CONV_KERNEL_SIZE:-5}
CONV_N_GROUPS=${CONV_N_GROUPS:-8}
ACTION_RECON_BETA=${ACTION_RECON_BETA:-0.5}

stage1_run_dir="data/outputs_two_stage_chunk_flow/${exp_name}_seed${seed}/${act}/${encoder_type}/${model}"
run_dir="${stage1_run_dir}"
critic_artifact_dir="${stage1_run_dir}/critic_c${CRITIC_STRIDE}_f${FINETUNE_STRIDE}"

echo -e "\033[33mstage1_run_dir: ${stage1_run_dir}\033[0m"
echo -e "\033[33mcritic_artifact_dir: ${critic_artifact_dir}\033[0m"


for lr_a in 2e-6; do
    for K_epochs in 5; do
            python train_real.py --config-name=${config_name}.yaml \
                task=${task_name} \
                hydra.run.dir=${run_dir} \
                training.debug=$DEBUG \
                training.seed=${ft_seed} \
                training.device="cuda:0" \
                exp_name=${exp_name} \
                logging.mode=${wandb_mode} \
                checkpoint.save_ckpt=${save_ckpt} \
                unio4.bppo_lr=3e-6 \
                unio4.rollout_length=30 \
                training.resume=True \
                policy._target_=rl_100.policy.rl100_3d.RL1003D \
                policy.ddim_noise_scheduler.num_train_timesteps=100 \
                policy.cm_noise_scheduler.num_train_timesteps=100 \
                use_action_embed=${USE_ACTION_EMBED} \
                use_conv_action_embed=${USE_CONV_ACTION_EMBED} \
                conv_hidden_dims=${CONV_HIDDEN_DIMS} \
                conv_latent_cz=${CONV_LATENT_CZ} \
                conv_kernel_size=${CONV_KERNEL_SIZE} \
                conv_n_groups=${CONV_N_GROUPS} \
                action_recon_beta=${ACTION_RECON_BETA} \
                horizon=${HORIZON} \
                n_action_steps=${N_ACTION_STEPS} \
                n_obs_steps=${N_OBS_STEPS} \
                ft_all_actions=False \
                num_inference_steps=10 \
                flow_inference_steps=10 \
                flow_distill_inference_steps=1 \
                flow_distill_teacher_steps=10 \
                flow_sde_type='cps' \
                flow_cps_logprob_mode='gaussian' \
                flow_noise_level=0.7 \
                flow_sde_window_size=0 \
                flow_logit_normal_sampling=False \
                flow_noise_on_final_step=True \
                policy.model=$model \
                policy.encoder_type=$encoder_type \
                policy.act=$act \
                unio4.bppo_steps=5000 \
                +unio4.stage1_resume_dir="${stage1_run_dir}" \
                +unio4.critic_artifact_dir="${critic_artifact_dir}" \
                +unio4.global_best_dir="${stage1_run_dir}/best" \
                online=True \
                policy.encoder_output_dim=64 \
                policy.diffusion_step_embed_dim=256 \
                policy.down_dims="[256,512,1024]" \
                policy.scheduler_type='flow' \
                critic.load_pretrain=True \
                ppo.lr_a=${lr_a} \
                ppo.K_epochs=${K_epochs} \
                ppo.lr_c=3e-4 \
                ppo.mini_batch_size=128 \
                ppo.batch_size=128 \
                task.env_runner.env_num=1 \
                task.env_runner.eval_episodes=1 \
                task.env_runner.fake_env=False \
                ++ppo.use_vec_env_online=False \
                ++ppo.train_env_num=${train_env_num} \
                ++ppo.eval_env_num=${train_env_num} \
                ppo.share_encoder=False \
                unio4.eval_times=1 \
                ppo.fix_encoder=False \
                ppo.max_train_steps=1000000 \
                policy.img_shape=[3,84,84] \
                policy.use_agent_pos=True \
                distill_phase=null \
                update_phase='step' \
                policy.mlp_policy_depth=3 \
                ppo.save_online_cp=True \
                ppo.online_cp_save_freq=1 \
                distill2mean=False \
                load_bc=True \
                clip_std_max=0.1 \
                ppo.load_online_cp=True \
                ppo.iql_ft=False \
                ppo.idql_eval=False \
                dataloader.num_workers=0 \
                val_dataloader.num_workers=0 \
                policy.use_vib=True \
                policy.use_recon=True \
                dynamics_type='mlp' \
                dynamics.dynamics_hidden_dims=${DYNAMICS_HIDDEN_DIMS} \
                dynamics.dynamics_weight_decay=${DYNAMICS_WEIGHT_DECAY} \
                dynamics.action_embed_layer_norm=${DYNAMICS_ACTION_EMBED_LAYER_NORM} \
                dynamics.action_scale_norm=${DYNAMICS_ACTION_SCALE_NORM} \
                dynamics.prediction_mode='full' \
                task.critic_dataset.sequence_stride=${CRITIC_STRIDE} \
                task.finetune_dataset.sequence_stride=${FINETUNE_STRIDE} \
                critic.omega=0.9 \
                critic.gamma=0.997 \
                critic.q_hidden_dim=${Q_HIDDEN_DIM} \
                critic.v_hidden_dim=${V_HIDDEN_DIM} \
                critic.q_layer_norm=${CRITIC_Q_LAYER_NORM} \
                critic.action_embed_layer_norm=${CRITIC_ACTION_EMBED_LAYER_NORM} \
                critic.action_scale_norm=${CRITIC_ACTION_SCALE_NORM} \
                ppo.online_iql_recon=True \
                ppo.fix_iql_encoder=False \
                ppo.is_share_iql_encoder=False \
                ppo.iql_encoder_update_with='q' \
                policy.joint_opt_encoder=False \
                predict_r=True \
                chunk_as_single_action=True \
                gamma=0.997 \
                ppo.recon=True \
                ppo.value_recon=True \
                ppo.per_step_recon=True \
                ppo.force_stochastic_online=True \
                policy.beta_kl=1e-3
    done
done
