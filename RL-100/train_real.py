if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)
import argparse
import os
import hydra
import torch
import dill
import pickle
import inspect
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
from copy import deepcopy
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
import fcntl
from hydra.core.hydra_config import HydraConfig
from rl_100.policy.rl100_3d import RL1003D
from rl_100.dataset.base_dataset import BaseDataset
from rl_100.env_runner.base_runner import BaseRunner
from rl_100.common.checkpoint_util import TopKCheckpointManager
from rl_100.common.pytorch_util import dict_apply, optimizer_to
from rl_100.model.diffusion.ema_model import EMAModel
from rl_100.model.common.lr_scheduler import get_scheduler
from rl_100.unidpg.transition_model.configs import loaded_args
from rl_100.unidpg.dynamics_eval_batch import dynamics_eval, train_dynamics
from rl_100.unidpg.uni_ppo import BehaviorProximalPolicyOptimization
from rl_100.unidpg.critic import IQL_Q_V_no, IQL_Q_V_online
from rl_100.unidpg.critic import ValueLearner
from rl_100.unidpg.net_online import ValueLearner_online
from collections import deque
from rl_100.model.common.cm_util import update_ema
import glob
OmegaConf.register_new_resolver("eval", eval, replace=True)

# Snapshot/restore RNG so the iql_ft branch consumes random state without
# leaking it into the subsequent PPO update. Enabled by default.
_IQLFT_RESTORE_RNG = os.environ.get("IQLFT_RESTORE_RNG_AFTER_IQL", "1") == "1"


def _save_online_reward_curves(path, chunk_rewards, episode_returns, episode_successes):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 8))
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 1)

    axes[0].plot(np.arange(1, len(chunk_rewards) + 1), chunk_rewards)
    axes[0].set_ylabel('chunk reward')

    episode_x = np.arange(1, len(episode_returns) + 1)
    return_mean_10 = [
        np.mean(episode_returns[max(0, i - 9):i + 1])
        for i in range(len(episode_returns))
    ]
    success_mean_10 = [
        np.mean(episode_successes[max(0, i - 9):i + 1])
        for i in range(len(episode_successes))
    ]
    axes[1].plot(episode_x, episode_returns, label='return')
    axes[1].plot(episode_x, return_mean_10, label='mean (10)')
    axes[1].set_ylabel('episode return')
    axes[1].legend()
    axes[2].plot(episode_x, episode_successes, label='success')
    axes[2].plot(episode_x, success_mean_10, label='rate (10)')
    axes[2].set_ylabel('success')
    axes[2].set_xlabel('episode')
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)


def _iqlft_snapshot_rng():
    snap = {
        "torch_cpu": torch.random.get_rng_state().clone(),
        "numpy": np.random.get_state(legacy=True),
        "random": random.getstate(),
    }
    if torch.cuda.is_available():
        snap["cuda_all"] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    return snap


def _iqlft_restore_rng(snap):
    torch.random.set_rng_state(snap["torch_cpu"])
    if torch.cuda.is_available() and "cuda_all" in snap:
        torch.cuda.set_rng_state_all(snap["cuda_all"])
    np.random.set_state(snap["numpy"])
    random.setstate(snap["random"])


# Snapshot/restore RNG around the iql_online constructor (and its checkpoint
# load) so that the extra IQL network does not perturb torch-cpu / cuda /
# numpy / random RNG. Enabled by the main IQL restore switch, or explicitly
# with IQLFT_RESTORE_RNG_AT_CONSTRUCT=1.
_IQLFT_RESTORE_RNG_AT_CONSTRUCT = (
    _IQLFT_RESTORE_RNG
    or
    os.environ.get("IQLFT_RESTORE_RNG_AT_CONSTRUCT", "0") == "1"
)

import warnings
warnings.filterwarnings("ignore")
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
import pprint
os.environ["WANDB_CONSOLE"] = "off"  # Or "silent" to suppress more messages
os.environ["WANDB_SILENT"] = "true"

def init_wandb_run(cfg, output_dir):
    logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
    init_timeout = int(logging_cfg.pop("init_timeout", 120))
    retry_init_timeout = int(logging_cfg.pop("retry_init_timeout", max(init_timeout * 2, 300)))
    settings_cfg = logging_cfg.pop("settings", {}) or {}

    def _wandb_init(timeout):
        settings = dict(settings_cfg)
        settings["init_timeout"] = timeout
        return wandb.init(
            dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            settings=wandb.Settings(**settings),
            **logging_cfg
        )

    try:
        return _wandb_init(init_timeout)
    except wandb.errors.CommError as exc:
        if "timeout" not in str(exc).lower():
            raise
        cprint(
            f"[WandB] init timed out after {init_timeout}s, retrying with {retry_init_timeout}s",
            "yellow",
        )
        return _wandb_init(retry_init_timeout)

class TrainDP3Workspace:
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        cfg.ppo.num_inference_steps = cfg.policy.num_inference_steps
        if getattr(cfg.ppo, 'iql_adv', False):
            raise ValueError(
                'ppo.iql_adv=True is no longer supported after removing '
                'dp_align_update_iql_no_share; use the default PPO path instead.'
            )

        # os.environ["CUDA_VISIBLE_DEVICES"] = "2"#.format(cfg.training.gpu_id)
        self.cfg = cfg
        self._output_dir = output_dir
        self._saving_thread = None
        print('Training workspace initialized 1')
        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # self.output_dir = self.output_dir()
        self.device = cfg.training.device
        # configure model
        self.model: RL1003D = hydra.utils.instantiate(cfg.policy)

        # load pretrained 2D encoder if configured
        if getattr(cfg, 'use_pretrained_2DEncoder', False):
            if 'channel' in cfg.policy.stage1_model_name:
                print('load pretrained encoder')
                self.model.obs_encoder.load_pretrained_encoder(
                    self.get_pretrained_model_path(cfg.policy.stage1_model_name), device=self.device)
                self.model.obs_encoder.switch_to_RL_stages()

        self.ema_model: RL1003D = None
        if cfg.training.use_ema:
            try:
                self.ema_model = copy.deepcopy(self.model)
            except Exception: # minkowski engine could not be copied. recreate it
                self.ema_model = hydra.utils.instantiate(cfg.policy)
        # unio4 for finetuning dp3
        self.unio4 = BehaviorProximalPolicyOptimization(
            policy=self.model,
            device=torch.device(cfg.training.device),
            policy_lr=cfg.unio4.bppo_lr,
            clip_ratio=cfg.unio4.clip_ratio,
            entropy_weight=cfg.unio4.entropy_weight,
            decay=cfg.unio4.decay,
            omega=cfg.unio4.omega,
            batch_size=cfg.unio4.bppo_batch_size,
            is_iql=cfg.critic.is_iql,
            temperature=cfg.unio4.temperature,
            ratio_strategy=cfg.unio4.ratio_strategy,
            top_k=cfg.unio4.top_k,
            num_inference_steps=cfg.policy.num_inference_steps,
            fix_encoder=cfg.unio4.fix_encoder,
            cfg=cfg,
        )

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def get_stage1_artifact_dir(self):
        return self.cfg.unio4.get('stage1_resume_dir', None) or self.output_dir

    def get_critic_artifact_dir(self):
        """Return the directory for critic/value/encoder artifacts.
        When chunk_as_single_action uses a stride-specific critic directory,
        keep critic artifacts separate from the shared stage1 BC/dynamics dir.
        This applies to both offline training and later online loading."""
        stage1_dir = self.get_stage1_artifact_dir()
        if self.cfg.chunk_as_single_action:
            explicit_dir = self.cfg.unio4.get('critic_artifact_dir', None)
            if explicit_dir:
                return explicit_dir

            inferred_dir = os.path.join(
                stage1_dir,
                f'critic_c{self.cfg.n_action_steps}_f{self.cfg.n_action_steps}',
            )
            if os.path.exists(os.path.join(inferred_dir, 'Q_bc_20.pt')):
                return inferred_dir
        return stage1_dir

    def get_stage1_checkpoint_path(self, tag='latest'):
        return pathlib.Path(self.get_stage1_artifact_dir()).joinpath('checkpoints', f'{tag}.ckpt')

    def get_global_best_dir(self):
        return self.cfg.unio4.get('global_best_dir', None) or os.path.join(self.output_dir, 'best')

    def get_global_best_ema_dir(self):
        return self.cfg.unio4.get('global_best_ema_dir', None) or os.path.join(self.output_dir, 'best_ema')

    def get_global_best_score_path(self):
        return os.path.join(self.get_global_best_dir(), 'best_score.csv')

    def get_global_best_ema_score_path(self):
        return os.path.join(self.get_global_best_ema_dir(), 'best_score.csv')

    def get_global_best_lock_path(self):
        best_dir = self.get_global_best_dir()
        return os.path.join(os.path.dirname(best_dir), '.global_best.lock')

    def get_global_best_ema_lock_path(self):
        best_dir = self.get_global_best_ema_dir()
        return os.path.join(os.path.dirname(best_dir), '.global_best_ema.lock')

    def _read_best_score(self, score_path):
        if os.path.exists(score_path):
            best_score = np.loadtxt(score_path, delimiter=',')
            if isinstance(best_score, np.ndarray):
                best_score = float(np.asarray(best_score).reshape(-1)[0])
            else:
                best_score = float(best_score)
            return best_score
        return float('-inf')

    def _maybe_update_best(self, score, best_dir, best_score_path, lock_path, save_fn, eval_name):
        os.makedirs(os.path.dirname(best_dir), exist_ok=True)
        with open(lock_path, 'a+') as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            best_saved_scores = self._read_best_score(best_score_path)
            is_updated = score > best_saved_scores
            if is_updated:
                os.makedirs(best_dir, exist_ok=True)
                save_fn(best_dir)
                np.savetxt(best_score_path, [score], fmt='%f', delimiter=',')
                meta_path = os.path.join(best_dir, 'best_meta.txt')
                with open(meta_path, 'w') as f:
                    f.write(f"score: {score}\n")
                    f.write(f"eval_name: {eval_name}\n")
                    f.write(f"source_run_dir: {self.output_dir}\n")
                    f.write(f"timestamp_dir: {self.unio4_output_dir}\n")
                    f.write(f"seed: {self.cfg.training.seed}\n")
                    f.write(f"rollout_length: {self.cfg.unio4.rollout_length}\n")
                    f.write(f"bppo_lr: {self.cfg.unio4.bppo_lr}\n")
            else:
                score = best_saved_scores
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        return score, is_updated

    def maybe_update_global_best(self, score):
        return self._maybe_update_best(
            score=score,
            best_dir=self.get_global_best_dir(),
            best_score_path=self.get_global_best_score_path(),
            lock_path=self.get_global_best_lock_path(),
            save_fn=self.unio4.save,
            eval_name='Policy Eval',
        )

    def maybe_update_global_best_ema(self, score):
        if self.ema_model is None:
            return float('-inf'), False
        return self._maybe_update_best(
            score=score,
            best_dir=self.get_global_best_ema_dir(),
            best_score_path=self.get_global_best_ema_score_path(),
            lock_path=self.get_global_best_ema_lock_path(),
            save_fn=self.ema_model.save,
            eval_name='EMA Eval',
        )

    def get_online_best_ema_dir(self):
        return os.path.join(self.output_dir, 'online_best_ema')

    def get_online_best_ema_score_path(self):
        return os.path.join(self.get_online_best_ema_dir(), 'best_score.csv')

    def get_online_best_ema_lock_path(self):
        return os.path.join(os.path.dirname(self.get_online_best_ema_dir()), '.online_best_ema.lock')

    def maybe_update_online_best_ema(self, score):
        if self.ema_model is None:
            return float('-inf'), False
        return self._maybe_update_best(
            score=score,
            best_dir=self.get_online_best_ema_dir(),
            best_score_path=self.get_online_best_ema_score_path(),
            lock_path=self.get_online_best_ema_lock_path(),
            save_fn=self.ema_model.save,
            eval_name='Online EMA Eval',
        )

        print('Training workspace initialized 2')

    def run(self):
        # args = parse_args()
        cfg = copy.deepcopy(self.cfg)

        # Flow mode: validate distill_phase
        if cfg.distill_phase is not None and getattr(self.model, 'is_flow', False):
            if cfg.distill_phase not in ('after_dp', 'after_offline', 'online'):
                raise RuntimeError(f"Unsupported distill_phase='{cfg.distill_phase}' for flow mode.")

        copy_encoder = deepcopy(self.model.obs_encoder)

        if cfg.training.debug:
            cfg.training.num_epochs = 100
            cfg.training.max_train_steps = 10
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 20
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1
            RUN_ROLLOUT = True
            RUN_CKPT = False
            verbose = True
        else:
            RUN_ROLLOUT = True
            RUN_CKPT = True
            verbose = False
        
        self.verbose = verbose
        RUN_VALIDATION = False # reduce time cost
        self.RUN_ROLLOUT = RUN_ROLLOUT
        self.RUN_VALIDATION = RUN_VALIDATION
        self.output_dir = self.output_dir()

        self.unio4_output_dir = os.path.join(self.output_dir, time.strftime("%Y-%m-%d-%H-%M-%S"))
        # save config
        config = vars(cfg)  
        # Fix optimizer state for lr_scheduler if needed
        for group in self.optimizer.param_groups:
            if 'initial_lr' not in group:
                group['initial_lr'] = group['lr']
        def write_dict(f, d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    f.write(f"{' ' * indent}{key}:\n")
                    write_dict(f, value, indent + 4)  
                else:
                    f.write(f"{' ' * indent}{key:20} : {value}\n")

        os.makedirs(self.unio4_output_dir, exist_ok=True)
        self.unio4.set_ratio_log_dir(os.path.join(self.unio4_output_dir, 'ratio_logs'))
        config_path = os.path.join(self.unio4_output_dir, 'config.txt')

        with open(config_path, 'w') as f:
            write_dict(f, config)
        print('====================================Here==================================')
        # resume training
        loaded_stage1_checkpoint = False
        if cfg.training.resume:
            lastest_ckpt_path = self.get_stage1_checkpoint_path(tag='latest')
            lastest_cm_ckpt_path = self.get_stage1_checkpoint_path(tag='latest_cm')
            if lastest_cm_ckpt_path.is_file():
                print(f"Resuming cm model from checkpoint {lastest_cm_ckpt_path}")
                # Create teacher/distilled_model sub-modules before loading
                # so the checkpoint's keys (teacher.*, distilled_model.*) are accepted
                if cfg.distill_phase is not None:
                    self.model.set_target()
                self.load_checkpoint(path=lastest_cm_ckpt_path)
                loaded_stage1_checkpoint = True
            elif lastest_ckpt_path.is_file():
                print(f"Resuming diffusion model from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                loaded_stage1_checkpoint = True
            else:
                print(f"No checkpoint found at {lastest_ckpt_path}")
        # device transfer
        device = torch.device(cfg.training.device)
        # configure dataset
        dataset: BaseDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        self.dataset = dataset  # Save reference for finetuning dataloader
        self.shape_info = dataset.get_shape_info(self.cfg.horizon - self.model.start, self.cfg.n_obs_steps)
        # import pdb
        # pdb.set_trace()
        assert isinstance(dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(dataset)}")
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        use_checkpoint_normalizer = bool(
            cfg.eval
            and cfg.get('use_checkpoint_normalizer', False)
            and loaded_stage1_checkpoint
        )
        if use_checkpoint_normalizer:
            normalizer = self.model.normalizer
            cprint('reuse the normalizer stored in the evaluation checkpoint', 'yellow')
        elif (self.cfg.off2off and self.cfg.off2off_no_bc) or self.cfg.use_pre_norm:
            norm_dataset = hydra.utils.instantiate(cfg.task.norm_dataset)
            normalizer = norm_dataset.get_normalizer()
            cprint('***********************************reuse the normalizer of pre-dataset***********************************', 'yellow')
            cprint('***********************************reuse the normalizer of pre-dataset***********************************', 'yellow')
            cprint('***********************************reuse the normalizer of pre-dataset***********************************', 'yellow')
        else:
            normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # Note: all_val_data removed to avoid OOM - now using batched val_dataloader for validation
        
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        self.env_runner = env_runner
        if env_runner is not None:
            assert isinstance(env_runner, BaseRunner)
        if self.cfg.use_wandb:
            cfg.logging.name = str(cfg.logging.name)
            cprint("-----------------------------", "yellow")
            cprint(f"[WandB] group: {cfg.logging.group}", "yellow")
            cprint(f"[WandB] name: {cfg.logging.name}", "yellow")
            cprint("-----------------------------", "yellow")
            # disable wandb logging
            import logging
            wandb_logger = logging.getLogger("wandb")
            wandb_logger.setLevel(logging.ERROR)
            # configure logging
            wandb_run = init_wandb_run(cfg, self.output_dir)
            wandb.config.update(
                {
                    "output_dir": self.output_dir,
                },
                allow_val_change=True
            )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )


        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        # import pdb; pdb.set_trace()
        # training loop
        latest_path = self.get_checkpoint_path(tag='latest')
        skip_diffusion_training = cfg.online and (self.cfg.training.resume == False)
        # ===============================stage 1-1: set for diffusion training ===============================
        if not os.path.exists(latest_path) or self.cfg.training.resume == False or (self.cfg.off2off and not self.cfg.off2off_no_bc):
            # VIB module beta kl anealling
            total_steps = cfg.training.num_epochs * len(train_dataloader)
            if hasattr(self.model.obs_encoder, 'beta_kl'):
                target_beta_kl = self.model.obs_encoder.beta_kl
            for local_epoch_idx in range(cfg.training.num_epochs):
                # KL annealing
                if cfg.kl_annealing and hasattr(self.model.obs_encoder, 'beta_kl'):
                    progress = local_epoch_idx / max(cfg.training.num_epochs - 1, 1)
                    self.model.obs_encoder.beta_kl = target_beta_kl * progress
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):

                        t1 = time.time()
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                    
                        # compute loss
                        t1_1 = time.time()
                        raw_loss, loss_dict = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()
                        
                        t1_2 = time.time()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        t1_3 = time.time()
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)
                        t1_4 = time.time()
                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        t1_5 = time.time()
                        step_log.update(loss_dict)
                        t2 = time.time()
                        
                        if verbose:
                            print(f"total one step time: {t2-t1:.3f}")
                            print(f" compute loss time: {t1_2-t1_1:.3f}")
                            print(f" step optimizer time: {t1_3-t1_2:.3f}")
                            print(f" update ema time: {t1_4-t1_3:.3f}")
                            print(f" logging time: {t1_5-t1_4:.3f}")

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and RUN_ROLLOUT and env_runner is not None:
                    t3 = time.time()
                    # runner_log = env_runner.run(policy, dataset=dataset)
                    runner_log = env_runner.run(policy)
                    t4 = time.time()
                    # print(f"rollout time: {t4-t3:.3f}")
                    # log all
                    step_log.update(runner_log)
                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and RUN_VALIDATION:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss, loss_dict = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        if self.cfg.no_pre_action:
                            gt_action = gt_action[:, self.cfg.n_obs_steps - 1 :]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                if env_runner is None:
                    step_log['test_mean_score'] = - train_loss
                    
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                    if cfg.only_bc:
                        self.unio4.set_policy(self.model); self.unio4.set_old_policy()
                        os.makedirs(os.path.join(self.output_dir, 'bc'), exist_ok=True)
                        self.unio4.save(os.path.join(self.output_dir, 'bc'))
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                self.global_step += 1
                self.epoch += 1
                del step_log
        if cfg.eval:
            log_data = self.eval(
                eval_times=self.cfg.unio4.eval_times,
                online=False,
                data_collect=self.cfg.data_collect,
            )
            return log_data['test_mean_score']

        log_data = self.eval(eval_times=self.cfg.unio4.eval_times, online=True, traj_path='/', data_collect=self.cfg.data_collect)
        self.offline_best_path = self.get_global_best_dir()
        self.offline_last_path = os.path.join(self.output_dir, 'last')
        # =============================== stage 1-1: end diffusion training ===============================
        if self.cfg.distill_phase == 'after_dp':
            self.distill2cm(train_dataloader, val_dataloader, wandb_run, env_runner, phase=self.cfg.distill_phase)
        # =============================== stage 1-3: set for critic and dynamics training ===============================
        # print('re-create critic dataset without validation data')
        self.train_dataloader = train_dataloader
        
        self.model.set_critic_normalizer(normalizer)
        self.model.to(device)
        # for Uni-O4 fine-tuning
        iql, Q_bc, value = self.model.initialize_critic(
            device=device,
            q_hidden_dim=self.cfg.critic.q_hidden_dim,
            q_depth=self.cfg.critic.q_depth,
            q_lr=self.cfg.critic.q_lr,
            target_update_freq=self.cfg.critic.target_update_freq,
            tau=self.cfg.critic.tau,
            gamma=self.cfg.critic.gamma,
            v_hidden_dim=self.cfg.critic.v_hidden_dim,
            v_depth=self.cfg.critic.v_depth,
            v_lr=self.cfg.critic.v_lr,
            omega=self.cfg.critic.omega,
            is_double_q=self.cfg.critic.is_double_q,
            is_iql=self.cfg.critic.is_iql,
            is_share_encoder=self.cfg.critic.is_share_encoder,
            use_action_embed=self.cfg.use_action_embed,
            fix_encoder=cfg.critic.fix_encoder,
            chunk_as_single_action=self.cfg.chunk_as_single_action,
            n_action_steps=self.cfg.n_action_steps,
            use_conv_action_embed=getattr(self.cfg, 'use_conv_action_embed', False),
            conv_hidden_dims=getattr(self.cfg, 'conv_hidden_dims', [128, 256]),
            conv_latent_cz=getattr(self.cfg, 'conv_latent_cz', 32),
            conv_kernel_size=getattr(self.cfg, 'conv_kernel_size', 5),
            conv_n_groups=getattr(self.cfg, 'conv_n_groups', 8),
            action_recon_beta=getattr(self.cfg, 'action_recon_beta', 0.5),
            q_layer_norm=getattr(self.cfg.critic, 'q_layer_norm', False),
            action_embed_layer_norm=getattr(self.cfg.critic, 'action_embed_layer_norm', False),
            action_scale_norm=getattr(self.cfg.critic, 'action_scale_norm', False),
            )
        stage1_artifact_dir = self.get_stage1_artifact_dir()
        critic_artifact_dir = self.get_critic_artifact_dir()
        os.makedirs(critic_artifact_dir, exist_ok=True)
        Q_bc_path = os.path.join(critic_artifact_dir, 'Q_bc_20.pt')
        value_path = os.path.join(critic_artifact_dir, 'value_20.pt')
        if self.cfg.critic.is_iql: 
            # Q_bc training
            if self.cfg.critic.is_share_encoder:   
                encoder_path = os.path.join(critic_artifact_dir, 'encoder.pt')
            else:
                encoder_path = None
            if os.path.exists(Q_bc_path):
                if self.cfg.critic.load_pretrain:
                    iql.load(Q_bc_path, value_path, encoder_path)
                    iql.eval()
                    iql.obs_encoder.eval()
            if not os.path.exists(Q_bc_path) or self.cfg.off2off:
                # --- Offline dataset role split for critic ---
                if self.cfg.offline and self.cfg.chunk_as_single_action and hasattr(self.cfg.task, 'critic_dataset'):
                    critic_dataset = hydra.utils.instantiate(self.cfg.task.critic_dataset)
                    cprint(f'Critic dataset: {len(critic_dataset)} samples '
                           f'(stride={getattr(critic_dataset, "sequence_stride", 1)})', 'cyan')
                    critic_dataloader = DataLoader(critic_dataset, **cfg.dataloader)
                else:
                    critic_dataloader = train_dataloader
                for local_epoch_idx in range(cfg.training.num_critic_epochs):
                    critic_step_log = dict()
                    # ========= train for this epoch ==========
                    q_train_losses, v_train_losses = list(), list()
                    with tqdm.tqdm(critic_dataloader, desc=f"Training epoch {self.epoch}",
                            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            Q_bc_loss, value_loss = iql.update(batch=batch)
                            q_train_losses.append(Q_bc_loss); v_train_losses.append(value_loss)
                    q_loss_mean, v_loss_mean = np.mean(q_train_losses), np.mean(v_train_losses)
                    print('Step: {}, Q loss: {}, Value loss: {}'.format(local_epoch_idx, q_loss_mean, v_loss_mean))
                    wandb_run.log({'Q_loss': q_loss_mean, 'value_loss': v_loss_mean})
                iql.save(Q_bc_path, value_path, encoder_path)
            q_eval = iql.minQ 
        # load dynamics parameters
        prediction_mode = getattr(self.cfg.dynamics, 'prediction_mode', 'last')
        if self.cfg.chunk_as_single_action and prediction_mode != "full":
            raise ValueError(
                "chunk_as_single_action=True requires dynamics.prediction_mode='full'. "
                "A chunk dynamics step advances the whole observation window, so "
                "'last' mode would mix stale observations with the predicted chunk endpoint."
            )
        dynamics_encoder = self.model.get_dynamics_encoder()
        if self.cfg.dynamics_type=="diffusion":
            dynamics_path = os.path.join(stage1_artifact_dir, f'saved_models_diffusion_{prediction_mode}')
        else:
            dynamics_path = os.path.join(stage1_artifact_dir, f'saved_models_{prediction_mode}')
        # set dynamics parameters
        if prediction_mode == "full" and self.cfg.n_obs_steps > 1:
            # In "full" mode, obs features are flattened [B, n_obs_steps * feature_dim],
            # so encoder_output_dim must reflect the full window size.
            self.cfg.lddm.encoder_output_dim = self.model.obs_feature_dim * self.cfg.n_obs_steps
        else:
            self.cfg.lddm.encoder_output_dim = self.model.obs_feature_dim
        if getattr(self.cfg, 'use_conv_action_embed', False):
            from rl_100.model.action_ae import ActionChunkEncoder
            conv_encoder = ActionChunkEncoder(
                action_dim=self.model.action_dim,
                hidden_dims=list(getattr(self.cfg, 'conv_hidden_dims', [128, 256])),
                latent_cz=getattr(self.cfg, 'conv_latent_cz', 32),
                kernel_size=getattr(self.cfg, 'conv_kernel_size', 5),
                n_groups=getattr(self.cfg, 'conv_n_groups', 8),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, self.cfg.n_action_steps, self.model.action_dim)
                self.cfg.lddm.action_embed_dim = conv_encoder(dummy).reshape(1, -1).shape[-1]
        elif prediction_mode == "full" and self.cfg.n_obs_steps > 1:
            # Keep action_embed_dim at single-step feature_dim to avoid scaling the action encoder.
            self.cfg.lddm.action_embed_dim = self.model.obs_feature_dim
        dynamics =  train_dynamics(
            env_runner.env, 
            self.model.normalizer, 
            dynamics_encoder, 
            dynamics_path, 
            self.cfg, 
            self.model.obs_feature_dim, 
            self.model.action_dim,
            chunk_as_single_action=self.cfg.chunk_as_single_action,
            n_action_steps=self.cfg.n_action_steps,
            n_obs_steps=self.cfg.n_obs_steps,
            device=device,
            )
        if (not os.path.exists(os.path.join(dynamics_path, "dynamics.pth")) and cfg.offline) or self.cfg.off2off:
            epoch = 0
            step_log = dict()
            dynamics_losses = list()
            for local_epoch_idx in range(cfg.dynamics.dynamics_max_epochs):
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {epoch}", 
                        leave=False, mininterval=cfg.dynamics.tqdm_interval_sec) as tepoch:
                    epoch += 1
                    for batch_idx, batch in enumerate(tepoch):
                        t1 = time.time()
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if self.cfg.chunk_as_single_action:
                            nobs_features, next_nobs_features = dynamics.obs2latent(batch['obs']), dynamics.next_obs2latent(batch['next_obs'])
                            single_nob_features, single_next_nob_features = nobs_features[:, -1, :], next_nobs_features[:, -1, :]
                        else:
                            nobs_features, next_nobs_features = dynamics.obs2latent(batch['obs']), dynamics.obs2latent(batch['next_obs'])
                            single_nob_features, single_next_nob_features = nobs_features[:, -1, :], next_nobs_features[:, -1, :]
                    
                        if prediction_mode == "full":
                            batch_size = nobs_features.shape[0]
                            train_nobs = nobs_features.reshape(batch_size, -1)  # [B, n_obs_steps * feature_dim]
                            train_next_nobs = next_nobs_features.reshape(batch_size, -1)
                        else:
                            train_nobs = single_nob_features
                            train_next_nobs = single_next_nob_features
                        
                        dynamics_loss = dynamics.learn(batch=batch, nobs_features=train_nobs, next_nobs_features=train_next_nobs)
                        dynamics.optimize(dynamics_loss)
                        dynamics_losses.append(dynamics_loss.item())  
                if (local_epoch_idx + 1) % 10 == 0:
                    print('dynamics loss: {}'.format(np.array(dynamics_losses[-10:]).mean()))      
                
                # Batched validation to avoid OOM
                with torch.no_grad():
                    val_losses_all = []
                    for val_batch in val_dataloader:
                        val_batch = dict_apply(val_batch, lambda x: x.to(device, non_blocking=True))
                        if self.cfg.chunk_as_single_action:
                            val_nobs_features, val_next_nobs_features = dynamics.obs2latent(val_batch['obs']), dynamics.next_obs2latent(val_batch['next_obs'])
                            val_single_nob_features, val_single_next_nob_features = val_nobs_features[:, -1, :], val_next_nobs_features[:, -1, :]
                        else:
                            val_nobs_features, val_next_nobs_features = dynamics.obs2latent(val_batch['obs']), dynamics.obs2latent(val_batch['next_obs'])
                            val_single_nob_features, val_single_next_nob_features = val_nobs_features[:, -1, :], val_next_nobs_features[:, -1, :]
                        
                        if prediction_mode == "full":
                            batch_size = val_nobs_features.shape[0]
                            val_nobs = val_nobs_features.reshape(batch_size, -1)
                            val_next_nobs = val_next_nobs_features.reshape(batch_size, -1)
                        else:
                            val_nobs = val_single_nob_features
                            val_next_nobs = val_single_next_nob_features
                        
                        # Compute validation loss for this batch
                        val_inputs, val_targets = dynamics.format_samples_for_training(val_batch, val_nobs, val_next_nobs)
                        batch_val_losses = dynamics.validate(val_inputs, val_targets)
                        val_losses_all.append(batch_val_losses)
                    
                    # Aggregate validation losses across batches
                    val_losses_all = np.array(val_losses_all)  # [num_batches, num_ensemble]
                    new_holdout_losses = val_losses_all.mean(axis=0).tolist()  # [num_ensemble]
                    
                # Update holdout losses and early stopping logic
                dynamics._update_holdout_and_log(new_holdout_losses, np.mean(dynamics_losses), wandb_run, epoch)
                if (dynamics.cnt >= cfg.dynamics.max_epochs_since_update) or (cfg.dynamics.dynamics_max_epochs and (epoch >= cfg.dynamics.dynamics_max_epochs)):
                    break
            dynamics.post_well_learned()
            dynamics.save(dynamics_path)  
        elif cfg.offline:
            dynamics.load(dynamics_path)
        #===============================================Stage 2 finetune dp3 by unio4 offline===============================================
        self.unio4.set_policy(self.model); self.unio4.set_old_policy()
        if cfg.eval:
            if self.cfg.unio4.idql_eval:
                log_data = self.unio4_eval(
                    idql_eval = True,
                    dynamics = dynamics,
                    first_action = self.cfg.unio4.first_action,
                    get_np = True,
                    iql = iql,
                    Q = Q_bc,
                    repeat_num = 128,
                    eval_times=self.cfg.unio4.eval_times
                )
            else:
                log_data = self.eval(eval_times=self.cfg.unio4.eval_times)
            
            score = log_data['test_mean_score']
            return score
        else:
            if cfg.offline:
                self.finetune_dp3(dynamics, Q_bc, value, iql, wandb, ema)
                
        #===============================================Stage 2-2 distill to cm from finetuned diffusion=================================================
        if self.cfg.distill_phase == 'after_offline':
            # import pdb; pdb.set_trace()
            if self.cfg.offline_cp_timestamp and self.cfg.offline_cp_timestep is not None:
                self.unio4.load(os.path.join(self.output_dir, self.cfg.offline_cp_timestamp, self.cfg.offline_cp_timestep))
            else:
                self.unio4.load(os.path.join(self.offline_best_path))
            self.distill2cm(train_dataloader, val_dataloader, wandb_run, env_runner, phase=self.cfg.distill_phase)
        #===============================================Stage 3 finetune dp3 by unio4 online=================================================
        if cfg.online:

            if self.cfg.load_bc:
                if self.cfg.distill_phase == 'online' and not self.cfg.ppo.load_online_cp:
                    raise RuntimeError(
                        "distill_phase='online' requires an offline policy checkpoint. "
                        "Disable load_bc or resume from an online checkpoint."
                    )
                self.unio4._policy.model.load_state_dict(self.model.model.state_dict())
                self.unio4._policy.obs_encoder.load_state_dict(self.model.obs_encoder.state_dict())
                self.unio4.set_old_policy()
            elif self.cfg.distill_phase in ('after_dp', 'after_offline'):
                # distill2cm() already ran on the correct model and promoted the student.
                # For after_dp: load promoted student from offline_best_path/last/
                # For after_offline: self.unio4._policy was distilled+promoted, skip reload.
                if self.cfg.distill_phase == 'after_dp':
                    self.unio4.load(os.path.join(self.offline_best_path, 'last'))
                # after_offline: unio4._policy already has promoted student, no reload needed

                # Fix PB-8: ppo.load() / set_old_policy() don't persist flow_inference_steps.
                # after_dp: neither _policy nor _old_policy has correct steps after ppo.load().
                # after_offline: _policy is correct after promote, but _old_policy is stale.
                if getattr(self.unio4._policy, 'is_flow', False):
                    target_steps = self.unio4._policy.flow_distill_inference_steps
                    self.unio4._policy.flow_inference_steps = target_steps
                    self.unio4._old_policy.flow_inference_steps = target_steps
                    cprint(f'restored flow_inference_steps={target_steps} for promoted model (policy + old_policy)', 'yellow')
            else:
                if self.cfg.offline_cp_timestamp and self.cfg.offline_cp_timestep is not None:
                    self.unio4.load(os.path.join(self.output_dir, self.cfg.offline_cp_timestamp, self.cfg.offline_cp_timestep))
                else:
                    self.unio4.load(os.path.join(self.offline_best_path))
            if self.cfg.distill_phase == 'online' and not self.cfg.ppo.load_online_cp:
                offline_distilled_path = os.path.join(self.offline_best_path, 'last', 'distilled_model.pt')
                if os.path.exists(offline_distilled_path):
                    cprint(
                        'found offline distilled model for online distill: {}'.format(
                            offline_distilled_path),
                        'green'
                    )
                else:
                    cprint(
                        'offline distilled model not found at {}; running offline distill before online'.format(
                            offline_distilled_path),
                        'yellow'
                    )
                    self.distill2cm(
                        train_dataloader,
                        val_dataloader,
                        wandb_run,
                        env_runner,
                        phase='after_offline'
                    )
                    if not os.path.exists(offline_distilled_path):
                        raise RuntimeError(
                            "Offline distill completed but did not create "
                            f"{offline_distilled_path}"
                        )
            if cfg.ppo.iql_ft:
                _iqlft_construct_snapshot = (
                    _iqlft_snapshot_rng() if _IQLFT_RESTORE_RNG_AT_CONSTRUCT else None
                )
                online_iql_encoder = deepcopy(self.model.obs_encoder)
                iql_online = IQL_Q_V_no(
                device=self.device,
                state_dim=self.model.obs_feature_dim * self.model.n_obs_steps,
                feature_dim=self.model.obs_feature_dim,
                action_dim=self.model.action_dim,
                q_hidden_dim=self.cfg.critic.q_hidden_dim,
                q_depth=self.cfg.critic.q_depth,
                Q_lr=self.cfg.ppo.ft_q_lr,
                target_update_freq=self.cfg.critic.target_update_freq,
                tau=self.cfg.critic.tau,
                gamma=self.cfg.critic.gamma,
                v_hidden_dim=self.cfg.critic.v_hidden_dim,
                v_depth=self.cfg.critic.v_depth,
                v_lr=self.cfg.ppo.ft_v_lr,
                omega=self.cfg.ppo.iql_omega,
                is_double_q=self.cfg.critic.is_double_q,
                dp3_normalizer=self.model.normalizer,
                obs_encoder=online_iql_encoder,
                n_obs_steps=self.model.n_obs_steps,
                is_share_encoder=self.cfg.ppo.is_share_iql_encoder,
                use_pc_color=self.model.use_pc_color,
                use_action_embed=self.cfg.use_action_embed,
                fix_encoder=self.cfg.ppo.fix_iql_encoder,
                encoder_update_with=self.cfg.ppo.iql_encoder_update_with,
                n_action_steps=self.cfg.n_action_steps,
                chunk_as_single_action=self.cfg.chunk_as_single_action,
                use_conv_action_embed=getattr(self.cfg, 'use_conv_action_embed', False),
                conv_hidden_dims=getattr(self.cfg, 'conv_hidden_dims', [128, 256]),
                conv_latent_cz=getattr(self.cfg, 'conv_latent_cz', 32),
                conv_kernel_size=getattr(self.cfg, 'conv_kernel_size', 5),
                conv_n_groups=getattr(self.cfg, 'conv_n_groups', 8),
                action_recon_beta=getattr(self.cfg, 'action_recon_beta', 0.5),
                q_layer_norm=getattr(self.cfg.critic, 'q_layer_norm', False),
                action_embed_layer_norm=getattr(self.cfg.critic, 'action_embed_layer_norm', False),
                action_scale_norm=getattr(self.cfg.critic, 'action_scale_norm', False),
                )
                iql_online.eval_with_raw_obs = True
                if os.path.exists(Q_bc_path):
                    if self.cfg.critic.load_pretrain:
                        online_encoder_path = encoder_path if encoder_path and os.path.exists(encoder_path) else None
                        if not self.cfg.ppo.is_share_iql_encoder:
                            iql_online.load_with_encoder(Q_bc_path, value_path, online_encoder_path)
                        else:
                            iql_online.load(
                                Q_bc_path,
                                value_path,
                                online_encoder_path,
                                force_load=online_encoder_path is not None,
                            )
                        cprint('load Q_bc and value for online iql finetuning successfully', 'green')
                if _IQLFT_RESTORE_RNG_AT_CONSTRUCT:
                    _iqlft_restore_rng(_iqlft_construct_snapshot)
            else:
                iql_online = None
            self.online_ft(dynamics, Q_bc, value, iql, iql_online, copy_encoder, wandb, ema)

    def finetune_dp3(self, dynamics, Q, value, iql, wandb, ema):
        # evaluation for dp3 pretrained by bc

        policy_refs = [self.model, self.unio4._policy, self.unio4._old_policy]
        aug_restore = []
        disabled_aug = False
        offline_use_aug = getattr(self.cfg, 'offline_use_aug', False)
        if not offline_use_aug:
            for policy_ref in policy_refs:
                if hasattr(policy_ref, 'module'):
                    policy_ref = policy_ref.module
                if hasattr(policy_ref, 'use_aug'):
                    aug_restore.append((policy_ref, policy_ref.use_aug))
                    if policy_ref.use_aug:
                        policy_ref.use_aug = False
                        disabled_aug = True 
            if disabled_aug:
                print('Disabled image augmentation for offline RL finetuning stage')



        # Create a new dataloader with configurable batch size for finetuning
        if self.cfg.offline and self.cfg.chunk_as_single_action and hasattr(self.cfg.task, 'finetune_dataset'):
            finetune_dataset = hydra.utils.instantiate(self.cfg.task.finetune_dataset)
            cprint(f'Finetune dataset: {len(finetune_dataset)} samples '
                   f'(stride={getattr(finetune_dataset, "sequence_stride", 1)})', 'cyan')
        else:
            finetune_dataset = self.dataset

        finetune_batch_size = getattr(self.cfg.unio4, 'finetune_batch_size', self.cfg.dataloader.batch_size)
        finetune_dataloader_cfg = OmegaConf.to_container(self.cfg.dataloader)
        finetune_dataloader_cfg['batch_size'] = finetune_batch_size
        # Mirror train_ddp.py: pinned-memory copies on nested batches are brittle on single-GPU sweeps.
        finetune_dataloader_cfg['pin_memory'] = False
        finetune_dataloader_cfg['persistent_workers'] = False
        finetune_dataloader_cfg['num_workers'] = min(finetune_dataloader_cfg.get('num_workers', 8), 2)
        self.finetune_dataloader = DataLoader(finetune_dataset, **finetune_dataloader_cfg)
        self.finetune_dataloader_iter = iter(self.finetune_dataloader)
        print(f'Finetuning with batch size: {finetune_batch_size}')
        
        self.unio4.set_old_policy()
        # Fix encoder in eval mode for the entire offline finetuning stage
        if self.cfg.unio4.fix_encoder:
            self.unio4._policy.obs_encoder.eval()
            self.unio4._old_policy.obs_encoder.eval()
        best_bppo_path = self.unio4_output_dir
        os.makedirs(best_bppo_path, exist_ok=True)
        best_saved_scores = float('-inf')
        run_idql_eval = bool(self.cfg.unio4.idql_eval)
        run_ema_eval = bool(self.cfg.training.use_ema and self.ema_model is not None)
        if run_idql_eval:
            idql_log_data = self.unio4_eval(
                idql_eval = True,
                dynamics = dynamics,
                first_action = self.cfg.unio4.first_action,
                get_np = True,
                iql = iql,
                Q = Q,
                repeat_num = 128,
                eval_times=self.cfg.unio4.eval_times,
                eval_name='IDQL Eval',
            )
        else:
            idql_log_data = None
        if run_ema_eval:
            ema_log_data = self.eval(
                eval_times=self.cfg.unio4.eval_times,
                policy_override=self.ema_model,
                eval_name='EMA Eval',
            )
            _, is_updated_ema = self.maybe_update_global_best_ema(ema_log_data['test_mean_score'])
            if is_updated_ema:
                print('------------saved best EMA model----------------')
        else:
            ema_log_data = None
        normal_log_data = self.eval(
            eval_times=self.cfg.unio4.eval_times,
            policy_override=self.unio4._policy,
            eval_name='Policy Eval',
        )
        
        # Save/select best checkpoint by the actual offline finetuned policy
        best_bppo_scores = normal_log_data['test_mean_score']
        # offline policy evaluation for dp3 pretrained by bc
        best_saved_scores, is_updated = self.maybe_update_global_best(best_bppo_scores)
        if is_updated:
            print('------------saved best model----------------')
        best_mean_qs = dynamics.rollout(
            self.unio4._policy, 
            Q, 
            iql, 
            self.sample_finetune_batch(), 
            rollout_length=self.cfg.unio4.rollout_length, 
            is_iql=self.cfg.critic.is_iql,
            use_gae=self.cfg.unio4.use_gae,
            first_action = self.cfg.dynamics.first_action,
        )
        print('rollout trajectory q mean:{}'.format(best_mean_qs))
        update_num = 0
        success_num = 0
        current_bppo_score = 0
        scores, opes = [], []
        idql_scores, ema_scores, normal_scores = [], [], []
        scores.append(best_bppo_scores)
        if run_idql_eval:
            idql_scores.append(idql_log_data['test_mean_score'])
        if run_ema_eval:
            ema_scores.append(ema_log_data['test_mean_score'])
        normal_scores.append(normal_log_data['test_mean_score'])
        opes.append(best_mean_qs[0].detach().cpu().numpy())
        init_log_data = {
            'current_bppo_scores': best_bppo_scores, 
            'current_mean_qs': best_mean_qs,
            'normal_eval_scores': normal_log_data['test_mean_score']
        }
        if run_ema_eval:
            init_log_data['ema_eval_scores'] = ema_log_data['test_mean_score']
        if run_idql_eval:
            init_log_data['idql_eval_scores'] = idql_log_data['test_mean_score']
        wandb.log(init_log_data)
        for step in tqdm.tqdm(range(int(self.cfg.unio4.bppo_steps)), desc='bppo updating ......'):
            if self.cfg.unio4.is_linear_decay:
                bppo_lr_now = self.cfg.unio4.bppo_lr * (1 - step / self.cfg.unio4.bppo_steps)
                q_lr_now = self.cfg.critic.q_lr * (1 - step / self.cfg.unio4.bppo_steps)
                clip_ratio_now = self.cfg.unio4.clip_ratio * (1 - step / self.cfg.unio4.bppo_steps)
            else:
                bppo_lr_now = None
                q_lr_now = None
                clip_ratio_now = None
            if step > 200:
                self.cfg.unio4.is_clip_decay = False
                self.cfg.unio4.is_bppo_lr_decay = False
            # finetune dp3 by unio4
            losses = self.unio4.update_distribution(
                batch=self.sample_finetune_batch(),
                value=value,
                Q=Q,
                iql=iql,
                is_clip_decay = self.cfg.unio4.is_clip_decay,
                is_lr_decay = self.cfg.unio4.is_bppo_lr_decay,
                is_linear_decay=self.cfg.unio4.is_linear_decay,
                bppo_lr_now= bppo_lr_now,
                clip_ratio_now= clip_ratio_now,
                dynamics=dynamics,
                use_gae=self.cfg.unio4.use_gae,
                fix_encoder=self.cfg.unio4.fix_encoder,
                final_reward=self.cfg.unio4.final_reward,
                gamma=self.cfg.critic.gamma,
                lamda=self.cfg.ppo.lamda,
                )
            if self.cfg.training.use_ema:
                ema.step(self.unio4._policy)
            wandb.log({'dpg_loss': losses})
            # evaluation during training
            if (step+1) % self.cfg.unio4.eval_freq == 0:
                if run_idql_eval:
                    idql_log_data = self.unio4_eval(
                        idql_eval = True,
                        dynamics = dynamics,
                        first_action = self.cfg.unio4.first_action,
                        get_np = True,
                        iql = iql,
                        Q = Q,
                        repeat_num = 128,
                        eval_times=self.cfg.unio4.eval_times,
                        eval_name='IDQL Eval',
                    )
                    idql_current_scores = idql_log_data['test_mean_score']
                    idql_scores.append(idql_current_scores)

                if run_ema_eval:
                    ema_log_data = self.eval(
                        eval_times=self.cfg.unio4.eval_times,
                        policy_override=self.ema_model,
                        eval_name='EMA Eval',
                    )
                    ema_current_scores = ema_log_data['test_mean_score']
                    ema_scores.append(ema_current_scores)
                    _, is_updated_ema = self.maybe_update_global_best_ema(ema_current_scores)
                    if is_updated_ema:
                        print('------------saved best EMA model----------------')
                else:
                    ema_current_scores = None
                
                # Run policy eval on the offline finetuned policy itself.
                normal_log_data = self.eval(
                    eval_times=self.cfg.unio4.eval_times,
                    policy_override=self.unio4._policy,
                    eval_name='Policy Eval',
                )
                normal_current_scores = normal_log_data['test_mean_score']
                normal_scores.append(normal_current_scores)
                
                # Save/select best checkpoint by policy eval, while still logging IDQL/EMA.
                current_bppo_scores = normal_current_scores
                
                best_saved_scores, is_updated = self.maybe_update_global_best(current_bppo_scores)
                if is_updated:
                    print('------------saved best model----------------')
                else:
                    os.makedirs(os.path.join(best_bppo_path, 'score_{}'.format(step)), exist_ok=True)
                    self.unio4.save(os.path.join(best_bppo_path, 'score_{}'.format(step)))
                    print('------------saved {} model----------------'.format(current_bppo_scores))
                scores.append(current_bppo_scores)
                score_parts = [f"Step: {step}"]
                if run_idql_eval:
                    score_parts.append(f"IDQL Score: {idql_current_scores}")
                if run_ema_eval:
                    score_parts.append(f"EMA Score: {ema_current_scores}")
                score_parts.append(f"Normal Score: {normal_current_scores}")
                score_parts.append(f"Selected Score: {current_bppo_scores}")
                print(", ".join(score_parts))
                eval_log_data = {
                    'current_bppo_scores': current_bppo_scores,
                    'normal_eval_scores': normal_current_scores
                }
                if run_ema_eval:
                    eval_log_data['ema_eval_scores'] = ema_current_scores
                if run_idql_eval:
                    eval_log_data['idql_eval_scores'] = idql_current_scores
                wandb.log(eval_log_data)
            # offline policy evaluation to determin whether to update behavior policy
            if (step+1)% self.cfg.unio4.eval_step == 0:
                current_mean_qs = dynamics.rollout(
                    self.unio4._policy, 
                    Q, 
                    iql, 
                    self.sample_finetune_batch(), 
                    rollout_length=self.cfg.unio4.rollout_length, 
                    is_iql=self.cfg.critic.is_iql,
                    use_gae=self.cfg.unio4.use_gae,
                    first_action = self.cfg.dynamics.first_action,
                )
                wandb.log({'current_mean_qs': current_mean_qs})
                print('rollout trajectory q mean:{}'.format(current_mean_qs))
                print(f"Step: {step}, Loss: ", losses)
                if self.cfg.unio4.is_update_old_policy:
                    if current_mean_qs > best_mean_qs:
                        best_mean_qs = current_mean_qs
                        self.unio4.set_old_policy()
                        print('------------------------------update behavior policy----------------------------------------')  
                opes.append(current_mean_qs[0].detach().cpu().numpy())
            np.savetxt(os.path.join(best_bppo_path, 'each_ope_score.csv'), opes, fmt='%f', delimiter=',') 
            np.savetxt(os.path.join(best_bppo_path, 'each_scores.csv'), scores, fmt='%f', delimiter=',')
            # Save separate CSV files for idql_eval and normal_eval
            if run_idql_eval and len(idql_scores) > 0:
                np.savetxt(os.path.join(best_bppo_path, 'each_idql_eval_scores.csv'), idql_scores, fmt='%f', delimiter=',')
            if run_ema_eval and len(ema_scores) > 0:
                np.savetxt(os.path.join(best_bppo_path, 'each_ema_eval_scores.csv'), ema_scores, fmt='%f', delimiter=',')
            if len(normal_scores) > 0:
                np.savetxt(os.path.join(best_bppo_path, 'each_normal_eval_scores.csv'), normal_scores, fmt='%f', delimiter=',')
        np.savetxt(os.path.join(best_bppo_path, 'last_ope_score.csv'), opes, fmt='%f', delimiter=',')
        # Save final separate CSV files for idql_eval and normal_eval
        if run_idql_eval and len(idql_scores) > 0:
            np.savetxt(os.path.join(best_bppo_path, 'last_idql_eval_scores.csv'), idql_scores, fmt='%f', delimiter=',')
        if run_ema_eval and len(ema_scores) > 0:
            np.savetxt(os.path.join(best_bppo_path, 'last_ema_eval_scores.csv'), ema_scores, fmt='%f', delimiter=',')
        if len(normal_scores) > 0:
            np.savetxt(os.path.join(best_bppo_path, 'last_normal_eval_scores.csv'), normal_scores, fmt='%f', delimiter=',')
        os.makedirs(os.path.join(self.output_dir, 'last'), exist_ok=True)
        self.unio4.save(os.path.join(self.output_dir, 'last'))
        for policy_ref, use_aug in aug_restore:
            policy_ref.use_aug = use_aug
        if disabled_aug:
            print('Restored image augmentation setting after offline RL finetuning stage')
        self.unio4.flush_ratio_logs(force=True)
        # wandb.finish()

    def _prepare_offline_iql_batch_for_online(self, offline_batch):
        """Match offline samples to the online IQL buffer contract."""
        start = self.cfg.n_obs_steps - 1

        if getattr(self.cfg, 'chunk_as_single_action', False):
            end = start + self.cfg.n_action_steps
            action_len = offline_batch['action'].shape[1]
            if action_len < end:
                raise ValueError(
                    f"offline IQL batch action horizon {action_len} is shorter "
                    f"than required chunk slice [{start}:{end}]")

            offline_batch['obs'] = dict_apply(
                offline_batch['obs'],
                lambda x: x[:, :self.cfg.n_obs_steps])
            offline_batch['next_obs'] = dict_apply(
                offline_batch['next_obs'],
                lambda x: x[:, -self.cfg.n_obs_steps:])

            offline_batch['action'] = offline_batch['action'][:, start:end]
            if self.cfg.action_norm:
                offline_batch['action'] = self.model.normalizer['action'].normalize(
                    offline_batch['action'])

            reward_chunk = offline_batch['reward'][:, start:end]
            if reward_chunk.shape[-1] == 1:
                reward_chunk = reward_chunk.squeeze(-1)
            gamma = float(getattr(self.cfg, 'gamma', self.cfg.critic.gamma))
            gamma_weights = torch.pow(
                torch.tensor(gamma, device=reward_chunk.device, dtype=reward_chunk.dtype),
                torch.arange(
                    self.cfg.n_action_steps,
                    device=reward_chunk.device,
                    dtype=reward_chunk.dtype,
                ),
            )
            offline_batch['reward'] = (
                reward_chunk * gamma_weights.reshape(1, -1)
            ).sum(dim=1).reshape(-1, 1, 1)
            offline_batch['not_done'] = offline_batch['not_done'][:, end - 1:end]
            return offline_batch

        offline_batch['action'] = offline_batch['action'][:, start:]
        offline_batch['reward'] = offline_batch['reward'][:, start:]
        offline_batch['not_done'] = offline_batch['not_done'][:, start:]
        if self.cfg.action_norm:
            offline_batch['action'] = self.model.normalizer['action'].normalize(
                offline_batch['action'])
        return offline_batch

    def _next_offline_iql_batch_for_online(self):
        """Reuse the offline dataloader iterator during online IQL updates."""
        offline_iter = getattr(self, '_online_iql_offline_iter', None)
        if offline_iter is None:
            offline_iter = iter(self.train_dataloader)
            self._online_iql_offline_iter = offline_iter

        try:
            offline_batch = next(offline_iter)
        except StopIteration:
            offline_iter = iter(self.train_dataloader)
            self._online_iql_offline_iter = offline_iter
            offline_batch = next(offline_iter)

        offline_batch = dict_apply(
            offline_batch,
            lambda x: x.to(self.device, non_blocking=True))
        return self._prepare_offline_iql_batch_for_online(offline_batch)

    def online_ft(self, dynamics, Q, value, iql, iql_online, copy_encoder, wandb, ema):
        from rl_100.unidpg.online_buffer import ReplayBuffer
        from rl_100.unidpg.online_buffer import IqlBuffer
        use_vec_env = getattr(self.cfg.ppo, 'use_vec_env_online', False)

        # VIB: optionally force stochastic sampling in online stage while encoder is in eval mode.
        enable_force_stochastic = getattr(self.cfg.ppo, 'force_stochastic_online', True)

        def _set_force_stochastic(encoder, val):
            if hasattr(encoder, 'force_stochastic'):
                encoder.force_stochastic = val

        def _set_iql_deterministic(iql_ref):
            if iql_ref is None:
                return
            encoders = [getattr(iql_ref, 'obs_encoder', None)]
            for net in [iql_ref._Q, iql_ref._target_Q, iql_ref._value]:
                encoders.append(getattr(net, '_obs_encoder', None))
            for encoder in encoders:
                if encoder is not None:
                    encoder.eval()
                    _set_force_stochastic(encoder, False)

        _set_force_stochastic(self.model.obs_encoder, enable_force_stochastic)
        _set_force_stochastic(self.unio4._policy.obs_encoder, enable_force_stochastic)
        _set_iql_deterministic(iql)
        _set_iql_deterministic(iql_online)
        if self.cfg.distill_phase == 'online':
            self.unio4._policy.set_target()
            distilled_path = os.path.join(self.offline_best_path, 'last/distilled_model.pt')
            if self.cfg.ppo.load_online_cp:
                cprint(
                    'skip offline distilled model load because ppo.load_online_cp=True; '
                    'online checkpoint will restore distilled model',
                    'yellow'
                )
            elif os.path.exists(distilled_path):
                self.unio4._policy.distilled_model.load_state_dict(torch.load(distilled_path))
                print('load distilled model from {} for online distill successfully'.format(distilled_path))
            else:
                raise RuntimeError(
                    "distill_phase='online' requires offline distill first, but "
                    f"{distilled_path} does not exist."
                )
            cm_optimizer, cm_lr_scheduler = self.get_distill_optimizer()
        else:
            cm_optimizer, cm_lr_scheduler = None, None
        online_ft_path = os.path.join(self.output_dir, 'online_ft', time.strftime("%Y-%m-%d-%H-%M-%S"))
        config = vars(self.cfg)  

        def write_dict(f, d, indent=0):
            for key, value in d.items():
                if isinstance(value, dict):
                    f.write(f"{' ' * indent}{key}:\n")
                    write_dict(f, value, indent + 4)  
                else:
                    f.write(f"{' ' * indent}{key:20} : {value}\n")

        os.makedirs(online_ft_path, exist_ok=True)
        config_path = os.path.join(online_ft_path, 'config.txt')

        with open(config_path, 'w') as f:
            write_dict(f, config)

        wandb.define_metric('online_reward_step')
        wandb.define_metric('online_reward/*', step_metric='online_reward_step')

        reward_scaler = None
        if self.cfg.ppo.scale_strategy == 'dynamic' or self.cfg.ppo.scale_strategy == 'number':
            critic_dataset = hydra.utils.instantiate(self.cfg.task.critic_dataset)
            assert isinstance(critic_dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(critic_dataset)}")
            critic_dataloader = DataLoader(critic_dataset, **self.cfg.dataloader)  
            # if self.cfg.ppo.share_encoder:
            online_value_encoder = self.unio4._policy.obs_encoder
            # else:
            #     online_value_encoder = copy_encoder   
            value = ValueLearner(
                self.device, 
                self.model.global_cond_dim, 
                self.cfg.critic.v_hidden_dim, 
                self.cfg.critic.v_depth, 
                self.cfg.critic.v_lr, 
                self.model.normalizer, 
                online_value_encoder, 
                self.model.n_obs_steps, 
                self.model.use_pc_color,
                share_encoder=self.cfg.ppo.share_encoder,
                )
            if self.cfg.ppo.share_encoder:
                v_path = os.path.join(self.output_dir, 'value_{}_{}.pt'.format(self.cfg.ppo.scale_strategy, self.cfg.ppo.share_encoder))
            else:
                v_path = os.path.join(self.output_dir, 'value_{}.pt'.format(self.cfg.ppo.scale_strategy))
            from rl_100.unidpg.utils import RewardScaling
            # reward_scaler = RewardScaling(shape=1, gamma=0.99)
            scale_dataset = hydra.utils.instantiate(self.cfg.task.scale_dataset)
            assert isinstance(critic_dataset, BaseDataset), print(f"dataset must be BaseDataset, got {type(critic_dataset)}")
            scale_dataloader = DataLoader(scale_dataset, **self.cfg.dataloader)
            reward_scaler = scale_dataset.reward_norm
            cprint('start training value network with dynamic reward scaling', 'green')
            if os.path.exists(v_path):
                value.load(v_path)
            elif self.cfg.ppo.scale_strategy == 'number':
                epoch = 0
                for local_epoch_idx in range(self.cfg.ppo.num_critic_epochs):

                    v_train_losses = list()
                    epoch += 1
                    with tqdm.tqdm(critic_dataloader, desc=f"Training epoch {epoch}", 
                                leave=False, mininterval=self.cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch['reward'], batch['return'] = batch['reward'] * 0.1, batch['return'] * 0.1
                            batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                            value_loss = value.update(batch)
                            v_train_losses.append(value_loss)
                    if local_epoch_idx % int(10) == 0:
                        print('Step: {}, Value loss: {}'.format(local_epoch_idx, np.mean(v_train_losses)))
                value.save(v_path)
            elif self.cfg.ppo.scale_strategy == 'dynamic':

                epoch = 0
                for local_epoch_idx in range(self.cfg.ppo.num_critic_epochs):
                    v_train_losses = list()
                    epoch += 1
                    with tqdm.tqdm(scale_dataloader, desc=f"Training epoch {epoch}", 
                                leave=False, mininterval=self.cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            batch['reward'], batch['return'] = batch['reward'], batch['return']
                            batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                            value_loss = value.update(batch)
                            v_train_losses.append(value_loss)
                    if local_epoch_idx % int(10) == 0:
                        print('Step: {}, Value loss: {}'.format(local_epoch_idx, np.mean(v_train_losses)))

                value.save(v_path)

            value_net = value._value
        else:
            value_net = iql.get_online_value_buget(self.cfg)
       
        # configure env
        # env_runner: BaseRunner
        # env_runner = hydra.utils.instantiate(
        #     self.cfg.task.env_runner,
        #     output_dir=self.output_dir)
        # # TODO: add seed in env's init
        # assert isinstance(env_runner, BaseRunner)
        # Sync inference step config after all distill-phase load/promote logic,
        # before buffer creation. Promoted student uses fewer steps (e.g. 1),
        # so buffer shapes must match the active policy's output.
        if getattr(self.unio4._policy, 'is_flow', False):
            active_steps = self.unio4._policy.flow_inference_steps
            if active_steps != self.cfg.ppo.num_inference_steps:
                cprint(f'syncing num_inference_steps: {self.cfg.ppo.num_inference_steps} -> {active_steps}', 'yellow')
                self.cfg.ppo.num_inference_steps = active_steps
                self.cfg.num_inference_steps = active_steps
                self.cfg.policy.num_inference_steps = active_steps

        replay_buffer = ReplayBuffer(args=self.cfg.ppo, shape_info=self.shape_info, device=self.device)

        # Define replay buffer persistence paths (per-run, under this hydra output dir)
        replay_buffer_path = os.path.join(self.output_dir, 'replay_buffer_checkpoint.pkl')
        is_save_image = bool(getattr(self.cfg.ppo, 'is_save_image', False))

        # On-policy safety: only restore the pkl if the matching online policy checkpoint
        # exists (it was saved right after a PPO update and before new collection). Otherwise
        # the a_logprob in the buffer was produced by a policy we can no longer recover, so
        # loading it would break on-policy. Drop the stale pkl in that case.
        online_cp_root = os.path.join(self.output_dir, 'online_ft')
        latest_online_cp_dir = None
        if os.path.isdir(online_cp_root):
            candidates = []
            for _sub in sorted(os.listdir(online_cp_root)):
                _full = os.path.join(online_cp_root, _sub)
                if os.path.isdir(_full) and os.path.exists(os.path.join(_full, 'update_num.txt')):
                    candidates.append(_full)
            if candidates:
                latest_online_cp_dir = candidates[-1]
        has_online_policy_cp = latest_online_cp_dir is not None

        if os.path.exists(replay_buffer_path) and not has_online_policy_cp:
            print(f"[On-policy guard] Dropping orphan replay buffer at {replay_buffer_path} "
                  f"(no matching online policy checkpoint under {online_cp_root})")
            try:
                os.remove(replay_buffer_path)
            except OSError as _e:
                print(f"[On-policy guard] failed to remove orphan buffer: {_e}")

        # Load replay buffer checkpoint if exists
        if os.path.exists(replay_buffer_path):
            try:
                with open(replay_buffer_path, 'rb') as f:
                    checkpoint = pickle.load(f)

                # Required fields
                required_fields = ['point_cloud', 'agent_pos', 'action', 'a_logprob',
                                   'next_point_cloud', 'next_agent_pos', 'reward', 'done', 'dw', 'count']

                # Restore required fields
                for field in required_fields:
                    if field in checkpoint:
                        setattr(replay_buffer, field, checkpoint[field])
                    else:
                        print(f"Warning: required field '{field}' missing from replay checkpoint")
                        replay_buffer.count = 0
                        break

                # Restore optional image fields
                if is_save_image:
                    if 'image' in checkpoint and hasattr(replay_buffer, 'image'):
                        replay_buffer.image = checkpoint['image']
                    else:
                        print("Warning: image field missing from checkpoint but is_save_image=True")
                    if 'next_image' in checkpoint and hasattr(replay_buffer, 'next_image'):
                        replay_buffer.next_image = checkpoint['next_image']

                # Restore optional imagin_robot fields
                if hasattr(replay_buffer, 'use_imagin_robot') and replay_buffer.use_imagin_robot:
                    if 'imagin_robot' in checkpoint and hasattr(replay_buffer, 'imagin_robot'):
                        replay_buffer.imagin_robot = checkpoint['imagin_robot']
                    if 'next_imagin_robot' in checkpoint and hasattr(replay_buffer, 'next_imagin_robot'):
                        replay_buffer.next_imagin_robot = checkpoint['next_imagin_robot']

                print(f"Loaded replay buffer checkpoint with {replay_buffer.count} transitions")
            except Exception as e:
                print(f"Failed to load replay buffer checkpoint: {e}")
                replay_buffer.count = 0

        def save_replay_buffer_checkpoint():
            if replay_buffer.count <= 0:
                return
            try:
                checkpoint = {
                    'point_cloud': replay_buffer.point_cloud,
                    'agent_pos': replay_buffer.agent_pos,
                    'action': replay_buffer.action,
                    'a_logprob': replay_buffer.a_logprob,
                    'next_point_cloud': replay_buffer.next_point_cloud,
                    'next_agent_pos': replay_buffer.next_agent_pos,
                    'reward': replay_buffer.reward,
                    'done': replay_buffer.done,
                    'dw': replay_buffer.dw,
                    'count': replay_buffer.count,
                }

                if is_save_image and hasattr(replay_buffer, 'image'):
                    checkpoint['image'] = replay_buffer.image
                    checkpoint['next_image'] = replay_buffer.next_image

                if hasattr(replay_buffer, 'use_imagin_robot') and replay_buffer.use_imagin_robot:
                    if hasattr(replay_buffer, 'imagin_robot'):
                        checkpoint['imagin_robot'] = replay_buffer.imagin_robot
                    if hasattr(replay_buffer, 'next_imagin_robot'):
                        checkpoint['next_imagin_robot'] = replay_buffer.next_imagin_robot

                with open(replay_buffer_path, 'wb') as f:
                    pickle.dump(checkpoint, f)
                print(f"[Real Robot] Saved replay buffer checkpoint with {replay_buffer.count} transitions")
            except Exception as e:
                print(f"[Real Robot] Failed to save replay buffer checkpoint: {e}")

        def verify_replay_buffer_checkpoint():
            if not (os.path.exists(replay_buffer_path) and replay_buffer.count > 0):
                return
            try:
                with open(replay_buffer_path, 'rb') as f:
                    checkpoint = pickle.load(f)

                required_fields = ['point_cloud', 'agent_pos', 'action', 'a_logprob',
                                   'next_point_cloud', 'next_agent_pos', 'reward', 'done', 'dw']
                count = checkpoint.get('count', 0)
                for field in required_fields:
                    if field in checkpoint:
                        checkpoint_val = checkpoint[field][:count]
                        buffer_val = getattr(replay_buffer, field)[:count]
                        if not np.array_equal(checkpoint_val, buffer_val):
                            raise RuntimeError(f"Replay buffer mismatch on field '{field}' at episode boundary")

                if is_save_image and 'image' in checkpoint and hasattr(replay_buffer, 'image'):
                    if not np.array_equal(checkpoint['image'][:count], replay_buffer.image[:count]):
                        raise RuntimeError("Replay buffer mismatch on field 'image' at episode boundary")

                if hasattr(replay_buffer, 'use_imagin_robot') and replay_buffer.use_imagin_robot:
                    if 'imagin_robot' in checkpoint and hasattr(replay_buffer, 'imagin_robot'):
                        if not np.array_equal(checkpoint['imagin_robot'][:count], replay_buffer.imagin_robot[:count]):
                            raise RuntimeError("Replay buffer mismatch on field 'imagin_robot' at episode boundary")
            except Exception as e:
                print(f"Replay buffer consistency check failed: {e}")
                raise

        if self.cfg.ppo.iql_ft or self.cfg.update_phase == 'outloop': 
            iql_buffer = IqlBuffer(None, args=self.cfg.ppo, shape_info=self.shape_info, device=self.device)
            # iql_buffer.initial_with_dataset(self.all_data)
            iql = iql_online
        # Auto-resume online checkpoint whenever it exists on disk, so the loaded
        # replay buffer (if any) stays on-policy with the current policy.
        auto_load_online_cp = self.cfg.ppo.load_online_cp or has_online_policy_cp
        if auto_load_online_cp:
            if latest_online_cp_dir is None:
                cprint(f"[Resume] load_online_cp requested but no ckpt found under "
                       f"{online_cp_root}; proceeding fresh from offline policy.", 'yellow')
                auto_load_online_cp = False
            else:
                cprint(f"[Resume] loading online ckpt from {latest_online_cp_dir}", 'green')
                iql, value_net = self.load_online_checkpoints(latest_online_cp_dir, iql, value_net, ema)
        self.unio4.transfer2online(critic=value_net, dynamics=dynamics, cfg=self.cfg, cm_optimizer=cm_optimizer, cm_lr_scheduler=cm_lr_scheduler)

        # Sync EMA to current online policy starting point (only for fresh offline→online,
        # NOT when resuming from online checkpoint which already restored EMA)
        if self.cfg.training.use_ema and self.ema_model is not None and ema is not None:
            if not auto_load_online_cp:
                ema_state = self.ema_model.state_dict()
                policy_state = self.unio4._policy.state_dict()
                filtered_state = {k: v for k, v in policy_state.items() if k in ema_state}
                self.ema_model.load_state_dict(filtered_state, strict=False)
                ema.optimization_step = 0

        if use_vec_env:
            raise RuntimeError(
                "train_real.py supports only single-env real online training; "
                "set ppo.use_vec_env_online=False."
            )

        # start training and data collection
        total_steps = 0
        env_runner = self.env_runner
        env = env_runner.env
        # if hasattr(env, 'seed'):
        #     env.seed(int(self.cfg.training.seed))
        all_success_rates, all_returns = [], []
        cm_all_success_rates, cm_all_returns = [], []
        all_idql_success_rates, all_idql_returns = [], []
        all_ema_success_rates, all_ema_returns = [], []
        # if self.cfg.ppo.idql_eval:
        #     idql_log_data = self.unio4_eval(
        #             idql_eval = True,
        #             dynamics = dynamics,
        #             first_action = self.cfg.unio4.first_action,
        #             get_np = True,
        #             use_gae=self.cfg.unio4.use_gae,
        #             iql = iql,
        #             Q = Q,
        #             repeat_num = 128,
        #             eval_times=self.cfg.unio4.eval_times,
        #             traj_path=online_ft_path,
        #             data_collect=self.cfg.data_collect
        #             )
        #     all_idql_success_rates.append(idql_log_data['test_mean_score'])
        #     all_idql_returns.append(idql_log_data['mean_returns'])
        #     log_data = self.eval(eval_times=self.cfg.unio4.eval_times, online=True, traj_path=online_ft_path, data_collect=self.cfg.data_collect)
        #     if self.cfg.distill_phase == 'online':
        #         cm_log_data = self.eval(
        #             online=True, eval_times=self.cfg.unio4.eval_times,
        #             use_cm=True, distill2mean=self.cfg.distill2mean, traj_path=online_ft_path, data_collect=self.cfg.data_collect)
        #         cm_all_success_rates.append(cm_log_data['test_mean_score'])
        #         cm_all_returns.append(cm_log_data['mean_returns'])
        #     else:
        #         cm_all_success_rates.append(0)
        #         cm_all_returns.append(0)
        # else:
        #     log_data = self.eval(eval_times=self.cfg.unio4.eval_times, online=True, traj_path=online_ft_path, data_collect=self.cfg.data_collect)
        #     if self.cfg.distill_phase == 'online':
        #         cm_log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times, use_cm=True, distill2mean=self.cfg.distill2mean, traj_path=online_ft_path, data_collect=self.cfg.data_collect)
        #         cm_all_success_rates.append(cm_log_data['test_mean_score'])
        #         cm_all_returns.append(cm_log_data['mean_returns'])
        #     else:
        #         cm_all_success_rates.append(0)
        #         cm_all_returns.append(0)
        #     all_idql_success_rates.append(0)
        #     all_idql_returns.append(0)
        # all_success_rates.append(log_data['test_mean_score'])
        # all_returns.append(log_data['mean_returns'])
        # # Initial EMA eval
        # ema_log_data = None
        # if self.cfg.training.use_ema and self.ema_model is not None:
        #     ema_log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times, traj_path=online_ft_path, data_collect=self.cfg.data_collect,
        #                              policy_override=self.ema_model, eval_name='Online EMA Eval')
        #     all_ema_success_rates.append(ema_log_data['test_mean_score'])
        #     all_ema_returns.append(ema_log_data['mean_returns'])
        #     _, is_updated_ema = self.maybe_update_online_best_ema(ema_log_data['test_mean_score'])
        #     if is_updated_ema:
        #         print('------------saved online best EMA model----------------')
        # else:
        #     all_ema_success_rates.append(0)
        #     all_ema_returns.append(0)
        # cprint('start online finetuning, initial policy SR: {}, EMA SR: {}'.format(
        #     log_data['test_mean_score'],
        #     ema_log_data['test_mean_score'] if ema_log_data else 'N/A'), 'green')
        # wandb.log({'online ppo success rates': log_data['test_mean_score'], 'cm success rates': cm_all_success_rates, 'cm returns': cm_all_returns,
        #                 'online ppo returns': log_data['mean_returns'],
        #                 'online ema success rates': ema_log_data['test_mean_score'] if ema_log_data else 0,
        #                 'online ema returns': ema_log_data['mean_returns'] if ema_log_data else 0,})
        # progress_bar = tqdm.tqdm(total=self.cfg.ppo.max_train_steps, desc="Training Progress")
        evaluate_num = 0
        actor_losses, critic_losses, bc_losses, distill_losses = [], [], [], []
        q_train_losses, v_train_losses = [], []
        total_mean_return = []
        total_reward_sub = 0
        total_episode_r =  deque(maxlen=10)
        chunk_reward_history = []
        episode_return_history = []
        episode_success_history = []
        reward_curve_path = os.path.join(online_ft_path, 'online_reward_curves.png')
        episode_reward = 0
        pre_reward = 0  # Track previous reward for None handling in step_online
        time1 = 0
        episode_steps = 0
        update_num = 0
        idql_log_data = None
        while total_steps < self.cfg.ppo.max_train_steps:
            # start rollout
            obs = env.reset()

            # Ask operator to confirm reward label and physical reset (skip before first episode)
            if total_steps > 0 and not getattr(env_runner, 'handles_keyboard_result', False):
                print("\nEpisode ended. Was it successful?")
                user_input = input("Type number for success, letter for failure: ").strip()
                # Use the last character to avoid interference from stop key
                if user_input:
                    user_input = user_input[-1]
                if user_input and user_input[0].isdigit():
                    print("Episode: Success recorded!")
                else:
                    print("Episode: Failure recorded!")
                time.sleep(1)

            # policy.reset()
            done = False
            total_count_sub = 0
            if self.cfg.ppo.scale_strategy == 'dynamic':
                reward_scaler.reset()
            print('episode reward: {}, episode length: {}'.format(episode_reward, episode_steps))

            episode_steps = 0
            episode_reward = 0
            # obs['image'] = np.transpose(obs['image'], (0,2,3,1))
            if self.cfg.ppo.clip_std_decay:
                decay_value = self.value_decay(initial_value=self.cfg.clip_std_max, total_steps=total_steps, max_train_steps=self.cfg.ppo.max_train_steps)
                self.unio4._policy.noise_scheduler.clip_std_max = decay_value
            while not done:
                episode_steps += 1
                np_obs_dict = dict(obs)
                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                                      lambda x: torch.from_numpy(x).to(
                                          device=self.device))
                # run policy
                obs_dict_input = {}  # flush unused keys
                obs_dict_input['point_cloud'] = obs_dict['point_cloud'].unsqueeze(0)
                obs_dict_input['agent_pos'] = obs_dict['agent_pos'].unsqueeze(0)
                if 'dexart' in self.cfg.task_name:
                    obs_dict_input['imagin_robot'] = obs_dict['imagin_robot'].unsqueeze(0)
                obs_dict_input['image'] = (obs_dict['image'].unsqueeze(0)).to(torch.float)
                if self.cfg.ppo.idql_rollout:
                    action, all_x, a_logprob = self.unio4._policy.sample_action_with_logprob(obs_dict_input, dynamics=dynamics, first_action=self.cfg.unio4.first_action, use_gae=self.cfg.unio4.use_gae, iql=iql, Q=Q, repeat_num=128)
                else:
                    action, all_x, a_logprob = self.unio4._policy.all_step_action_logprob(obs_dict_input, fix_encoder=self.cfg.ppo.fix_encoder)

                # device_transfer
                all_x = all_x.squeeze(1).detach().to('cpu').numpy()
                a_logprob = a_logprob.squeeze(1).detach().to('cpu').numpy()

                # step env - use step_online for real robot training
                next_obs, reward, done, info = env.step_online(action.squeeze(0).detach().to('cpu').numpy(), gamma=self.cfg.gamma)

                # Handle None reward (can occur during keyboard input)
                if reward is None:
                    reward = pre_reward
                reward = float(reward)
                episode_reward += reward
                pre_reward = reward

                # next_obs['image'] = np.transpose(next_obs['image'], (0,2,3,1))
                is_timeout = bool(
                    done
                    and isinstance(info, dict)
                    and np.asarray(info.get('timeout', False)).any()
                )
                dw = bool(done and not is_timeout)
                # store transition
                obs_dict = dict_apply(obs_dict,
                                      lambda x: x.detach().to('cpu').numpy())
                # next_obs_dict = dict_apply(dict(next_obs), lambda x: x.squeeze())
                if self.cfg.ppo.scale_strategy == 'number':
                    replay_buffer.store(obs_dict, all_x, a_logprob, reward * 0.1, next_obs, done, dw)
                elif self.cfg.ppo.scale_strategy == 'dynamic':
                    scaled_r = reward_scaler(reward)[0]
                    replay_buffer.store(obs_dict, all_x, a_logprob, scaled_r, next_obs, done, dw)
                else:
                    replay_buffer.store(obs_dict, all_x, a_logprob, reward, next_obs, done, dw)

                if self.cfg.ppo.iql_ft or self.cfg.update_phase == 'outloop':
                    iql_buffer.store(obs=obs_dict, action=all_x[-1], reward=reward, next_obs=next_obs, done=done)

                if self.cfg.update_phase == 'outloop':
                    alpha = 0.8 + (1 - 0.8) * (total_steps / self.cfg.ppo.max_train_steps) # linearly increase the alpha from 0.5 to 1
                    idql_bs = int(getattr(self.cfg.ppo, 'idql_batch_size', 256))
                    online_sample_size = int(alpha * idql_bs)
                    offline_sample_size = idql_bs - online_sample_size
                    online_batch = iql_buffer.sample(batch_size=online_sample_size)
                    offline_batch = self.sample_batch(batch_size=offline_sample_size)
                    offline_batch = self._prepare_offline_iql_batch_for_online(offline_batch)
                    merged_batch = iql_buffer.merge(online_batch, offline_batch)
                    distill_loss = self.unio4.distill_update(merged_batch, online=True)
                    distill_losses.append(distill_loss)
                obs = next_obs
                total_steps += 1
                chunk_reward_history.append(reward)
                chunk_success = bool(
                    isinstance(info, dict)
                    and np.asarray(info.get('is_success', False)).any()
                )
                reward_log = {
                    'online_reward_step': total_steps,
                    'online_reward/chunk': reward,
                    'online_reward/episode_return_running': episode_reward,
                }
                if done:
                    total_episode_r.append(episode_reward)
                    episode_return_history.append(episode_reward)
                    episode_success_history.append(float(chunk_success))
                    reward_log.update({
                        'online_reward/episode_return': episode_reward,
                        'online_reward/episode_return_mean_10': float(np.mean(total_episode_r)),
                        'online_reward/success': float(chunk_success),
                    })
                    _save_online_reward_curves(
                        reward_curve_path,
                        chunk_reward_history,
                        episode_return_history,
                        episode_success_history,
                    )
                wandb.log(reward_log)
                # progress_bar.update(1)
                total_count_sub += 1 
                if replay_buffer.count == self.cfg.ppo.batch_size:
                    update_num += 1
                    if self.cfg.ppo.iql_ft:   
                        # iql_buffer.store(obs=obs_dict, action=all_x[-1], reward=reward, next_obs=next_obs, done=done)
                        if total_steps > self.cfg.ppo.online_start_training:
                            print('start online iql training')
                            for _ in range(self.cfg.ppo.iql_steps):
                                alpha = self.cfg.ppo.data_ratio + (1 - self.cfg.ppo.data_ratio) * (total_steps / self.cfg.ppo.max_train_steps) # linearly increase the alpha from 0.5 to 1
                                idql_bs = int(getattr(self.cfg.ppo, 'idql_batch_size', 256))
                                online_sample_size = int(alpha * idql_bs)
                                offline_sample_size = idql_bs - online_sample_size
                                online_batch = iql_buffer.sample(batch_size=online_sample_size)
                                offline_batch = self._next_offline_iql_batch_for_online()
                                merged_batch = iql_buffer.merge(online_batch, offline_batch)
                                merged_batch = dict_apply(merged_batch, lambda x: x[:idql_bs]) # batch size idql_bs, and online batch is larger
                                Q_bc_loss, value_loss = iql.update(batch=merged_batch, online=True, pre_cut=True, online_recon=self.cfg.ppo.online_iql_recon)
                            if total_steps % self.cfg.ppo.evaluate_freq  == 0:
                                print('Step: {}, Q loss: {}, Value loss: {}'.format(total_steps, Q_bc_loss, value_loss))
                                wandb.log({'online iql Q_loss': Q_bc_loss, 'online iql value value_loss': value_loss})
                            q_train_losses.append(Q_bc_loss); v_train_losses.append(value_loss)
                        if self.cfg.ppo.fix_encoder:
                            if self.cfg.ppo.iql_q_encoder:
                                # print('======================using iql q encoder======================')
                                self.unio4._policy.obs_encoder.load_state_dict(iql._Q._obs_encoder.state_dict())
                            elif self.cfg.ppo.iql_v_encoder:
                                self.unio4._policy.obs_encoder.load_state_dict(iql._value._obs_encoder.state_dict())
                    time2 = time.time()
                    pre_training_time = time.time()
                    pre_training_time = time.time()
                    actor_loss, critic_loss, bc_loss, distill_loss = self.unio4.dp_align_update_no_share(replay_buffer, total_steps)
                    if distill_loss != 0:
                        distill_losses.append(distill_loss)
                    post_training_time = time.time()
                    print('pure policy updated time: {}'.format(post_training_time - pre_training_time))
                    # print('Step: {}, actor_loss: {}, critic_loss: {}'.format(total_steps, actor_loss, critic_loss))
                    time3 = time.time()
                    if self.cfg.training.use_ema and ema is not None:
                        ema.step(self.unio4._policy)
                    ppo_elapsed = getattr(self.unio4, 'last_ppo_elapsed', None)
                    ppo_time_str = f'; ppo loop: {ppo_elapsed:.2f}s' if ppo_elapsed is not None else ''
                    print('step {}; collecting data time: {}; update time: {}{}'.format(total_steps, time2 - time1, time3 - time2, ppo_time_str))
                    replay_buffer.count = 0

                    # Remove replay buffer checkpoint after PPO update
                    if os.path.exists(replay_buffer_path):
                        os.remove(replay_buffer_path)

                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                    bc_losses.append(bc_loss)
                    time1 = time.time()
                    if self.cfg.ppo.save_online_cp and update_num % self.cfg.ppo.online_cp_save_freq == 0:
                        self.save_online_checkpoints(online_ft_path, update_num, iql, ema)

                if done:
                    if isinstance(info, dict) and 'is_success' in info:
                        final_success = bool(np.asarray(info['is_success']).any())
                        print(f"[Real Robot] Episode keyboard success: {final_success}")
                    save_replay_buffer_checkpoint()
                    verify_replay_buffer_checkpoint()

                if done and total_steps % self.cfg.ppo.evaluate_freq == 0:
                    evaluate_num += 1
                    if self.cfg.ppo.idql_eval:
                        idql_log_data = self.unio4_eval(
                                idql_eval = True,
                                dynamics = dynamics,
                                first_action = self.cfg.unio4.first_action,
                                get_np = True,
                                use_gae=self.cfg.unio4.use_gae,
                                iql = iql,
                                Q = Q,
                                repeat_num = 128,
                                eval_times=self.cfg.unio4.eval_times,
                                traj_path=online_ft_path,
                                data_collect=self.cfg.data_collect
                                )
                        log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times, traj_path=online_ft_path, data_collect=self.cfg.data_collect)
                        all_idql_success_rates.append(idql_log_data['test_mean_score'])
                        all_idql_returns.append(idql_log_data['mean_returns'])
                        if self.cfg.distill_phase == 'online':
                            cm_log_data = self.eval(
                                online=True, eval_times=self.cfg.unio4.eval_times,
                                use_cm=True, distill2mean=self.cfg.distill2mean, traj_path=online_ft_path, data_collect=self.cfg.data_collect)
                            cm_all_success_rates.append(cm_log_data['test_mean_score'])
                            cm_all_returns.append(cm_log_data['mean_returns'])
                        else:
                            cm_all_success_rates.append(0)
                            cm_all_returns.append(0)
                    else:
                        log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times, traj_path=online_ft_path, data_collect=self.cfg.data_collect)
                        if self.cfg.distill_phase == 'online':
                            cm_log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times, use_cm=True, distill2mean=self.cfg.distill2mean, traj_path=online_ft_path, data_collect=self.cfg.data_collect)
                            cm_all_success_rates.append(cm_log_data['test_mean_score'])
                            cm_all_returns.append(cm_log_data['mean_returns'])
                        else:
                            cm_all_success_rates.append(0)
                            cm_all_returns.append(0)
                        all_idql_success_rates.append(0)
                        all_idql_returns.append(0)

                    all_success_rates.append(log_data['test_mean_score'])
                    all_returns.append(log_data['mean_returns'])

                    # Online EMA eval
                    ema_log_data = None
                    if self.cfg.training.use_ema and self.ema_model is not None:
                        ema_log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times,
                                                 policy_override=self.ema_model, eval_name='Online EMA Eval',
                                                 traj_path=online_ft_path, data_collect=self.cfg.data_collect)
                        all_ema_success_rates.append(ema_log_data['test_mean_score'])
                        all_ema_returns.append(ema_log_data['mean_returns'])
                        _, is_updated_ema = self.maybe_update_online_best_ema(ema_log_data['test_mean_score'])
                        if is_updated_ema:
                            print('------------saved online best EMA model----------------')
                    else:
                        all_ema_success_rates.append(0)
                        all_ema_returns.append(0)

                    cprint(
                        'timestep {}: collecting performance: {} evaluate success rates: {}; evaluate returns: {}  actor_loss: {}; critic_loss: {}; bc_loss: {}; distill_loss: {}; cm_SR: {}; cm_ret: {}; idql_SR: {}; idql_ret: {}; ema_SR: {}; ema_ret: {};'.format(
                            total_steps,
                            np.mean(total_episode_r),
                            log_data['test_mean_score'],
                            log_data['mean_returns'],
                            np.mean(actor_losses[int(-self.cfg.ppo.evaluate_freq):]),
                            np.mean(critic_losses[int(-self.cfg.ppo.evaluate_freq):]),
                            np.mean(bc_losses[int(-self.cfg.ppo.evaluate_freq):]),
                            np.mean(distill_losses[int(-self.cfg.ppo.evaluate_freq):]),
                            cm_log_data['test_mean_score'] if self.cfg.distill_phase == 'online' else 0,
                            cm_log_data['mean_returns'] if self.cfg.distill_phase == 'online' else 0,
                            idql_log_data['test_mean_score'] if idql_log_data else 0,
                            idql_log_data['mean_returns'] if idql_log_data else 0,
                            ema_log_data['test_mean_score'] if ema_log_data else 0,
                            ema_log_data['mean_returns'] if ema_log_data else 0,
                        ),
                        'green'
                    )
                    wandb.log({
                        'online ppo success rates': log_data['test_mean_score'], 
                        'online ppo returns': log_data['mean_returns'],
                        'online ppo collect returns': np.mean(total_episode_r),
                        'online actor_loss': np.mean(actor_losses[int(-self.cfg.ppo.evaluate_freq):]), 
                        'online critic_loss': np.mean(critic_losses[int(-self.cfg.ppo.evaluate_freq):]), 
                        'online bc_loss': np.mean(bc_losses[int(-self.cfg.ppo.evaluate_freq):]),
                        'online distill_loss': np.mean(distill_losses[int(-self.cfg.ppo.evaluate_freq):]),
                        'cm_success rates': cm_log_data['test_mean_score'] if self.cfg.distill_phase == 'online' else 0,
                        'cm_returns': cm_log_data['mean_returns'] if self.cfg.distill_phase == 'online' else 0,
                        'idql_success rates': idql_log_data['test_mean_score'] if idql_log_data else 0,
                        'idql_returns': idql_log_data['mean_returns'] if idql_log_data else 0,
                        'online ema success rates': ema_log_data['test_mean_score'] if ema_log_data else 0,
                        'online ema returns': ema_log_data['mean_returns'] if ema_log_data else 0,
                        })
                    # if self.cfg.ppo.iql_ft:
                    #     wandb.log({'Q_loss': np.mean(q_train_losses[int(-self.cfg.ppo.evaluate_freq):]), 'value_loss': np.mean(v_train_losses[int(-self.cfg.ppo.evaluate_freq):])})
                    #     cprint('timestep {} q_loss: {}; v_loss: {}'.format(total_steps, np.mean(q_train_losses[int(-self.cfg.ppo.evaluate_freq):]), np.mean(v_train_losses[int(-self.cfg.ppo.evaluate_freq):])), 'green')
                    
                    os.makedirs(online_ft_path, exist_ok=True)
                    np.savetxt(os.path.join(online_ft_path, 'success_rates.csv'), all_success_rates, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(online_ft_path, 'returns.csv'), all_returns, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(online_ft_path, 'idql_success_rates.csv'), all_idql_success_rates, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(online_ft_path, 'idql_returns.csv'), all_idql_returns, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(online_ft_path, 'cm_success_rates.csv'), cm_all_success_rates, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(online_ft_path, 'cm_returns.csv'), cm_all_returns, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(online_ft_path, 'ema_success_rates.csv'), all_ema_success_rates, fmt='%f', delimiter=',')
                    np.savetxt(os.path.join(online_ft_path, 'ema_returns.csv'), all_ema_returns, fmt='%f', delimiter=',')
        os.makedirs(os.path.join(online_ft_path, 'online_last'), exist_ok=True)

        self.unio4.save(os.path.join(online_ft_path, 'online_last'))
        if self.cfg.training.use_ema and self.ema_model is not None:
            os.makedirs(os.path.join(online_ft_path, 'online_last_ema'), exist_ok=True)
            self.ema_model.save(os.path.join(online_ft_path, 'online_last_ema'))
        self.unio4.flush_ratio_logs(force=True)

    def _online_ft_vec(self, dynamics, Q, iql, iql_online, wandb, online_ft_path, cm_optimizer, cm_lr_scheduler, ema=None, reward_scaler_template=None):
        """Vec env online finetuning branch (ppo.use_vec_env_online=True).
        Uses manual env list (not SubprocVecEnv) to support MultiStepWrapper kwargs."""
        from rl_100.unidpg.online_buffer_vec import ReplayBuffer as VecReplayBuffer
        from rl_100.unidpg.online_buffer import ReplayBuffer as FlatReplayBuffer
        from rl_100.unidpg.uni_ppo import compute_gae_per_env
        import copy as copy_module

        # VIB: optionally force stochastic sampling in online stage while encoder is in eval mode.
        enable_force_stochastic = getattr(self.cfg.ppo, 'force_stochastic_online', True)

        def _set_force_stochastic(encoder, val):
            if hasattr(encoder, 'force_stochastic'):
                encoder.force_stochastic = val

        def _set_iql_deterministic(iql_ref):
            if iql_ref is None:
                return
            encoders = [getattr(iql_ref, 'obs_encoder', None)]
            for net in [iql_ref._Q, iql_ref._target_Q, iql_ref._value]:
                encoders.append(getattr(net, '_obs_encoder', None))
            for encoder in encoders:
                if encoder is not None:
                    encoder.eval()
                    _set_force_stochastic(encoder, False)

        _set_force_stochastic(self.model.obs_encoder, enable_force_stochastic)
        _set_force_stochastic(self.unio4._policy.obs_encoder, enable_force_stochastic)
        _set_iql_deterministic(iql)
        _set_iql_deterministic(iql_online)

        # --- guard unsupported combinations ---
        assert getattr(self.cfg, 'update_phase', 'inloop') != 'outloop', \
            'vec_env v1 does not support update_phase=outloop'
        assert not getattr(self.cfg.ppo, 'iql_adv', False), \
            'vec_env v1 does not support ppo.iql_adv=True'
        assert not getattr(self.cfg.ppo, 'idql_rollout', False), \
            'vec_env v1 does not support ppo.idql_rollout=True'

        train_env_num = getattr(self.cfg.ppo, 'train_env_num', 1)
        env_runner = self.env_runner
        steps_per_update = self.cfg.ppo.batch_size // train_env_num
        assert self.cfg.ppo.batch_size % train_env_num == 0, \
            f'batch_size ({self.cfg.ppo.batch_size}) must be divisible by train_env_num ({train_env_num})'

        use_subproc_vec_rollout = (
            getattr(self.cfg, 'feature_type', None) == '2D'
            and hasattr(env_runner, 'make_subproc_vec_env')
        )
        vec_env = None
        if use_subproc_vec_rollout:
            vec_env = env_runner.make_subproc_vec_env(
                train_env_num,
                record_video_first=False,
                reward_agg_method='discounted_sum',
                gamma=self.cfg.gamma,
            )
            vec_env.seed(int(self.cfg.training.seed))
            envs = None
        else:
            envs = [env_runner.make_env(record_video=False) for _ in range(train_env_num)]
            for env_idx, env in enumerate(envs):
                env.seed(int(self.cfg.training.seed) + env_idx)
        max_steps = self.cfg.task.env_runner.max_steps

        # per-env reward scalers for dynamic scaling
        if self.cfg.ppo.scale_strategy == 'dynamic':
            if reward_scaler_template is None:
                raise RuntimeError(
                    'vec dynamic reward scaling requires a non-null reward_scaler_template')
            import copy as copy_module_std
            reward_scalers = [copy_module_std.deepcopy(reward_scaler_template) for _ in range(train_env_num)]
            for scaler in reward_scalers:
                scaler.reset()

        replay_buffer = VecReplayBuffer(
            args=self.cfg.ppo, shape_info=self.shape_info,
            device=self.device, env_num=train_env_num,
            steps_per_update=steps_per_update)
        replay_buffer.reset()

        iql_ft = getattr(self.cfg.ppo, 'iql_ft', False)
        if iql_ft:
            from rl_100.unidpg.online_buffer import IqlBuffer
            iql_buffer = IqlBuffer(None, args=self.cfg.ppo, shape_info=self.shape_info, device=self.device)

        obs_debug_printed = False

        def stack_obs_dicts(obs_list):
            """Stack vec rollout observations into a batched float tensor dict."""
            nonlocal obs_debug_printed

            if len(obs_list) == 0:
                raise RuntimeError('vec rollout received an empty obs_list')

            expected_keys = tuple(self.shape_info['obs'].keys())
            reference_keys = tuple(obs_list[0].keys())
            missing_from_first = [key for key in expected_keys if key not in reference_keys]
            if missing_from_first:
                raise KeyError(
                    f"vec rollout obs is missing required keys {missing_from_first}; "
                    f"available keys: {sorted(reference_keys)}")

            batched = {}
            for key in expected_keys:
                missing_envs = [idx for idx, obs in enumerate(obs_list) if key not in obs]
                if missing_envs:
                    raise KeyError(
                        f"vec rollout obs key '{key}' missing from env indices {missing_envs}")

                try:
                    stacked = np.stack([obs[key] for obs in obs_list], axis=0)
                except ValueError as exc:
                    shapes = [np.asarray(obs[key]).shape for obs in obs_list]
                    raise ValueError(
                        f"vec rollout obs key '{key}' has inconsistent shapes across envs: {shapes}"
                    ) from exc

                if stacked.size == 0:
                    raise ValueError(f"vec rollout obs key '{key}' produced an empty batch")

                batched[key] = torch.from_numpy(stacked).to(device=self.device, dtype=torch.float)

            if not obs_debug_printed:
                print(f'vec rollout obs keys: {list(batched.keys())}')
                if 'image' in batched:
                    image = batched['image']
                    print(
                        'vec rollout image batch: '
                        f'shape={tuple(image.shape)}, dtype={image.dtype}, '
                        f'min={image.min().item():.4f}, max={image.max().item():.4f}'
                    )
                if 'point_cloud' in batched:
                    point_cloud = batched['point_cloud']
                    print(
                        'vec rollout point_cloud batch: '
                        f'shape={tuple(point_cloud.shape)}, dtype={point_cloud.dtype}'
                    )
                if 'agent_pos' in batched:
                    agent_pos = batched['agent_pos']
                    print(
                        'vec rollout agent_pos batch: '
                        f'shape={tuple(agent_pos.shape)}, dtype={agent_pos.dtype}'
                    )
                obs_debug_printed = True

            return batched

        def unstack_obs_batch(obs_batch_np):
            keys = list(obs_batch_np.keys())
            batch_size = obs_batch_np[keys[0]].shape[0]
            return [
                {k: obs_batch_np[k][i] for k in keys}
                for i in range(batch_size)
            ]

        # --- initial eval ---
        all_success_rates, all_returns = [], []
        cm_all_success_rates, cm_all_returns = [], []
        all_idql_success_rates, all_idql_returns = [], []
        all_ema_success_rates, all_ema_returns = [], []
        if self.cfg.ppo.idql_eval:
            idql_log_data = self.unio4_eval(
                idql_eval=True, dynamics=dynamics,
                first_action=self.cfg.unio4.first_action, get_np=True,
                use_gae=self.cfg.unio4.use_gae, iql=iql, Q=Q,
                repeat_num=128, eval_times=self.cfg.unio4.eval_times)
            all_idql_success_rates.append(idql_log_data['test_mean_score'])
            all_idql_returns.append(idql_log_data['mean_returns'])
            log_data = self.eval(eval_times=self.cfg.unio4.eval_times, online=True)
            if self.cfg.distill_phase == 'online':
                cm_log_data = self.eval(
                    online=True, eval_times=self.cfg.unio4.eval_times,
                    use_cm=True, distill2mean=self.cfg.distill2mean)
                cm_all_success_rates.append(cm_log_data['test_mean_score'])
                cm_all_returns.append(cm_log_data['mean_returns'])
            else:
                cm_all_success_rates.append(0)
                cm_all_returns.append(0)
        else:
            log_data = self.eval(eval_times=self.cfg.unio4.eval_times, online=True)
            if self.cfg.distill_phase == 'online':
                cm_log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times,
                                        use_cm=True, distill2mean=self.cfg.distill2mean)
                cm_all_success_rates.append(cm_log_data['test_mean_score'])
                cm_all_returns.append(cm_log_data['mean_returns'])
            else:
                cm_all_success_rates.append(0)
                cm_all_returns.append(0)
            all_idql_success_rates.append(0)
            all_idql_returns.append(0)
        all_success_rates.append(log_data['test_mean_score'])
        all_returns.append(log_data['mean_returns'])
        # Initial EMA eval
        ema_log_data = None
        if self.cfg.training.use_ema and self.ema_model is not None:
            ema_log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times,
                                     policy_override=self.ema_model, eval_name='Online EMA Eval')
            all_ema_success_rates.append(ema_log_data['test_mean_score'])
            all_ema_returns.append(ema_log_data['mean_returns'])
            _, is_updated_ema = self.maybe_update_online_best_ema(ema_log_data['test_mean_score'])
            if is_updated_ema:
                print('------------saved online best EMA model----------------')
        else:
            all_ema_success_rates.append(0)
            all_ema_returns.append(0)
        cprint('start vec online finetuning, env_num={}, initial policy SR: {}, EMA SR: {}'.format(
            train_env_num, log_data['test_mean_score'],
            ema_log_data['test_mean_score'] if ema_log_data else 'N/A'), 'green')
        wandb.log({
            'online ppo success rates': log_data['test_mean_score'],
            'online ppo returns': log_data['mean_returns'],
            'cm_success rates': cm_all_success_rates[-1] if cm_all_success_rates else 0,
            'cm_returns': cm_all_returns[-1] if cm_all_returns else 0,
            'idql_success rates': all_idql_success_rates[-1] if all_idql_success_rates else 0,
            'idql_returns': all_idql_returns[-1] if all_idql_returns else 0,
            'online ema success rates': ema_log_data['test_mean_score'] if ema_log_data else 0,
            'online ema returns': ema_log_data['mean_returns'] if ema_log_data else 0,
        })

        # --- main loop state ---
        total_steps = 0
        evaluate_num = 0
        next_eval_at = self.cfg.ppo.evaluate_freq
        actor_losses, critic_losses, bc_losses, distill_losses = [], [], [], []
        q_train_losses, v_train_losses = [], []
        total_episode_r = deque(maxlen=10)
        episode_rewards = [0.0] * train_env_num
        episode_steps_per_env = [0] * train_env_num
        update_num = 0
        time1 = time.time()
        idql_log_data = None

        # init per-env obs
        if use_subproc_vec_rollout:
            obs_list = unstack_obs_batch(vec_env.reset())
        else:
            obs_list = [envs[i].reset() for i in range(train_env_num)]

        while total_steps < self.cfg.ppo.max_train_steps:
            if getattr(self.cfg.ppo, 'clip_std_decay', False):
                decay_value = self.value_decay(
                    initial_value=self.cfg.clip_std_max,
                    total_steps=total_steps,
                    max_train_steps=self.cfg.ppo.max_train_steps)
                self.unio4._policy.noise_scheduler.clip_std_max = decay_value

            # save obs before step (for buffer store)
            obs_before_step = [dict(obs) for obs in obs_list]

            # batched policy inference
            obs_dict_input = stack_obs_dicts(obs_list)
            with torch.no_grad():
                action, all_x, a_logprob = self.unio4._policy.all_step_action_logprob(
                    obs_dict_input, fix_encoder=self.cfg.ppo.fix_encoder)

            all_x_np = all_x.detach().cpu().numpy()
            a_logprob_np = a_logprob.detach().cpu().numpy()
            action_np = action.detach().cpu().numpy()

            # per-env step
            next_obs_list = [None] * train_env_num
            step_rewards = np.zeros(train_env_num)
            step_dones = np.zeros(train_env_num)
            step_dws = np.zeros(train_env_num)

            if use_subproc_vec_rollout:
                obs_after_step_np, reward_batch, done_batch, info_batch = vec_env.step(action_np)
                reset_obs_list = unstack_obs_batch(obs_after_step_np)
                for i in range(train_env_num):
                    reward = float(reward_batch[i])
                    done = bool(done_batch[i])
                    info = info_batch[i]

                    episode_rewards[i] += reward
                    episode_steps_per_env[i] += 1

                    dw = done and episode_steps_per_env[i] != max_steps

                    if done and 'terminal_observation' in info:
                        next_obs_list[i] = info['terminal_observation']
                    else:
                        next_obs_list[i] = reset_obs_list[i]

                    if self.cfg.ppo.scale_strategy == 'number':
                        step_rewards[i] = reward * 0.1
                    elif self.cfg.ppo.scale_strategy == 'dynamic':
                        step_rewards[i] = reward_scalers[i](reward)[0]
                    else:
                        step_rewards[i] = reward

                    step_dones[i] = float(done)
                    step_dws[i] = float(dw)

                    if iql_ft:
                        iql_buffer.store(obs=obs_before_step[i], action=all_x_np[-1, i],
                                         reward=reward, next_obs=next_obs_list[i],
                                         done=step_dones[i])

                    if done:
                        total_episode_r.append(episode_rewards[i])
                        print(f'env {i} episode reward: {episode_rewards[i]:.2f}, steps: {episode_steps_per_env[i]}')
                        episode_rewards[i] = 0.0
                        episode_steps_per_env[i] = 0
                        if self.cfg.ppo.scale_strategy == 'dynamic':
                            reward_scalers[i].reset()

                    obs_list[i] = reset_obs_list[i]
            else:
                for i in range(train_env_num):
                    next_obs, reward, done, info = envs[i].step(
                        action_np[i], reward_agg_method='discounted_sum', gamma=self.cfg.gamma)

                    episode_rewards[i] += reward
                    episode_steps_per_env[i] += 1

                    # dw: true termination (not max_steps truncation)
                    dw = done and episode_steps_per_env[i] != max_steps

                    if self.cfg.ppo.scale_strategy == 'number':
                        step_rewards[i] = reward * 0.1
                    elif self.cfg.ppo.scale_strategy == 'dynamic':
                        step_rewards[i] = reward_scalers[i](reward)[0]
                    else:
                        step_rewards[i] = reward

                    step_dones[i] = float(done)
                    step_dws[i] = float(dw)
                    next_obs_list[i] = next_obs  # terminal obs (before reset)

                    # per-env iql buffer store (before auto-reset)
                    # Use raw reward (not scaled) to match offline IQL data distribution
                    if iql_ft:
                        iql_buffer.store(obs=obs_before_step[i], action=all_x_np[-1, i],
                                         reward=reward, next_obs=next_obs_list[i],
                                         done=step_dones[i])

                    # auto-reset
                    if done:
                        total_episode_r.append(episode_rewards[i])
                        print(f'env {i} episode reward: {episode_rewards[i]:.2f}, steps: {episode_steps_per_env[i]}')
                        episode_rewards[i] = 0.0
                        episode_steps_per_env[i] = 0
                        if self.cfg.ppo.scale_strategy == 'dynamic':
                            reward_scalers[i].reset()
                        obs_list[i] = envs[i].reset()
                    else:
                        obs_list[i] = next_obs

            # build batched data for vec buffer
            obs_keys = list(obs_before_step[0].keys())
            obs_batch_np = {k: np.stack([obs_before_step[i][k] for i in range(train_env_num)], axis=0)
                            for k in obs_keys}
            next_obs_batch_np = {k: np.stack([next_obs_list[i][k] for i in range(train_env_num)], axis=0)
                                  for k in obs_keys}

            # all_x: (T+1, train_env_num, ...) -> (train_env_num, T+1, ...)
            all_x_for_buffer = np.moveaxis(all_x_np, 1, 0) if all_x_np.ndim > 2 and all_x_np.shape[1] == train_env_num else all_x_np
            a_logprob_for_buffer = np.moveaxis(a_logprob_np, 1, 0) if a_logprob_np.ndim > 2 and a_logprob_np.shape[1] == train_env_num else a_logprob_np

            replay_buffer.store(obs_batch_np, all_x_for_buffer, a_logprob_for_buffer,
                                step_rewards, next_obs_batch_np, step_dones, step_dws)

            total_steps += train_env_num

            # PPO update when buffer full
            if replay_buffer.count == steps_per_update:
                update_num += 1

                # --- online IQL training (before PPO update) ---
                if iql_ft:
                    if total_steps > self.cfg.ppo.online_start_training:
                        rng_snapshot = _iqlft_snapshot_rng() if _IQLFT_RESTORE_RNG else None
                        print('start online iql training')
                        for _ in range(self.cfg.ppo.iql_steps):
                            alpha = self.cfg.ppo.data_ratio + (1 - self.cfg.ppo.data_ratio) * (total_steps / self.cfg.ppo.max_train_steps)
                            idql_bs = int(getattr(self.cfg.ppo, 'idql_batch_size', 256))
                            online_sample_size = int(alpha * idql_bs)
                            offline_sample_size = idql_bs - online_sample_size
                            online_batch = iql_buffer.sample(batch_size=online_sample_size)
                            offline_batch = self._next_offline_iql_batch_for_online()
                            merged_batch = iql_buffer.merge(online_batch, offline_batch)
                            merged_batch = dict_apply(merged_batch, lambda x: x[:idql_bs])
                            Q_bc_loss, value_loss = iql.update(batch=merged_batch, online=True, pre_cut=True, online_recon=self.cfg.ppo.online_iql_recon)
                        if total_steps >= next_eval_at - self.cfg.ppo.evaluate_freq + train_env_num:
                            print('Step: {}, Q loss: {}, Value loss: {}'.format(total_steps, Q_bc_loss, value_loss))
                            wandb.log({'online iql Q_loss': Q_bc_loss, 'online iql value value_loss': value_loss})
                        q_train_losses.append(Q_bc_loss); v_train_losses.append(value_loss)
                    # encoder backfill
                    if self.cfg.ppo.fix_encoder:
                        if getattr(self.cfg.ppo, 'iql_q_encoder', False):
                            self.unio4._policy.obs_encoder.load_state_dict(iql._Q._obs_encoder.state_dict())
                        elif getattr(self.cfg.ppo, 'iql_v_encoder', False):
                            self.unio4._policy.obs_encoder.load_state_dict(iql._value._obs_encoder.state_dict())
                    if _IQLFT_RESTORE_RNG and total_steps > self.cfg.ppo.online_start_training:
                        _iqlft_restore_rng(rng_snapshot)

                # per-env GAE
                s_vec, a_vec, a_logprob_vec, r_vec, s_vec_, dw_vec, done_vec = \
                    replay_buffer.numpy_to_tensor_vec()

                with torch.no_grad():
                    flat_s = dict_apply(s_vec, lambda x: x.reshape(-1, *x.shape[2:]))
                    flat_s_ = dict_apply(s_vec_, lambda x: x.reshape(-1, *x.shape[2:]))
                    if self.unio4.args.share_encoder:
                        flat_vs, flat_vs_ = self.unio4._compute_critic_values_in_chunks(
                            flat_s, flat_s_, use_obs2latent=True)
                    else:
                        flat_vs, flat_vs_ = self.unio4._compute_critic_values_in_chunks(
                            flat_s, flat_s_, use_obs2latent=False)
                    vs = flat_vs.reshape(steps_per_update, train_env_num, 1)
                    vs_ = flat_vs_.reshape(steps_per_update, train_env_num, 1)

                    adv, v_target = compute_gae_per_env(
                        r_vec, done_vec, dw_vec, vs, vs_,
                        self.cfg.ppo.gamma, self.cfg.ppo.lamda, self.cfg.n_action_steps)

                # create flat buffer for dp_align_update_no_share
                flat_args = copy_module.copy(self.cfg.ppo)
                flat_args.batch_size = steps_per_update * train_env_num
                flat_replay = FlatReplayBuffer(args=flat_args, shape_info=self.shape_info,
                                               device=self.device)

                # flatten vec buffer into flat buffer
                if not replay_buffer.wo_visual:
                    flat_replay.point_cloud = replay_buffer.point_cloud[:steps_per_update].reshape(
                        -1, *replay_buffer.point_cloud.shape[2:])
                    flat_replay.image = replay_buffer.image[:steps_per_update].reshape(
                        -1, *replay_buffer.image.shape[2:])
                    if replay_buffer.use_imagin_robot:
                        flat_replay.imagin_robot = replay_buffer.imagin_robot[:steps_per_update].reshape(
                            -1, *replay_buffer.imagin_robot.shape[2:])
                flat_replay.agent_pos = replay_buffer.agent_pos[:steps_per_update].reshape(
                    -1, *replay_buffer.agent_pos.shape[2:])
                flat_replay.action = replay_buffer.action[:steps_per_update].reshape(
                    -1, *replay_buffer.action.shape[2:])
                flat_replay.a_logprob = replay_buffer.a_logprob[:steps_per_update].reshape(
                    -1, *replay_buffer.a_logprob.shape[2:])
                flat_replay.reward = replay_buffer.reward[:steps_per_update].reshape(-1, 1)
                if not replay_buffer.wo_visual:
                    flat_replay.next_point_cloud = replay_buffer.next_point_cloud[:steps_per_update].reshape(
                        -1, *replay_buffer.next_point_cloud.shape[2:])
                    flat_replay.next_image = replay_buffer.next_image[:steps_per_update].reshape(
                        -1, *replay_buffer.next_image.shape[2:])
                    if replay_buffer.use_imagin_robot:
                        flat_replay.next_imagin_robot = replay_buffer.next_imagin_robot[:steps_per_update].reshape(
                            -1, *replay_buffer.next_imagin_robot.shape[2:])
                flat_replay.next_agent_pos = replay_buffer.next_agent_pos[:steps_per_update].reshape(
                    -1, *replay_buffer.next_agent_pos.shape[2:])
                flat_replay.done = replay_buffer.done[:steps_per_update].reshape(-1, 1)
                flat_replay.dw = replay_buffer.dw[:steps_per_update].reshape(-1, 1)
                flat_replay.count = steps_per_update * train_env_num

                precomputed = {
                    'adv': adv,
                    'v_target': v_target,
                    'vs': flat_vs.reshape(-1, 1),
                }

                time2 = time.time()
                actor_loss, critic_loss, bc_loss, distill_loss = self.unio4.dp_align_update_no_share(
                    flat_replay, total_steps, precomputed=precomputed)
                if distill_loss != 0:
                    distill_losses.append(distill_loss)
                time3 = time.time()
                if self.cfg.training.use_ema and ema is not None:
                    ema.step(self.unio4._policy)
                print(f'step {total_steps}; collecting data time: {time2 - time1:.2f}; '
                      f'update time: {time3 - time2:.2f}')

                replay_buffer.reset()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                bc_losses.append(bc_loss)
                time1 = time.time()

                if getattr(self.cfg.ppo, 'save_online_cp', False) and \
                   update_num % getattr(self.cfg.ppo, 'online_cp_save_freq', 100) == 0:
                    self.save_online_checkpoints(online_ft_path, update_num, iql, ema)

            # eval (threshold-based, not modulo)
            if total_steps >= next_eval_at:
                next_eval_at += self.cfg.ppo.evaluate_freq
                evaluate_num += 1
                if self.cfg.ppo.idql_eval:
                    idql_log_data = self.unio4_eval(
                        idql_eval=True, dynamics=dynamics,
                        first_action=self.cfg.unio4.first_action, get_np=True,
                        use_gae=self.cfg.unio4.use_gae, iql=iql, Q=Q,
                        repeat_num=128, eval_times=self.cfg.unio4.eval_times)
                    log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times)
                    all_idql_success_rates.append(idql_log_data['test_mean_score'])
                    all_idql_returns.append(idql_log_data['mean_returns'])
                    if self.cfg.distill_phase == 'online':
                        cm_log_data = self.eval(
                            online=True, eval_times=self.cfg.unio4.eval_times,
                            use_cm=True, distill2mean=self.cfg.distill2mean)
                        cm_all_success_rates.append(cm_log_data['test_mean_score'])
                        cm_all_returns.append(cm_log_data['mean_returns'])
                    else:
                        cm_all_success_rates.append(0)
                        cm_all_returns.append(0)
                else:
                    log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times)
                    if self.cfg.distill_phase == 'online':
                        cm_log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times,
                                                use_cm=True, distill2mean=self.cfg.distill2mean)
                        cm_all_success_rates.append(cm_log_data['test_mean_score'])
                        cm_all_returns.append(cm_log_data['mean_returns'])
                    else:
                        cm_all_success_rates.append(0)
                        cm_all_returns.append(0)
                    all_idql_success_rates.append(0)
                    all_idql_returns.append(0)

                all_success_rates.append(log_data['test_mean_score'])
                all_returns.append(log_data['mean_returns'])

                # Online EMA eval
                ema_log_data = None
                if self.cfg.training.use_ema and self.ema_model is not None:
                    ema_log_data = self.eval(online=True, eval_times=self.cfg.unio4.eval_times,
                                             policy_override=self.ema_model, eval_name='Online EMA Eval')
                    all_ema_success_rates.append(ema_log_data['test_mean_score'])
                    all_ema_returns.append(ema_log_data['mean_returns'])
                    _, is_updated_ema = self.maybe_update_online_best_ema(ema_log_data['test_mean_score'])
                    if is_updated_ema:
                        print('------------saved online best EMA model----------------')
                else:
                    all_ema_success_rates.append(0)
                    all_ema_returns.append(0)

                cprint(
                    'timestep {}: collect perf: {} eval SR: {}; eval ret: {} actor_loss: {}; critic_loss: {}; cm_SR: {}; cm_ret: {}; idql_SR: {}; idql_ret: {}; ema_SR: {}; ema_ret: {};'.format(
                        total_steps, np.mean(total_episode_r) if total_episode_r else 0,
                        log_data['test_mean_score'], log_data['mean_returns'],
                        np.mean(actor_losses[-100:]) if actor_losses else 0,
                        np.mean(critic_losses[-100:]) if critic_losses else 0,
                        cm_log_data['test_mean_score'] if self.cfg.distill_phase == 'online' else 0,
                        cm_log_data['mean_returns'] if self.cfg.distill_phase == 'online' else 0,
                        idql_log_data['test_mean_score'] if idql_log_data else 0,
                        idql_log_data['mean_returns'] if idql_log_data else 0,
                        ema_log_data['test_mean_score'] if ema_log_data else 0,
                        ema_log_data['mean_returns'] if ema_log_data else 0,
                    ), 'green')

                wandb.log({
                    'online ppo success rates': log_data['test_mean_score'],
                    'online ppo returns': log_data['mean_returns'],
                    'online ppo collect returns': np.mean(total_episode_r) if total_episode_r else 0,
                    'online actor_loss': np.mean(actor_losses[-100:]) if actor_losses else 0,
                    'online critic_loss': np.mean(critic_losses[-100:]) if critic_losses else 0,
                    'cm_success rates': cm_log_data['test_mean_score'] if self.cfg.distill_phase == 'online' else 0,
                    'cm_returns': cm_log_data['mean_returns'] if self.cfg.distill_phase == 'online' else 0,
                    'idql_success rates': idql_log_data['test_mean_score'] if idql_log_data else 0,
                    'idql_returns': idql_log_data['mean_returns'] if idql_log_data else 0,
                    'online ema success rates': ema_log_data['test_mean_score'] if ema_log_data else 0,
                    'online ema returns': ema_log_data['mean_returns'] if ema_log_data else 0,
                })

                os.makedirs(online_ft_path, exist_ok=True)
                np.savetxt(os.path.join(online_ft_path, 'success_rates.csv'),
                           all_success_rates, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'returns.csv'),
                           all_returns, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'cm_success_rates.csv'),
                           cm_all_success_rates, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'cm_returns.csv'),
                           cm_all_returns, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'idql_success_rates.csv'),
                           all_idql_success_rates, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'idql_returns.csv'),
                           all_idql_returns, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'ema_success_rates.csv'),
                           all_ema_success_rates, fmt='%f', delimiter=',')
                np.savetxt(os.path.join(online_ft_path, 'ema_returns.csv'),
                           all_ema_returns, fmt='%f', delimiter=',')

        # cleanup
        if vec_env is not None:
            vec_env.close()
        else:
            for e in envs:
                e.close() if hasattr(e, 'close') else None
        os.makedirs(os.path.join(online_ft_path, 'online_last'), exist_ok=True)
        self.unio4.save(os.path.join(online_ft_path, 'online_last'))
        if self.cfg.training.use_ema and self.ema_model is not None:
            os.makedirs(os.path.join(online_ft_path, 'online_last_ema'), exist_ok=True)
            self.ema_model.save(os.path.join(online_ft_path, 'online_last_ema'))
        self.unio4.flush_ratio_logs(force=True)

    def unio4_eval(self, idql_eval: bool = False, dynamics = None, first_action = False, get_np = True, use_gae = True, iql = None, Q = None, repeat_num = 100, eval_times: int = 1, use_cm=False, distill2mean=False, eval_name: str = 'IDQL Eval', traj_path=None, data_collect=False):
        # load the latest checkpoint

        cfg = copy.deepcopy(self.cfg)
        env_runner = self.env_runner
        policy = self.unio4._policy
        if cfg.training.use_ema:
            if cfg.unio4.use_ema_eval:
                policy = self.ema_model

        # VIB: deterministic eval uses the latent mean; data collection can
        # force stochastic latent sampling for RL rollout diversity.
        force_stochastic_for_collect = bool(
            data_collect and getattr(self.cfg.ppo, 'force_stochastic_online', True)
        )
        saved_force_stochastic = []
        seen_encoders = set()

        def _set_force_stochastic_temporarily(encoder, value):
            if encoder is None or not hasattr(encoder, 'force_stochastic'):
                return
            encoder_id = id(encoder)
            if encoder_id in seen_encoders:
                return
            seen_encoders.add(encoder_id)
            saved_force_stochastic.append((encoder, encoder.force_stochastic))
            encoder.force_stochastic = value

        _set_force_stochastic_temporarily(
            getattr(policy, 'obs_encoder', None),
            force_stochastic_for_collect,
        )
        if idql_eval and iql is not None:
            _set_force_stochastic_temporarily(getattr(iql, 'obs_encoder', None), False)
            for net_name in ['_Q', '_target_Q', '_value']:
                net = getattr(iql, net_name, None)
                _set_force_stochastic_temporarily(getattr(net, '_obs_encoder', None), False)
        if idql_eval and Q is not None:
            _set_force_stochastic_temporarily(getattr(Q, '_obs_encoder', None), False)
            _set_force_stochastic_temporarily(getattr(Q, 'obs_encoder', None), False)

        policy.eval()
        eval_env_num = getattr(self.cfg.ppo, 'eval_env_num', 1)
        try:
            idql_run_params = inspect.signature(env_runner.idql_run).parameters
        except (TypeError, ValueError):
            idql_run_params = {}
        try:
            run_params = inspect.signature(env_runner.run).parameters
        except (TypeError, ValueError):
            run_params = {}
        log_data = {'test_mean_score': [], 'mean_returns': []}
        try:
            for i in tqdm.tqdm(range(eval_times), desc='evaluating ......'):
                if idql_eval:
                    idql_kwargs = {
                        'dynamics': dynamics,
                        'first_action': first_action,
                        'get_np': get_np,
                        'use_gae': use_gae,
                        'iql': iql,
                        'Q': Q,
                        'repeat_num': repeat_num,
                        'use_cm': use_cm,
                        'distill2mean': distill2mean,
                        'eval_env_num': eval_env_num,
                        'traj_path': traj_path,
                        'data_collect': data_collect,
                    }
                    idql_kwargs = {k: v for k, v in idql_kwargs.items() if k in idql_run_params}
                    runner_log = env_runner.idql_run(policy, **idql_kwargs)
                else:
                    run_kwargs = {
                        'use_cm': use_cm,
                        'distill2mean': distill2mean,
                        'eval_env_num': eval_env_num,
                        'traj_path': traj_path,
                        'data_collect': data_collect,
                    }
                    run_kwargs = {k: v for k, v in run_kwargs.items() if k in run_params}
                    runner_log = env_runner.run(policy, **run_kwargs)
                cprint(f"---------------- {eval_name} Results --------------", 'magenta')
                for key, value in runner_log.items():
                    if isinstance(value, float):
                        cprint(f"{key}: {value:.4f}", 'magenta')
                log_data['test_mean_score'].append(runner_log['test_mean_score'])
                log_data['mean_returns'].append(runner_log['mean_returns'])
        finally:
            for encoder, prev_value in reversed(saved_force_stochastic):
                encoder.force_stochastic = prev_value
        log_data['test_mean_score'] = np.mean(log_data['test_mean_score'])
        log_data['mean_returns'] = np.mean(log_data['mean_returns'])
        # import pdb; pdb.set_trace()
        print(f'{eval_name} average success rates:', log_data['test_mean_score'])
        print(f'{eval_name} average rewards:', np.mean(log_data['mean_returns']))
        return log_data

    def get_distill_optimizer(self,):
        cfg = self.cfg
        cm_optimizer = torch.optim.AdamW(
            self.unio4._policy.distilled_model.parameters(),
            lr=cfg.optimizer.lr,
            betas=(cfg.optimizer.betas[0], cfg.optimizer.betas[1]),
            weight_decay=cfg.optimizer.weight_decay,
            eps=cfg.optimizer.eps)

        cm_lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=cm_optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
               cfg.ppo.max_train_steps * cfg.ppo.K_epochs)# \
                    # // cfg.training.gradient_accumulate_every
        )
        return cm_optimizer, cm_lr_scheduler

    def distill2cm(self, train_dataloader, val_dataloader, wandb_run, env_runner, phase: str = 'after_dp'): # distill from after dp/offline/online
        cprint('start distill to cm {}'.format(phase), 'green')
        # =============================== stage 1-2: set for distillation to consistency model using ddim solver ===============================
        cfg = self.cfg
        if phase == 'after_dp':
            model_to_optimize = self.model
            latest_cm_path = self.get_checkpoint_path(tag='latest_cm')
        elif phase == 'after_offline':
            model_to_optimize = self.unio4._policy
            latest_cm_path = os.path.join(self.offline_best_path, 'last', 'distilled_model.pt')
        else:
            raise RuntimeError(f"Unsupported distill phase: {phase}")

        if self.cfg.training.resume and os.path.exists(latest_cm_path):
            cprint(f'resume=True and found existing distill checkpoint at {latest_cm_path}; skip distill2cm', 'yellow')
            return

        model_to_optimize.set_target()
        cm_optimizer = torch.optim.AdamW(
            model_to_optimize.distilled_model.parameters(),
            lr=cfg.optimizer.lr,
            betas=(cfg.optimizer.betas[0], cfg.optimizer.betas[1]),
            weight_decay=cfg.optimizer.weight_decay,
            eps=cfg.optimizer.eps)

        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=cm_optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every
        )
        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )
        self.global_step = 0
        self.epoch = 0
        train_sampling_batch = None
        ema_model = deepcopy(model_to_optimize)
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=ema_model)
        if not os.path.exists(latest_cm_path):
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):

                        t1 = time.time()
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch
                    
                        # compute loss
                        t1_1 = time.time()
                        if getattr(model_to_optimize, 'is_flow', False):
                            raw_loss, loss_dict = model_to_optimize.compute_flow_distill_loss(
                                batch, distill2mean=self.cfg.distill2mean)
                        elif self.cfg.distill_loss_type == 'back_up':
                            raw_loss, loss_dict = model_to_optimize.compute_ddim2cm_loss(batch, distill2mean=self.cfg.distill2mean)
                        elif self.cfg.distill_loss_type == 'action':
                            raw_loss, loss_dict = model_to_optimize.compute_ddim2cm_loss_action(batch, distill2mean=self.cfg.distill2mean)
                        elif self.cfg.distill_loss_type == 'action_same_noise':
                            raw_loss, loss_dict = model_to_optimize.compute_ddim2cm_loss_action_same_noise(batch, distill2mean=self.cfg.distill2mean)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()
                        
                        t1_2 = time.time()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            torch.nn.utils.clip_grad_norm_(model_to_optimize.distilled_model.parameters(), cfg.training.max_grad_norm)
                            cm_optimizer.step()
                            lr_scheduler.step()
                            cm_optimizer.zero_grad(set_to_none=True)
                            if not getattr(model_to_optimize, 'is_flow', False):
                                update_ema(model_to_optimize.target_model.parameters(), model_to_optimize.distilled_model.parameters(), cfg.training.ema_decay)
                        t1_3 = time.time()
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(model_to_optimize)
                        t1_4 = time.time()
                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        t1_5 = time.time()
                        step_log.update(loss_dict)
                        t2 = time.time()
                        
                        if self.verbose:
                            print(f"total one step time: {t2-t1:.3f}")
                            print(f" compute loss time: {t1_2-t1_1:.3f}")
                            print(f" step optimizer time: {t1_3-t1_2:.3f}")
                            print(f" update ema time: {t1_4-t1_3:.3f}")
                            print(f" logging time: {t1_5-t1_4:.3f}")

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = model_to_optimize
                # if cfg.training.use_ema:
                #     policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 and self.RUN_ROLLOUT and env_runner is not None:
                    t3 = time.time()
                    # runner_log = env_runner.run(policy, dataset=dataset)
                    # log_data = self.eval(eval_times=self.cfg.unio4.eval_times)
                    runner_log = env_runner.run(policy, use_cm=True, distill2mean=self.cfg.distill2mean)
                    t4 = time.time()
                    # print(f"rollout time: {t4-t3:.3f}")
                    # log all
                    step_log.update(runner_log)  
                # run validation
                if (self.epoch % cfg.training.val_every) == 0 and self.RUN_VALIDATION:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
                                loss, loss_dict = model_to_optimize.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(self.device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        if self.cfg.no_pre_action:
                            gt_action = gt_action[:, self.cfg.n_obs_steps - 1 :]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                if env_runner is None:
                    step_log['test_mean_score'] = - train_loss
                    
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_ckpt:
                    if phase == 'after_dp':
                        # checkpointing
                        if cfg.checkpoint.save_last_ckpt:
                            self.save_checkpoint(tag='latest_cm')
                        if cfg.checkpoint.save_last_snapshot:
                            self.save_snapshot(tag='latest_cm')

                        # sanitize metric names
                        metric_dict = dict()
                        for key, value in step_log.items():
                            new_key = key.replace('/', '_')
                            metric_dict[new_key] = value
                            metric_dict['type'] = 'cm'
                        
                        # We can't copy the last checkpoint here
                        # since save_checkpoint uses threads.
                        # therefore at this point the file might have been empty!
                        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                        if cfg.only_bc:
                            self.unio4.set_policy(self.model); self.unio4.set_old_policy()
                            os.makedirs(os.path.join(self.output_dir, 'best_cm'), exist_ok=True)
                            self.unio4.save(os.path.join(self.output_dir, 'best_cm'))
                    os.makedirs(os.path.join(self.offline_best_path, '_{}'.format(str(self.epoch))), exist_ok=True)
                    model_to_optimize.save(os.path.join(os.path.join(self.offline_best_path, '_{}'.format(str(self.epoch)))))
                    os.makedirs(os.path.join(self.offline_best_path, 'last'), exist_ok=True)
                    model_to_optimize.save(os.path.join(self.offline_best_path, 'last'))
                    # if phase == 'after_offline':
                    #     self.unio4.save(os.path.join(os.path.join(self.offline_best_path, '_{}'.format(str(self.epoch)))))

                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                self.global_step += 1
                self.epoch += 1
                del step_log
            
            if phase == 'after_dp':
                self.save_checkpoint(tag='latest_cm')
            os.makedirs(os.path.join(self.offline_best_path, 'last'), exist_ok=True)
            model_to_optimize.save(os.path.join(self.offline_best_path, 'last'))
        # After flow distillation: promote student to default model and save promoted checkpoint
        if getattr(model_to_optimize, 'is_flow', False):
            model_to_optimize.promote_distilled_model()
            os.makedirs(os.path.join(self.offline_best_path, 'last'), exist_ok=True)
            model_to_optimize.save(os.path.join(self.offline_best_path, 'last'))
            cprint('saved promoted student checkpoint to {}'.format(
                os.path.join(self.offline_best_path, 'last')), 'green')
        # =============================== stage 1-2: end distillation training ===============================
    def eval(self, online=False, eval_times=1, use_cm=False, distill2mean=False, policy_override=None, eval_name='Eval', traj_path=None, data_collect=False):
        # load the latest checkpoint
        cfg = copy.deepcopy(self.cfg)

        env_runner = self.env_runner
        if policy_override is not None:
            policy = policy_override
        elif online:
            policy = self.unio4._policy
        else:
            if cfg.training.use_ema:
                policy = self.ema_model
            else:
                policy = self.model

        # VIB: deterministic eval uses the latent mean; data collection can
        # force stochastic latent sampling for RL rollout diversity.
        force_stochastic_for_collect = bool(
            data_collect and getattr(self.cfg.ppo, 'force_stochastic_online', True)
        )
        saved_force_stochastic = []
        if hasattr(policy, 'obs_encoder') and hasattr(policy.obs_encoder, 'force_stochastic'):
            saved_force_stochastic.append((policy.obs_encoder, policy.obs_encoder.force_stochastic))
            policy.obs_encoder.force_stochastic = force_stochastic_for_collect

        policy.eval()
        eval_env_num = getattr(self.cfg.ppo, 'eval_env_num', 1)
        try:
            run_params = inspect.signature(env_runner.run).parameters
        except (TypeError, ValueError):
            run_params = {}
        run_kwargs = {
            'use_cm': use_cm,
            'distill2mean': distill2mean,
            'eval_env_num': eval_env_num,
            'traj_path': traj_path,
            'data_collect': data_collect,
        }
        run_kwargs = {k: v for k, v in run_kwargs.items() if k in run_params}
        log_data = {'test_mean_score': [], 'mean_returns': []}
        try:
            for _ in range(eval_times):
                runner_log = env_runner.run(policy, **run_kwargs)
                log_data['test_mean_score'].append(runner_log['test_mean_score'])
                log_data['mean_returns'].append(runner_log['mean_returns'])

                cprint(f"---------------- {eval_name} Results --------------", 'magenta')
                for key, value in runner_log.items():
                    if isinstance(value, float):
                        cprint(f"{key}: {value:.4f}", 'magenta')
        finally:
            for encoder, prev_value in reversed(saved_force_stochastic):
                encoder.force_stochastic = prev_value
        log_data['test_mean_score'] = np.mean(log_data['test_mean_score'])
        log_data['mean_returns'] = np.mean(log_data['mean_returns'])
        print(f'{eval_name} average success rates:', np.mean(log_data['test_mean_score']))
        print(f'{eval_name} average rewards:', np.mean(log_data['mean_returns']))
        return log_data

    def value_decay(self, initial_value, total_steps, max_train_steps, min_value=0.1):
        value_now = initial_value * (1 - total_steps / max_train_steps)
        return np.clip(value_now, a_min=min_value, a_max=None)
    def load_online_checkpoints(self, online_ft_path, iql=None, value_net=None, ema=None):
        self.online_update_num_path = os.path.join(online_ft_path, 'update_num.txt')
        update_num = np.loadtxt(self.online_update_num_path, dtype=int)
        self.online_policy_cp_path = os.path.join(online_ft_path, 'policy', 'update_{}'.format(update_num))
        self.online_value_cp_path = os.path.join(online_ft_path, 'value', 'update_{}'.format(update_num))
        self.online_iql_cp_path = os.path.join(online_ft_path, 'iql', 'update_{}'.format(update_num))
        self.online_lr_cp_path = os.path.join(online_ft_path, 'lr', 'update_{}'.format(update_num), 'lr.txt')
        self.online_distilled_cp_path = os.path.join(online_ft_path, 'distilled', 'update_{}'.format(update_num))
        self.online_update_num = np.loadtxt(self.online_update_num_path, dtype=int)
        self.unio4.load(self.online_policy_cp_path) # 1. load ddim policy
        if hasattr(self.unio4, "critic"):
            self.unio4.load_critic(self.online_value_cp_path) # 2. load value net
        else:
            value_net.load_state_dict(torch.load(os.path.join(self.online_value_cp_path, 'critic.pth'))) #  load value net
            cprint('2. load value net from {}'.format(self.online_value_cp_path), 'green')
        if self.cfg.ppo.iql_ft:
            online_encoder_path = os.path.join(self.online_iql_cp_path, 'encoder.pth')
            if not os.path.exists(online_encoder_path):
                if not self.cfg.ppo.fix_iql_encoder:
                    raise FileNotFoundError(
                        'trainable online IQL encoder checkpoint is required: {}'.format(online_encoder_path)
                    )
                online_encoder_path = None
            iql.load(
            v_path=os.path.join(self.online_iql_cp_path, 'value.pth'),
            q_path=os.path.join(self.online_iql_cp_path, 'Q_bc.pth'),
            encoder_path=online_encoder_path,
            force_load=online_encoder_path is not None
            ) # 3. load iql
            cprint('3. load iql from {}'.format(self.online_iql_cp_path), 'green')
        if self.cfg.distill_phase == 'online':
            self.unio4._policy.distilled_model.load_state_dict(torch.load(os.path.join(self.online_distilled_cp_path, 'distilled.pth'))) # 4. load distilled model
            cprint('4. load distilled model from {}'.format(self.online_distilled_cp_path), 'green')
        lr_a, lr_c = np.loadtxt(self.online_lr_cp_path, dtype=float) # 5. load learning rate
        cprint('5. load learning rate from {}'.format(self.online_lr_cp_path), 'green')
        self.cfg.ppo.lr_a, self.cfg.ppo.lr_c = float(lr_a), float(lr_c)
        cprint('load online checkpoint from {}, and actor and critic learning rate is {} and {}'.format(online_ft_path, update_num, self.cfg.ppo.lr_a, self.cfg.ppo.lr_c), 'green')
        # 6. load EMA if available
        if self.cfg.training.use_ema and self.ema_model is not None:
            ema_cp_path = os.path.join(online_ft_path, 'ema', 'update_{}'.format(update_num))
            if os.path.exists(ema_cp_path):
                self.ema_model.load(ema_cp_path)
                cprint('6. load EMA from {}'.format(ema_cp_path), 'green')
                if ema is not None:
                    ema_step_path = os.path.join(ema_cp_path, 'optimization_step.txt')
                    if os.path.exists(ema_step_path):
                        ema.optimization_step = int(np.loadtxt(ema_step_path, dtype=int))
                        cprint('7. load EMA optimization_step from {}'.format(ema_step_path), 'green')
                    else:
                        ema.optimization_step = 0
                        cprint('7. EMA optimization_step not found, reset to 0 for backward compatibility', 'yellow')
            else:
                ema_state = self.ema_model.state_dict()
                policy_state = self.unio4._policy.state_dict()
                filtered_state = {k: v for k, v in policy_state.items() if k in ema_state}
                self.ema_model.load_state_dict(filtered_state, strict=False)
                cprint('6. EMA checkpoint not found, synced from policy', 'yellow')
        if value_net is not None:
            return iql, value_net
        else:
            return iql, self.unio4.critic

    def save_online_checkpoints(self, online_ft_path, update_num, iql=None, ema=None):
    

        self.online_update_num_path = os.path.join(online_ft_path, 'update_num.txt')
        self.online_policy_cp_path = os.path.join(online_ft_path, 'policy', 'update_{}'.format(update_num))
        self.online_value_cp_path = os.path.join(online_ft_path, 'value', 'update_{}'.format(update_num))
        self.online_iql_cp_path = os.path.join(online_ft_path, 'iql', 'update_{}'.format(update_num))
        self.online_distilled_cp_path = os.path.join(online_ft_path, 'distilled', 'update_{}'.format(update_num))
        self.online_lr_cp_path = os.path.join(online_ft_path, 'lr', 'update_{}'.format(update_num))
        os.makedirs(self.online_policy_cp_path, exist_ok=True)
        os.makedirs(self.online_value_cp_path, exist_ok=True)
        os.makedirs(self.online_iql_cp_path, exist_ok=True)
        os.makedirs(self.online_lr_cp_path, exist_ok=True)
        os.makedirs(self.online_distilled_cp_path, exist_ok=True)
        np.savetxt(self.online_update_num_path, [update_num], fmt='%d', delimiter=',')
        self.unio4.save(self.online_policy_cp_path) # save policy
        self.unio4.save_critic(self.online_value_cp_path) # save value
        if self.cfg.ppo.iql_ft:
            iql.save(
            v_path=os.path.join(self.online_iql_cp_path, 'value.pth'),
            q_path=os.path.join(self.online_iql_cp_path, 'Q_bc.pth'),
            encoder_path=os.path.join(self.online_iql_cp_path, 'encoder.pth')
            ) # save iql
        if self.cfg.distill_phase == 'online':
            torch.save(self.unio4._policy.distilled_model.state_dict(), os.path.join(self.online_distilled_cp_path, 'distilled.pth'))
        np.savetxt(os.path.join(self.online_lr_cp_path, 'lr.txt'), [self.unio4.lr_a, self.unio4.lr_c],fmt='%.10f', delimiter=',') # save learning rate
        if self.cfg.training.use_ema and self.ema_model is not None:
            ema_cp_path = os.path.join(online_ft_path, 'ema', 'update_{}'.format(update_num))
            os.makedirs(ema_cp_path, exist_ok=True)
            self.ema_model.save(ema_cp_path)
            if ema is not None:
                np.savetxt(os.path.join(ema_cp_path, 'optimization_step.txt'),
                           [ema.optimization_step], fmt='%d', delimiter=',')
        print('save online checkpoint to {}'.format(online_ft_path, update_num))
        
    # @property
    def output_dir(self):
        output_dir = self._output_dir
        if output_dir is None:
            output_dir = HydraConfig.get().runtime.output_dir
        return output_dir
    
    def sample_batch(self, batch_size: int = 512):
        # all_data = self.all_data
        # data_idxes = torch.from_numpy(np.random.randint(0, all_data['action'].shape[0], size=batch_size))
        # batch = dict_apply(all_data, lambda x: x[data_idxes]) 
        batch = next(iter(self.train_dataloader))
        return dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))
    
    def sample_finetune_batch(self):
        """Sample a batch from the finetuning dataloader with configurable batch size."""
        try:
            batch = next(self.finetune_dataloader_iter)
        except StopIteration:
            # Reset iterator when exhausted
            self.finetune_dataloader_iter = iter(self.finetune_dataloader)
            batch = next(self.finetune_dataloader_iter)
        return dict_apply(batch, lambda x: x.to(self.device, non_blocking=True))

    def save_checkpoint(self, path=None, tag='latest', 
            exclude_keys=None,
            include_keys=None,
            use_thread=False):
        if path is None:
            path = pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        else:
            path = pathlib.Path(path)
        if exclude_keys is None:
            exclude_keys = tuple(self.exclude_keys)
        if include_keys is None:
            include_keys = tuple(self.include_keys) + ('_output_dir',)

        path.parent.mkdir(parents=False, exist_ok=True)
        payload = {
            'cfg': self.cfg,
            'state_dicts': dict(),
            'pickles': dict()
        } 

        for key, value in self.__dict__.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                # modules, optimizers and samplers etc
                if key not in exclude_keys:
                    if use_thread:
                        payload['state_dicts'][key] = _copy_to_cpu(value.state_dict())
                    else:
                        payload['state_dicts'][key] = value.state_dict()
            elif key in include_keys:
                payload['pickles'][key] = dill.dumps(value)
        if use_thread:
            self._saving_thread = threading.Thread(
                target=lambda : torch.save(payload, path.open('wb'), pickle_module=dill))
            self._saving_thread.start()
        else:
            torch.save(payload, path.open('wb'), pickle_module=dill)
        
        del payload
        torch.cuda.empty_cache()
        print(f"Checkpoint saved to {path}")
        return str(path.absolute())
    
    def get_pretrained_model_path(self, stage1_model_name):
        # given a stage1 model name, return the path to the pretrained model
        data_folder_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'third_party', 'VRL3', 'src', 'vrl3data')
        model_folder_path = os.path.join(data_folder_path, "trained_models")
        model_path = os.path.join(model_folder_path, stage1_model_name + '_checkpoint.pth.tar')
        return model_path

    def get_checkpoint_path(self, tag='latest'):
        if tag=='latest' or tag=='latest_cm':
            return pathlib.Path(self.output_dir).joinpath('checkpoints', f'{tag}.ckpt')
        elif tag=='best': 
            # the checkpoints are saved as format: epoch={}-test_mean_score={}.ckpt
            # find the best checkpoint
            checkpoint_dir = pathlib.Path(self.output_dir).joinpath('checkpoints')
            all_checkpoints = os.listdir(checkpoint_dir)
            best_ckpt = None
            best_score = -1e10
            for ckpt in all_checkpoints:
                if 'latest' in ckpt:
                    continue
                score = float(ckpt.split('test_mean_score=')[1].split('.ckpt')[0])
                if score > best_score:
                    best_ckpt = ckpt
                    best_score = score
            return pathlib.Path(self.output_dir).joinpath('checkpoints', best_ckpt)
        else:
            raise NotImplementedError(f"tag {tag} not implemented")
            
            

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if exclude_keys is None:
            exclude_keys = tuple()
        if include_keys is None:
            include_keys = payload['pickles'].keys()

        for key, value in payload['state_dicts'].items():
            if key not in exclude_keys and key in self.__dict__:
                try:
                    self.__dict__[key].load_state_dict(value, **kwargs)
                except (RuntimeError, ValueError) as e:
                    print(f"Warning: Ignoring keys in state_dict for {key}: {e}")
                    # 特别处理optimizer相关的错误
                    if 'optimizer' in key.lower() or 'optim' in key.lower():
                        print(f"Skipping optimizer {key} due to parameter group mismatch")
                    continue
            else:
                print(f"Warning: Skipping key '{key}' - not found in model or excluded.")

        for key in include_keys:
            if key in payload['pickles'] and key in self.__dict__:
                self.__dict__[key] = dill.loads(payload['pickles'][key])
            else:
                print(f"Warning: Skipping pickle '{key}' - not found in payload or model.")
    def load_checkpoint(self, path=None, tag='latest',
            exclude_keys=None, 
            include_keys=None, 
            **kwargs):
        if path is None:
            path = self.get_checkpoint_path(tag=tag)
        else:
            path = pathlib.Path(path)
        payload = torch.load(path.open('rb'), pickle_module=dill, map_location='cpu')
        self.load_payload(payload, 
            exclude_keys=exclude_keys, 
            include_keys=include_keys)
        return payload
    
    @classmethod
    def create_from_checkpoint(cls, path, 
            exclude_keys=None, 
            include_keys=None,
            **kwargs):
        payload = torch.load(open(path, 'rb'), pickle_module=dill)
        instance = cls(payload['cfg'])
        instance.load_payload(
            payload=payload, 
            exclude_keys=exclude_keys,
            include_keys=include_keys,
            **kwargs)
        return instance

    def save_snapshot(self, tag='latest'):
        """
        Quick loading and saving for reserach, saves full state of the workspace.

        However, loading a snapshot assumes the code stays exactly the same.
        Use save_checkpoint for long-term storage.
        """
        path = pathlib.Path(self.output_dir).joinpath('snapshots', f'{tag}.pkl')
        path.parent.mkdir(parents=False, exist_ok=True)
        torch.save(self, path.open('wb'), pickle_module=dill)
        return str(path.absolute())
    
    @classmethod
    def create_from_snapshot(cls, path):
        return torch.load(open(path, 'rb'), pickle_module=dill)
    

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'rl_100', 'config'))
)
def main(cfg):

    workspace = TrainDP3Workspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
