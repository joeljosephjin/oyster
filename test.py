import gym
from stable_baselines3 import PPO as ALGO # DQN, PPO, A2C
from stable_baselines3.common.monitor import Monitor
import wandb
import os
import json


from rlkit.envs import ENVS
from configs.default import default_config
from rlkit.envs.wrappers import NormalizedBoxEnv


gpu=0
config = './configs/sparse-point-robot.json'

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

variant = default_config
if config:
    with open(os.path.join(config)) as f:
        exp_params = json.load(f)
    variant = deep_update_dict(exp_params, variant)
variant['util_params']['gpu_id'] = gpu

env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
tasks = env.get_all_task_idx()

log_dir = './logs/'

max_steps = 2000
env.set_max_steps(max_steps)


env = Monitor(env, log_dir)


## PPO
# policy, env, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, 
# clip_range=0.2, clip_range_vf=None, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=- 1, 
# target_kl=None, tensorboard_log=log_dir, create_eval_env=False, policy_kwargs=None, verbose=1, seed=None, device='auto', _init_setup_model=True

hyper_params = dict(policy='MlpPolicy', env=env, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, 
clip_range=0.2, clip_range_vf=None, ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, use_sde=False, sde_sample_freq=- 1, 
target_kl=None, tensorboard_log=log_dir, create_eval_env=False, policy_kwargs=None, verbose=1, seed=None, device='auto', _init_setup_model=True)

wandb.init(project="mrl-project", entity="joeljosephjin", sync_tensorboard=True, config=hyper_params)

wandb.config.max_steps = max_steps

model = ALGO(**hyper_params) # tensorboard --logdir ./logs/

model.learn(total_timesteps=int(1e5))

