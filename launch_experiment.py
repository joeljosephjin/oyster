"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch

# from rl_algo.py
from rlkit.core import logger
import gtimer as gt
from rlkit.data_management.path_builder import PathBuilder


# a lot of different stuff from different environments, big-ness necessary
from rlkit.envs import ENVS
# a lot of wrappers within wrappers, big-ness necessary
from rlkit.envs.wrappers import NormalizedBoxEnv
# also big
from rlkit.torch.sac.policies import TanhGaussianPolicy
# also big
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
# the main file, but its too big too
# from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.sacwb import PEARLSoftActorCritic
# mostly a network thing
from rlkit.torch.sac.agent import PEARLAgent
# it logs a lot o stuff; dont know if its even needed
from rlkit.launchers.launcher_util import setup_logger
# some useful tiny util functions
import rlkit.torch.pytorch_util as ptu
# dict of config stuff
from configs.default import default_config

# seed-ings
import random
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

@click.command()
@click.argument('config', default='./configs/sparse-point-robot.json')
@click.option('--gpu', default=0)
@click.option('--docker', is_flag=True, default=False)
@click.option('--debug', is_flag=True, default=False)
def main(config, gpu, docker, debug):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    variant['util_params']['gpu_id'] = gpu

    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder_input_dim = 2 * obs_dim + action_dim + reward_dim if variant['algo_params']['use_next_obs_in_context'] else obs_dim + action_dim + reward_dim
    context_encoder_output_dim = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
    )
    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )

    # optionally load pre-trained weights
    if variant['path_to_weights'] is not None:
        path = variant['path_to_weights']
        context_encoder.load_state_dict(torch.load(os.path.join(path, 'context_encoder.pth')))
        qf1.load_state_dict(torch.load(os.path.join(path, 'qf1.pth')))
        qf2.load_state_dict(torch.load(os.path.join(path, 'qf2.pth')))
        vf.load_state_dict(torch.load(os.path.join(path, 'vf.pth')))
        # TODO hacky, revisit after model refactor
        algorithm.networks[-2].load_state_dict(torch.load(os.path.join(path, 'target_vf.pth')))
        policy.load_state_dict(torch.load(os.path.join(path, 'policy.pth')))

    # optional GPU mode
    ptu.set_gpu_mode(variant['util_params']['use_gpu'], variant['util_params']['gpu_id'])
    if ptu.gpu_enabled():
        algorithm.to()

    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant['util_params']['debug']
    os.environ['DEBUG'] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = 'debug' if DEBUG else None
    experiment_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir'])

    # optionally save eval trajectories as pkl files
    if variant['algo_params']['dump_eval_paths']:
        pickle_dir = experiment_log_dir + '/eval_trajectories'
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    '''
    meta-training loop
    '''
    # it does nothing; pass
    algorithm.pretrain()
    # save epoch, explo_policy and train_env into a dictionary
    params = algorithm.get_epoch_snapshot(-1)
    
    # ??
    logger.save_itr_params(-1, params)
    # ??
    gt.reset()
    # ??
    gt.set_def_unique(False)
    # ??
    algorithm._current_path_builder = PathBuilder()

    # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
    # ??
    for it_ in gt.timed_for(
            range(algorithm.num_iterations),
            save_itrs=True,
    ):
        # log start time, init explo_path, train_time and log epochs
        algorithm._start_epoch(it_)
        # ??
        algorithm.training_mode(True)
        # at the start of each iteration
        if it_ == 0:
            # print('collecting initial pool of data for train and eval')
            # temp for evaluating
            # 
            for idx in algorithm.train_tasks:
                algorithm.task_idx = idx
                algorithm.env.reset_task(idx)
                algorithm.collect_data(algorithm.num_initial_steps, 1, np.inf)
        # Sample data from train tasks.
        for i in range(algorithm.num_tasks_sample):
            idx = np.random.randint(len(algorithm.train_tasks))
            algorithm.task_idx = idx
            algorithm.env.reset_task(idx)
            algorithm.enc_replay_buffer.task_buffers[idx].clear()

            # collect some trajectories with z ~ prior
            if algorithm.num_steps_prior > 0:
                algorithm.collect_data(algorithm.num_steps_prior, 1, np.inf)
            # collect some trajectories with z ~ posterior
            if algorithm.num_steps_posterior > 0:
                algorithm.collect_data(algorithm.num_steps_posterior, 1, algorithm.update_post_train)
            # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
            if algorithm.num_extra_rl_steps_posterior > 0:
                algorithm.collect_data(algorithm.num_extra_rl_steps_posterior, 1, algorithm.update_post_train, add_to_enc_buffer=False)

        # Sample train tasks and compute gradient updates on parameters.
        for train_step in range(algorithm.num_train_steps_per_itr):
            indices = np.random.choice(algorithm.train_tasks, algorithm.meta_batch)
            # training happens here

            
            algorithm._do_training(indices)
            # i guess embedding means 'z'
            mb_size = algorithm.embedding_mini_batch_size
            # num_updates = len([minibatch1, minibatch2, minibatch3...])
            num_updates = algorithm.embedding_batch_size // mb_size

            # sample context batch
            # get contexts by sampling from replay_buffer
            context_batch = algorithm.sample_context(indices)

            # zero out context and hidden encoder state
            algorithm.agent.clear_z(num_tasks=len(indices))

            # do this in a loop so we can truncate backprop in the recurrent encoder
            for i in range(num_updates):
                # use the i-th context each time
                # from (i)th to (i+1)th mb_size batch
                # c_i <- {c}
                context = context_batch[:, i * mb_size: i * mb_size + mb_size, :]
                # this is the main training step
                # algorithm._take_step(indices, context)
                num_tasks = len(indices)

                # data is (task, batch, feat)
                # s,a,r,n_s,terms <- replay_buffer of tasks
                obs, actions, rewards, next_obs, terms = algorithm.sample_sac(indices)

                # run inference in networks
                # pi,z <- agent(s,c)
                policy_outputs, task_z = algorithm.agent(obs, context)
                # new_a-s, mu, sigma, log_pi
                new_actions, policy_mean, policy_log_std, log_pi = policy_outputs[:4]

                # flattens out the task dimension
                # 2D state. get length and breadth for flattening the state value
                t, b, _ = obs.size()
                # flatten all them (s, a, n_s) normally
                obs = obs.view(t * b, -1); actions = actions.view(t * b, -1); next_obs = next_obs.view(t * b, -1)

                # Q and V networks
                # encoder will only get gradients from Q nets
                # q_{1/2} = qfunction_{1/2}(s, a, z)
                q1_pred = algorithm.qf1(obs, actions, task_z); q2_pred = algorithm.qf2(obs, actions, task_z)
                # v = vfunction(s, z)
                v_pred = algorithm.vf(obs, task_z.detach())
                # get targets for use in V and Q updates
                with torch.no_grad():
                    # target_v = target_v_function(n_s, z)
                    target_v_values = algorithm.target_vf(next_obs, task_z)

                # KL constraint on z if probabilistic
                algorithm.context_optimizer.zero_grad()
                if algorithm.use_information_bottleneck:
                    # compute KL( q(z|c) || r(z) )
                    kl_div = algorithm.agent.compute_kl_div()
                    # compute kl_loss from kl_div multiplied by a constant
                    kl_loss = algorithm.kl_lambda * kl_div
                    # backprop-ing thru the context i guess. i think it updates the q(z|c)
                    kl_loss.backward(retain_graph=True)

                # qf and encoder update (note encoder does not get grads from policy or vf)
                algorithm.qf1_optimizer.zero_grad()
                algorithm.qf2_optimizer.zero_grad()
                # flatten the rewards
                rewards_flat = rewards.view(algorithm.batch_size * num_tasks, -1)
                # scale rewards for Bellman update
                rewards_flat = rewards_flat * algorithm.reward_scale
                # flatten the terms
                terms_flat = terms.view(algorithm.batch_size * num_tasks, -1)
                # calculate q target value using bellman update
                q_target = rewards_flat + (1. - terms_flat) * algorithm.discount * target_v_values
                # calculate q loss for both at the same time
                qf_loss = torch.mean((q1_pred - q_target) ** 2) + torch.mean((q2_pred - q_target) ** 2)
                qf_loss.backward()
                # dunno what qf1 and qf2 are
                algorithm.qf1_optimizer.step()
                algorithm.qf2_optimizer.step()
                # update q(z|c) i guess
                algorithm.context_optimizer.step()

                # compute min Q on the new actions
                # min_q <- _min_q(s, a, z)
                min_q_new_actions = algorithm._min_q(obs, new_actions, task_z)

                # vf update
                v_target = min_q_new_actions - log_pi
                #------update vf
                vf_loss = algorithm.vf_criterion(v_pred, v_target.detach())
                algorithm.vf_optimizer.zero_grad()
                vf_loss.backward()
                algorithm.vf_optimizer.step()
                #------update vf

                algorithm._update_target_network()

                # policy update
                # n.b. policy update includes dQ/da
                log_policy_target = min_q_new_actions

                policy_loss = (
                        log_pi - log_policy_target
                ).mean()

                #------update policy
                mean_reg_loss = algorithm.policy_mean_reg_weight * (policy_mean**2).mean()
                std_reg_loss = algorithm.policy_std_reg_weight * (policy_log_std**2).mean()
                pre_tanh_value = policy_outputs[-1]
                pre_activation_reg_loss = algorithm.policy_pre_activation_weight * (
                    (pre_tanh_value**2).sum(dim=1).mean()
                )
                policy_reg_loss = mean_reg_loss + std_reg_loss + pre_activation_reg_loss
                policy_loss = policy_loss + policy_reg_loss

                algorithm.policy_optimizer.zero_grad()
                policy_loss.backward()
                algorithm.policy_optimizer.step()
                #------update policy

                # stop backprop
                algorithm.agent.detach_z()
            algorithm._n_train_steps_total += 1
        gt.stamp('train')

        algorithm.training_mode(False)

        # eval
        algorithm._try_to_eval(it_)
        gt.stamp('eval')

        algorithm._end_epoch()

if __name__ == "__main__":
    main()

