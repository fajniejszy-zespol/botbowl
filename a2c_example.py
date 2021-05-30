import os
import gym
from ffai import FFAIEnv
from torch.autograd import Variable
import torch.optim as optim
from multiprocessing import Process, Pipe
from ffai.ai.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
from a2c_agent import A2CAgent
from vecEnv import VecEnv
from cnn_policy import CNNPolicy
import ffai
import random
import time
import numpy as np
from uuid import uuid1
import pathlib
import warnings

# Blokuje wyświetlanie FutureWarningów
warnings.simplefilter(action='ignore', category=FutureWarning)

# Training configuration
num_steps = 1000000
num_processes = 8
steps_per_update = 20
learning_rate = 0.001
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.05
log_interval = 500
save_interval = 1000
ppcg = True

# Environment
env_name = "FFAI-1-v3"
#env_name = "FFAI-3-v3"
#num_steps = 5000000 # Increase training time
#log_interval = 1000
#save_interval = 2000
#env_name = "FFAI-5-v3"
#num_steps = 100000000 # Increase training time
#log_interval = 1000
#save_interval = 5000
# env_name = "FFAI-v3"
reset_steps = 5000  # The environment is reset after this many steps it gets stuck

# Self-play
selfplay = True  # Use this to enable/disable self-play
selfplay_window = 3
selfplay_save_steps = int(num_steps / 10)
selfplay_swap_steps = int(selfplay_save_steps / (2*selfplay_window))

# Architecture
num_hidden_nodes = 128
num_cnn_kernels = [32, 64]

# --- Reward function ---
rewards_own = {
    OutcomeType.TOUCHDOWN: 1,
    OutcomeType.SUCCESSFUL_CATCH: 0.1,
    OutcomeType.INTERCEPTION: 0.2,
    OutcomeType.SUCCESSFUL_PICKUP: 0.1,
    OutcomeType.FUMBLE: -0.1,
    OutcomeType.KNOCKED_DOWN: -0.1,
    OutcomeType.KNOCKED_OUT: -0.2,
    OutcomeType.CASUALTY: -0.5
}
rewards_opp = {
    OutcomeType.TOUCHDOWN: -1,
    OutcomeType.SUCCESSFUL_CATCH: -0.1,
    OutcomeType.INTERCEPTION: -0.2,
    OutcomeType.SUCCESSFUL_PICKUP: -0.1,
    OutcomeType.FUMBLE: 0.1,
    OutcomeType.KNOCKED_DOWN: 0.1,
    OutcomeType.KNOCKED_OUT: 0.2,
    OutcomeType.CASUALTY: 0.5
}
ball_progression_reward = 0.005


class Memory(object):
    def __init__(self, steps_per_update, num_processes, spatial_obs_shape, non_spatial_obs_shape, action_space):
        self.spatial_obs = torch.zeros(steps_per_update + 1, num_processes, *spatial_obs_shape)
        self.non_spatial_obs = torch.zeros(steps_per_update + 1, num_processes, *non_spatial_obs_shape)
        self.rewards = torch.zeros(steps_per_update, num_processes, 1)
        self.value_predictions = torch.zeros(steps_per_update + 1, num_processes, 1)
        self.returns = torch.zeros(steps_per_update + 1, num_processes, 1)
        action_shape = 1
        self.actions = torch.zeros(steps_per_update, num_processes, action_shape)
        self.actions = self.actions.long()
        self.masks = torch.ones(steps_per_update + 1, num_processes, 1)
        self.action_masks = torch.zeros(steps_per_update + 1, num_processes, action_space, dtype=torch.uint8)

    def cuda(self):
        self.spatial_obs = self.spatial_obs.cuda()
        self.non_spatial_obs = self.non_spatial_obs.cuda()
        self.rewards = self.rewards.cuda()
        self.value_predictions = self.value_predictions.cuda()
        self.returns = self.returns.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()
        self.action_masks = self.action_masks.cuda()

    def insert(self, step, spatial_obs, non_spatial_obs, action, value_pred, reward, mask, action_masks):
        self.spatial_obs[step + 1].copy_(spatial_obs)
        self.non_spatial_obs[step + 1].copy_(non_spatial_obs)
        self.actions[step].copy_(action)
        self.value_predictions[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step].copy_(mask)
        self.action_masks[step].copy_(action_masks)

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * gamma * self.masks[step] + self.rewards[step]


def main():
    es = [make_env(i) for i in range(num_processes)]
    envs = VecEnv([es[i] for i in range(num_processes)])
    starttime = time.time()
    spatial_obs_space = es[0].observation_space.spaces['board'].shape
    board_dim = (spatial_obs_space[1], spatial_obs_space[2])
    board_squares = spatial_obs_space[1] * spatial_obs_space[2]

    non_spatial_obs_space = es[0].observation_space.spaces['state'].shape[0] + es[0].observation_space.spaces['procedures'].shape[0] + es[0].observation_space.spaces['available-action-types'].shape[0]
    non_spatial_action_types = FFAIEnv.simple_action_types + FFAIEnv.defensive_formation_action_types + FFAIEnv.offensive_formation_action_types
    num_non_spatial_action_types = len(non_spatial_action_types)
    spatial_action_types = FFAIEnv.positional_action_types
    num_spatial_action_types = len(spatial_action_types)
    num_spatial_actions = num_spatial_action_types * spatial_obs_space[1] * spatial_obs_space[2]
    action_space = num_non_spatial_action_types + num_spatial_actions

    def compute_action_masks(observations):
        masks = []
        m = False
        for ob in observations:
            mask = np.zeros(action_space)
            i = 0
            for action_type in non_spatial_action_types:
                mask[i] = ob['available-action-types'][action_type.name]
                i += 1
            for action_type in spatial_action_types:
                if ob['available-action-types'][action_type.name] == 0:
                    mask[i:i+board_squares] = 0
                elif ob['available-action-types'][action_type.name] == 1:
                    position_mask = ob['board'][f"{action_type.name.replace('_', ' ').lower()} positions"]
                    position_mask_flatten = np.reshape(position_mask, (1, board_squares))
                    for j in range(board_squares):
                        mask[i + j] = position_mask_flatten[0][j]
                i += board_squares
            assert 1 in mask
            if m:
                print(mask)
            masks.append(mask)
        return masks

    def compute_action(action_idx):
        if action_idx < len(non_spatial_action_types):
            return non_spatial_action_types[action_idx], 0, 0
        spatial_idx = action_idx - num_non_spatial_action_types
        spatial_pos_idx = spatial_idx % board_squares
        spatial_y = int(spatial_pos_idx / board_dim[1])
        spatial_x = int(spatial_pos_idx % board_dim[1])
        spatial_action_type_idx = int(spatial_idx / board_squares)
        spatial_action_type = spatial_action_types[spatial_action_type_idx]
        return spatial_action_type, spatial_x, spatial_y

    # Clear log file
    try:
        os.remove(log_filename)
    except OSError:
        pass

    # MODEL
    ac_agent = CNNPolicy(spatial_obs_space, non_spatial_obs_space, hidden_nodes=num_hidden_nodes, kernels=num_cnn_kernels, actions=action_space)

    # OPTIMIZER
    optimizer = optim.RMSprop(ac_agent.parameters(), learning_rate)

    # MEMORY STORE
    memory = Memory(steps_per_update, num_processes, spatial_obs_space, (1, non_spatial_obs_space), action_space)

    # PPCG
    difficulty = 0.0
    dif_delta = 0.01

    # Reset environments
    obs = envs.reset(difficulty)
    spatial_obs, non_spatial_obs = update_obs(obs)

    # Add obs to memory
    memory.spatial_obs[0].copy_(spatial_obs)
    memory.non_spatial_obs[0].copy_(non_spatial_obs)

    # Variables for storing stats
    all_updates = 0
    all_episodes = 0
    all_steps = 0
    episodes = 0
    proc_rewards = np.zeros(num_processes)
    proc_tds = np.zeros(num_processes)
    proc_tds_opp = np.zeros(num_processes)
    episode_rewards = []
    episode_tds = []
    episode_tds_opp = []
    wins = []
    value_losses = []
    policy_losses = []
    log_updates = []
    log_episode = []
    log_steps = []
    log_win_rate = []
    log_td_rate = []
    log_td_rate_opp = []
    log_mean_reward = []
    log_difficulty = []

    # self-play
    selfplay_next_save = selfplay_save_steps
    selfplay_next_swap = selfplay_swap_steps
    selfplay_models = 0
    if selfplay:
        model_path = f"{MODEL_ROOT}/{model_name}_selfplay_0"
        torch.save(ac_agent, model_path)
        envs.swap(A2CAgent(name=f"selfplay-0", env_name=env_name, filename=model_path))
        selfplay_models += 1

    renderer = ffai.Renderer()

    while all_steps < num_steps:

        for step in range(steps_per_update):

            action_masks = compute_action_masks(obs)
            action_masks = torch.tensor(action_masks, dtype=torch.bool)

            values, actions = ac_agent.act(
                Variable(memory.spatial_obs[step]),
                Variable(memory.non_spatial_obs[step]),
                Variable(action_masks))

            action_objects = []

            for action in actions:
                action_type, x, y = compute_action(action.numpy()[0])
                action_object = {
                    'action-type': action_type,
                    'x': x,
                    'y': y
                }
                action_objects.append(action_object)

            obs, env_reward, shaped_reward, tds_scored, tds_opp_scored, done, info = envs.step(action_objects, difficulty=difficulty)
            #envs.render()

            '''
            for j in range(len(obs)):
                ob = obs[j]
                renderer.render(ob, j)
            '''

            reward = torch.from_numpy(np.expand_dims(np.stack(env_reward), 1)).float()
            shaped_reward = torch.from_numpy(np.expand_dims(np.stack(shaped_reward), 1)).float()
            r = reward.numpy()
            sr = shaped_reward.numpy()
            for i in range(num_processes):
                proc_rewards[i] += sr[i]
                proc_tds[i] += tds_scored[i]
                proc_tds_opp[i] += tds_opp_scored[i]

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            dones = masks.squeeze()
            episodes += num_processes - int(dones.sum().item())
            for i in range(num_processes):
                if done[i]:
                    if r[i] > 0:
                        wins.append(1)
                        difficulty += dif_delta
                    elif r[i] < 0:
                        wins.append(0)
                        difficulty -= dif_delta
                    else:
                        wins.append(0.5)
                        difficulty -= dif_delta
                    if ppcg:
                        difficulty = min(1.0, max(0, difficulty))
                    else:
                        difficulty = 1
                    episode_rewards.append(proc_rewards[i])
                    episode_tds.append(proc_tds[i])
                    episode_tds_opp.append(proc_tds_opp[i])
                    proc_rewards[i] = 0
                    proc_tds[i] = 0
                    proc_tds_opp[i] = 0

            # Update the observations returned by the environment
            spatial_obs, non_spatial_obs = update_obs(obs)

            # insert the step taken into memory
            memory.insert(step, spatial_obs, non_spatial_obs,
                          actions.data, values.data, shaped_reward, masks, action_masks)

        next_value = ac_agent(Variable(memory.spatial_obs[-1], requires_grad=False), Variable(memory.non_spatial_obs[-1], requires_grad=False))[0].data

        # Compute returns
        memory.compute_returns(next_value, gamma)

        spatial = Variable(memory.spatial_obs[:-1])
        spatial = spatial.view(-1, *spatial_obs_space)
        non_spatial = Variable(memory.non_spatial_obs[:-1])
        non_spatial = non_spatial.view(-1, non_spatial.shape[-1])

        actions = Variable(torch.LongTensor(memory.actions.view(-1, 1)))
        actions_mask = Variable(memory.action_masks[:-1])

        # Evaluate the actions taken
        action_log_probs, values, dist_entropy = ac_agent.evaluate_actions(spatial, non_spatial, actions, actions_mask)

        values = values.view(steps_per_update, num_processes, 1)
        action_log_probs = action_log_probs.view(steps_per_update, num_processes, 1)

        advantages = Variable(memory.returns[:-1]) - values
        value_loss = advantages.pow(2).mean()
        #value_losses.append(value_loss)

        # Compute loss
        action_loss = -(Variable(advantages.data) * action_log_probs).mean()
        #policy_losses.append(action_loss)

        optimizer.zero_grad()

        total_loss = (value_loss * value_loss_coef + action_loss - dist_entropy * entropy_coef)
        total_loss.backward()

        nn.utils.clip_grad_norm_(ac_agent.parameters(), max_grad_norm)

        optimizer.step()

        memory.non_spatial_obs[0].copy_(memory.non_spatial_obs[-1])
        memory.spatial_obs[0].copy_(memory.spatial_obs[-1])

        # Updates
        all_updates += 1
        # Episodes
        all_episodes += episodes
        episodes = 0
        # Steps
        all_steps += num_processes * steps_per_update

        # Self-play save
        if selfplay and all_steps >= selfplay_next_save:
            selfplay_next_save = max(all_steps+1, selfplay_next_save+selfplay_save_steps)
            model_path = f"{MODEL_ROOT}/{model_name}_selfplay_{selfplay_models}"
            print(f"Saving {model_path}")
            torch.save(ac_agent, model_path)
            selfplay_models += 1

        # Self-play swap
        if selfplay and all_steps >= selfplay_next_swap:
            selfplay_next_swap = max(all_steps + 1, selfplay_next_swap+selfplay_swap_steps)
            lower = max(0, selfplay_models-1-(selfplay_window-1))
            i = random.randint(lower, selfplay_models-1)
            model_path = f"{MODEL_ROOT}/{model_name}_selfplay_{i}"
            print(f"Swapping opponent to {model_path}")
            envs.swap(A2CAgent(name=f"selfplay-{i}", env_name=env_name, filename=model_path))

        # Save
        if all_updates % save_interval == 0 and len(episode_rewards) >= num_processes:
            # Save to files
            with open(log_filename, "a") as myfile:
                myfile.write(log_to_file)

        # Logging
        if all_updates % log_interval == 0 and len(episode_rewards) >= num_processes:
            td_rate = np.mean(episode_tds)
            td_rate_opp = np.mean(episode_tds_opp)
            episode_tds.clear()
            episode_tds_opp.clear()
            mean_reward = np.mean(episode_rewards)
            episode_rewards.clear()
            win_rate = np.mean(wins)
            wins.clear()
            #mean_value_loss = np.mean(value_losses)
            #mean_policy_loss = np.mean(policy_losses)    
            
            log_updates.append(all_updates)
            log_episode.append(all_episodes)
            log_steps.append(all_steps)
            log_win_rate.append(win_rate)
            log_td_rate.append(td_rate)
            log_td_rate_opp.append(td_rate_opp)
            log_mean_reward.append(mean_reward)
            log_difficulty.append(difficulty)
            timer = time.time() - starttime
            log = "Up: {}, Ep: {}, Steps: {}, Win: {:.2f}, TD: {:.2f}, TD_opp: {:.2f}, Rev: {:.3f}, Diff: {:.2f}, Time: {:.2f}" \
                .format(all_updates, all_episodes, all_steps, win_rate, td_rate, td_rate_opp, mean_reward, difficulty, timer)

            log_to_file = "{}, {}, {}, {}, {}, {}, {}\n" \
                .format(all_updates, all_episodes, all_steps, win_rate, td_rate, td_rate_opp, mean_reward, difficulty)

            print(log)

            episodes = 0
            value_losses.clear()
            policy_losses.clear()

            # Save model
            torch.save(ac_agent, f"{MODEL_ROOT}/" + model_name)
            
            # plot
            n = 3
            if ppcg:
                n += 1
            fig, axs = plt.subplots(1, n, figsize=(4*n, 5))
            axs[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axs[0].plot(log_steps, log_mean_reward)
            axs[0].set_title('Reward')
            #axs[0].set_ylim(bottom=0.0)
            axs[0].set_xlim(left=0)
            axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axs[1].plot(log_steps, log_td_rate, label="Learner")
            axs[1].set_title('TD/Episode')
            axs[1].set_ylim(bottom=0.0)
            axs[1].set_xlim(left=0)
            if selfplay:
                axs[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                axs[1].plot(log_steps, log_td_rate_opp, color="red", label="Opponent")
            axs[2].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axs[2].plot(log_steps, log_win_rate)
            axs[2].set_title('Win rate')            
            axs[2].set_yticks(np.arange(0, 1.001, step=0.1))
            axs[2].set_xlim(left=0)
            if ppcg:
                axs[3].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                axs[3].plot(log_steps, log_difficulty)
                axs[3].set_title('Difficulty')
                axs[3].set_yticks(np.arange(0, 1.001, step=0.1))
                axs[3].set_xlim(left=0)
            fig.tight_layout()
            fig.savefig(f"{PLOT_ROOT}/{model_name}{'_selfplay' if selfplay else ''}.png")
            plt.close('all')
            
            # plot 2
            n = 1
            if selfplay:
                n += 1
            if ppcg:
                n += 3
            fig2, axs2 = plt.subplots(1, n, figsize=(4*n, 5))
            axs2[0].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
            axs2[0].semilogy(log_steps, np.divide(log_win_rate, log_mean_reward))
            axs2[0].set_title('Win rate/Reward')
            axs2[0].set_xlim(left=0)
            
            if selfplay:
                axs2[1].ticklabel_format(axis="x", style="sci", scilimits=(0,0))
                axs2[1].semilogy(log_steps, np.divide(log_td_rate_opp, log_td_rate))
                axs2[1].set_title('TD_rate/TD_rate_opp')
                axs2[1].set_xlim(left=0)
            
            if ppcg:
                axs2[2].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                axs2[2].semilogy(log_steps, np.divide(log_td_rate, log_difficulty))
                axs2[2].set_title('TD_rate/Difficulty')
                axs2[2].set_xlim(left=0)
                    
                axs2[3].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                axs2[3].semilogy(log_steps, np.divide(log_win_rate, log_difficulty))
                axs2[3].set_title('Win rate/Difficulty')
                axs2[3].set_xlim(left=0)
                
                axs2[4].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
                axs2[4].semilogy(log_steps, np.divide(log_mean_reward, log_difficulty))
                axs2[4].set_title('Reward/Difficulty')
                axs2[4].set_xlim(left=0)
                
            fig2.tight_layout()
            fig2.savefig(f"{PLOT_ROOT}/{model_name}_new_plots_{'_selfplay' if selfplay else ''}.png")
            plt.close('all')


    torch.save(ac_agent, f"{MODEL_ROOT}/" + model_name)
    envs.close()


def update_obs(observations):
    """
    Takes the observation returned by the environment and transforms it to an numpy array that contains all of
    the feature layers and non-spatial info
    """
    spatial_obs = []
    non_spatial_obs = []

    for obs in observations:
        '''
        for k, v in obs['board'].items():
            print(k)
            print(v)
        '''
        spatial_ob = np.stack(obs['board'].values())

        state = list(obs['state'].values())
        procedures = list(obs['procedures'].values())
        actions = list(obs['available-action-types'].values())

        non_spatial_ob = np.stack(state+procedures+actions)

        # feature_layers = np.expand_dims(feature_layers, axis=0)
        non_spatial_ob = np.expand_dims(non_spatial_ob, axis=0)

        spatial_obs.append(spatial_ob)
        non_spatial_obs.append(non_spatial_ob)

    return torch.from_numpy(np.stack(spatial_obs)).float(), torch.from_numpy(np.stack(non_spatial_obs)).float()


def make_env(worker_id):
    print("Initializing worker", worker_id, "...")
    env = gym.make(env_name)
    return env

if __name__ == "__main__":
    SESSION_ID = uuid1()
    SESSION_ROOT = f'sessions/{SESSION_ID}'
    LOG_ROOT = f'{SESSION_ROOT}/logs'
    MODEL_ROOT = f'{SESSION_ROOT}/models'
    PLOT_ROOT = f'{SESSION_ROOT}/plots'

    model_name = env_name
    log_filename = f"{LOG_ROOT}/{model_name}.dat"

    # Tworzy wszystkie potrzebne pliki w sesji, z jakiegoś powodu sama funkcja ensure dirs odmawiała współpracy
    pathlib.Path('sessions/' + str(SESSION_ID)).mkdir(parents=True, exist_ok=True)
    pathlib.Path('sessions/' + str(SESSION_ID) + '/logs').mkdir(parents=True, exist_ok=True)
    pathlib.Path('sessions/' + str(SESSION_ID) + '/models').mkdir(parents=True, exist_ok=True)
    pathlib.Path('sessions/' + str(SESSION_ID) + '/plots').mkdir(parents=True, exist_ok=True)

    print(f'Current session id: {SESSION_ID}')
    print(f'\tlogs:   {LOG_ROOT}')
    print(f'\tmodels: {MODEL_ROOT}')
    print(f'\tplots:  {PLOT_ROOT}')
    print()

    main()