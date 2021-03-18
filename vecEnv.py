from multiprocessing import Process, Pipe
import numpy as np
import os
import gym
from ffai import FFAIEnv
from torch.autograd import Variable
import torch.optim as optim
from ffai.ai.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import ffai
import random
import time



rewards_own = {
    OutcomeType.TOUCHDOWN: 1,
    OutcomeType.CATCH: 0.1,
    OutcomeType.INTERCEPTION: 0.2,
    OutcomeType.SUCCESSFUL_PICKUP: 0.1,
    OutcomeType.FUMBLE: -0.1,
    OutcomeType.KNOCKED_DOWN: -0.1,
    OutcomeType.KNOCKED_OUT: -0.2,
    OutcomeType.CASUALTY: -0.5
}
rewards_opp = {
    OutcomeType.TOUCHDOWN: -1,
    OutcomeType.CATCH: -0.1,
    OutcomeType.INTERCEPTION: -0.2,
    OutcomeType.SUCCESSFUL_PICKUP: -0.1,
    OutcomeType.FUMBLE: 0.1,
    OutcomeType.KNOCKED_DOWN: 0.1,
    OutcomeType.KNOCKED_OUT: 0.2,
    OutcomeType.CASUALTY: 0.5
}
reset_steps = 5000 
ball_progression_reward = 0.005
def reward_function(env, info, shaped=False):
    r = 0
    for outcome in env.get_outcomes():
        if not shaped and outcome.outcome_type != OutcomeType.TOUCHDOWN:
            continue
        team = None
        if outcome.player is not None:
            team = outcome.player.team
        elif outcome.team is not None:
            team = outcome.team
        if team == env.own_team and outcome.outcome_type in rewards_own:
            r += rewards_own[outcome.outcome_type]
        if team == env.opp_team and outcome.outcome_type in rewards_opp:
            r += rewards_opp[outcome.outcome_type]
    if info['ball_progression'] > 0:
        r += info['ball_progression'] * ball_progression_reward
    return r

def worker(remote, parent_remote, env, worker_id):
    parent_remote.close()

    steps = 0
    tds = 0
    tds_opp = 0
    next_opp = ffai.make_bot('random')

    while True:
        command, data = remote.recv()
        if command == 'step':
            steps += 1
            action, dif = data[0], data[1]
            obs, reward, done, info = env.step(action)
            tds_scored = info['touchdowns'] - tds
            tds = info['touchdowns']
            tds_opp_scored = info['opp_touchdowns'] - tds_opp
            tds_opp = info['opp_touchdowns']
            reward_shaped = reward_function(env, info, shaped=True)
            ball_carrier = env.game.get_ball_carrier()
            # PPCG
            if dif < 1.0:
                if ball_carrier and ball_carrier.team == env.game.state.home_team:
                    extra_endzone_squares = int((1.0 - dif) * 25.0)
                    distance_to_endzone = ball_carrier.position.x - 1
                    if distance_to_endzone <= extra_endzone_squares:
                        #reward_shaped += rewards_own[OutcomeType.TOUCHDOWN]
                        env.game.state.stack.push(Touchdown(env.game, ball_carrier))
            if done or steps >= reset_steps:
                # If we  get stuck or something - reset the environment
                if steps >= reset_steps:
                    print("Max. number of steps exceeded! Consider increasing the number.")
                done = True
                env.opp_actor = next_opp
                obs = env.reset()
                steps = 0
                tds = 0
                tds_opp = 0
            remote.send((obs, reward, reward_shaped, tds_scored, tds_opp_scored, done, info))
        elif command == 'reset':
            dif = data
            steps = 0
            tds = 0
            tds_opp = 0
            env.opp_actor = next_opp
            obs = env.reset()
            # set_difficulty(env, dif)
            remote.send(obs)
        elif command == 'render':
            env.render()
        elif command == 'swap':
            next_opp = data
        elif command == 'close':
            break

class VecEnv():
    def __init__(self, envs):
        """
        envs: list of FFAI environments to run in subprocesses
        """
        self.closed = False
        nenvs = len(envs)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])

        self.ps = [Process(target=worker, args=(work_remote, remote, env, envs.index(env)))
                   for (work_remote, remote, env) in zip(self.work_remotes, self.remotes, envs)]

        for p in self.ps:
            p.daemon = True  # If the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions, difficulty=1.0):
        cumul_rewards = None
        cumul_shaped_rewards = None
        cumul_tds_scored = None
        cumul_tds_opp_scored = None
        cumul_dones = None
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', [action, difficulty]))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, rews_shaped, tds, tds_opp, dones, infos = zip(*results)
        if cumul_rewards is None:
            cumul_rewards = np.stack(rews)
            cumul_shaped_rewards = np.stack(rews_shaped)
            cumul_tds_scored = np.stack(tds)
            cumul_tds_opp_scored = np.stack(tds_opp)
        else:
            cumul_rewards += np.stack(rews)
            cumul_shaped_rewards += np.stack(rews_shaped)
            cumul_tds_scored += np.stack(tds)
            cumul_tds_opp_scored += np.stack(tds_opp)
        if cumul_dones is None:
            cumul_dones = np.stack(dones)
        else:
            cumul_dones |= np.stack(dones)
        return np.stack(obs), cumul_rewards, cumul_shaped_rewards, cumul_tds_scored, cumul_tds_opp_scored, cumul_dones, infos

    def reset(self, difficulty=1.0):
        for remote in self.remotes:
            remote.send(('reset', difficulty))
        return np.stack([remote.recv() for remote in self.remotes])

    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))

    def swap(self, agent):
        for remote in self.remotes:
            remote.send(('swap', agent))

    def close(self):
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    @property
    def num_envs(self):
        return len(self.remotes)