import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

import datetime
import os
import random
from collections import namedtuple, deque
import shutil
import numpy as np

from utils.gym_wrappers import MainGymWrapper
import gym

# from utils.arguments import breakout_args
import time

# GAMMA = breakout_args['gamma']
# MEMORY_SIZE = int(breakout_args['memory_size']/breakout_args['n_participants'])
# BATCH_SIZE = breakout_args['batch_size']
# TRAINING_FREQUENCY = breakout_args['training_frequency']
# TARGET_NETWORK_UPDATE_FREQUENCY = breakout_args['target_network_update_frequency']
# MODEL_PERSISTENCE_UPDATE_FREQUENCY = breakout_args['model_persistence_update_frequency']
# REPLAY_START_SIZE = int(breakout_args['replay_start_size']/breakout_args['n_participants'])
# EXPLORATION_MAX = breakout_args['exploration_max']
# EXPLORATION_MIN = breakout_args['exploration_min']
# EXPLORATION_TEST = breakout_args['exploration_test']
# EXPLORATION_STEPS = breakout_args['exploration_steps']
# EXPLORATION_DECAY = (EXPLORATION_MAX-EXPLORATION_MIN)/EXPLORATION_STEPS

# N_PARTICIPANTS = breakout_args['n_participants']

Transition = namedtuple('Transition',
                        ('current_state', 'action' , 'reward', 'next_state', 'terminal'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class BaseGameModel:

    def __init__(self, action_space):
        self.action_space = action_space

    def get_move(self, state):
        pass

    def move(self, state):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def step_update(self, total_step):
        pass

    def _get_date(self):
        return str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))


class DDQNTrainer(BaseGameModel):

    def __init__(self,action_space, model, optimizer, device, args, noise=None, noise_level=None):
        BaseGameModel.__init__(self,
                               action_space)
        self.GAMMA = args['gamma']
        self.MEMORY_SIZE = int(args['memory_size']/args['n_participants'])
        self.BATCH_SIZE = args['batch_size']
        self.TRAINING_FREQUENCY = args['training_frequency']
        self.TARGET_NETWORK_UPDATE_FREQUENCY = args['target_network_update_frequency']
        self.MODEL_PERSISTENCE_UPDATE_FREQUENCY = args['model_persistence_update_frequency']
        self.REPLAY_START_SIZE = int(args['replay_start_size']/args['n_participants'])
        self.EXPLORATION_MAX = args['exploration_max']
        self.EXPLORATION_MIN = args['exploration_min']
        self.EXPLORATION_TEST = args['exploration_test']
        self.EXPLORATION_STEPS = args['exploration_steps']
        self.EXPLORATION_DECAY = (self.EXPLORATION_MAX-self.EXPLORATION_MIN)/self.EXPLORATION_STEPS
        self.N_PARTICIPANTS = args['n_participants']

        self.n_selected = self.N_PARTICIPANTS
        self.device = device
        self.ddqn = model
        self.optimizer = optimizer
        self.ddqn_target = model
        self._reset_target_network()
        self.epsilon = self.EXPLORATION_MAX
        self.pre_run = 0
        self.loss_record = []

        # personalized memory size/exploration ratio for each clients
        if noise != "memory_size":
            self.memory = ReplayMemory(self.MEMORY_SIZE)
        else:
            self.memory = ReplayMemory(int(self.MEMORY_SIZE * self.N_PARTICIPANTS * noise_level))

        if noise != "exploration":
            self.epsilon = self.EXPLORATION_MAX
        else:
            self.epsilon = noise_level

        self.noise_level = noise_level
        self.noise = noise

    def move(self, state):
        if np.random.rand() < self.epsilon or len(self.memory) < self.REPLAY_START_SIZE:
            return random.randrange(self.action_space)
        q_values = self.ddqn(torch.tensor(np.expand_dims(np.asarray(state).astype(np.float64)
                                , axis=0),device=self.device, dtype=torch.float32))

        return np.argmax(q_values[0].detach().cpu().numpy())

    def remember(self, current_state, action, reward, next_state, terminal):
        current_state, action, reward, next_state = np.asarray(current_state), np.asarray(action), np.asarray(reward), np.asarray(next_state)
        
        # add noise to the reward/state 
        if self.noise == "reward_noise":
            if np.random.rand() < self.noise_level:
                reward = np.random.choice([-1,0,1])
        elif self.noise == "state_noise":
            if np.random.rand() < self.noise_level:
                next_state = np.uint8(next_state+(np.random.randn(*next_state.shape)* 50))
                current_state = np.uint8(current_state+(np.random.randn(*current_state.shape)* 50))

        self.memory.push(torch.unsqueeze(torch.tensor(current_state, dtype=torch.float32),dim=0),
                            torch.unsqueeze(torch.tensor(action, dtype=torch.int64),dim=0),
                            torch.tensor(reward, dtype=torch.float32),
                            torch.unsqueeze(torch.tensor(next_state, dtype=torch.float32),dim=0),
                            torch.tensor(terminal != True, dtype=torch.bool))
        if len(self.memory) > self.memory.capacity:
            self.memory.pop(0)

    def step_update(self, total_step):
        if len(self.memory) < self.REPLAY_START_SIZE:
            return None, None

        loss, average_max_q = None, None
        if total_step % self.TRAINING_FREQUENCY == 0:
            loss, accuracy, average_max_q = self._train()

        self._update_epsilon()

        # if total_step % MODEL_PERSISTENCE_UPDATE_FREQUENCY == 0:
        #     self._save_model()

        if total_step % self.TARGET_NETWORK_UPDATE_FREQUENCY == 0:
            self._reset_target_network()
            print('{{"metric": "epsilon", "value": {}}}'.format(self.epsilon))
            print('{{"metric": "total_step", "value": {}}}'.format(total_step))
        
        return loss, average_max_q

    def _train(self):
        # torch version of training
        batch = self.memory.sample(self.BATCH_SIZE)
        if len(batch) < self.BATCH_SIZE:
            return
        batch = Transition(*zip(*batch))

        non_final_mask = torch.tensor(batch.terminal, device=self.device)

        non_final_next_states = torch.cat([s for s,t in zip(batch.next_state,batch.terminal)
                                                if t]).to(self.device)

        state_batch = torch.cat(batch.current_state).to(self.device)
        action_batch = torch.unsqueeze(torch.cat(batch.action),dim=1).to(self.device)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        state_action_values = self.ddqn(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.ddqn_target(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.ddqn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.loss_record.append(float(loss.detach().cpu().numpy()))
        return float(loss.detach().cpu().numpy()), 0, float(torch.mean(next_state_values).detach().cpu().numpy())

    def _update_epsilon(self):
        if self.noise != "exploration":
            self.epsilon -= self.EXPLORATION_DECAY * self.n_selected
            self.epsilon = max(self.EXPLORATION_MIN, self.epsilon)
        else:
            pass

    def _reset_target_network(self):
        self.ddqn_target.load_state_dict(self.ddqn.state_dict())

    def _save_model(self):
        torch.save(self.ddqn.state_dict(), self.model_path)

    def _load_model(self):
        self.ddqn.load_state_dict(torch.load(self.model_path))
        self.ddqn.eval()
    
    def _assign_model(self, state_dict):
        self.ddqn.load_state_dict(state_dict)
        self.ddqn_target.load_state_dict(state_dict)
    
    def _assign_epsilon(self, epsilon, n_selected):
        if self.noise != "exploration":
            self.n_selected = n_selected
            self.epsilon = epsilon
        else:
            pass

    def get_model(self):
        return self.ddqn

class DDQNSolver(BaseGameModel):

    def __init__(self, action_space, model, device, args):
        BaseGameModel.__init__(self,
                               action_space
                               )
        self.device = device
        self.ddqn = model
        self.epsilon = args['exploration_test']
        self.args = args
        # self._load_model()

    def _assign_model(self, state_dict):
        self.ddqn.load_state_dict(state_dict)
        self.ddqn.eval()

    def _assign_epsilon(self, epsilon):
        if self.args["attack"] != "exploration":
            self.epsilon = epsilon
        else:
            self.epsilon = 0.1

    def move(self, state):
        if np.random.rand() < self.epsilon:
            return random.randrange(self.action_space)
        q_values = self.ddqn(torch.tensor(np.expand_dims(np.asarray(state).astype(np.float64)
                                , axis=0),device=self.device, dtype=torch.float32))

        return np.argmax(q_values[0].detach().cpu().numpy())

    def step_update(self, total_step):
        return None, None

class AtariEnv:
    def __init__(self, game_name):
        self.env = MainGymWrapper.wrap(gym.make(game_name + "Deterministic-v4"))
        self.action_space = self.env.action_space.n
    
    def agent_play(self, agent, total_step_limit, total_run_limit=None, verbose=True):
        run = 0
        total_step = 0

        start_time = time.time()

        scores, steps = [], []
        end_flag = False

        agent.loss_record = []
        while True:
            run_start_time = time.time()

            if total_run_limit is not None and run >= total_run_limit:
                if verbose:
                    print ("Reached total run limit of: " + str(total_run_limit))
                end_flag = True
                break

            run += 1
            current_state = self.env.reset()
            step = 0
            score = 0
            while True:
                if total_step_limit is not None  and total_step >= total_step_limit:
                    if verbose:
                        print ("Reached total step limit of: " + str(total_step_limit))
                    end_flag = True
                    break
                total_step += 1
                step += 1

                action = agent.move(current_state)
                next_state, reward, terminal, info = self.env.step(action)
                
                # clip reward
                reward = np.sign(reward)
                
                score += reward
                agent.remember(current_state, action, reward, next_state, terminal)
                current_state = next_state

                agent.step_update(total_step)

                if terminal:
                    # print("Run: {}".format(run))
                    scores.append(score)
                    steps.append(step)
                    break
            run_end_time = time.time()
            # if isinstance(agent, DDQNTrainer):
                # print("----Time Used: {:.2f} m-----Run Time: {:.2f} m----Estimated Time Left: {:.2f} m".format((run_end_time-start_time)/60
                #                                                 , (run_end_time-run_start_time)/60, (run_end_time-start_time)*total_run_limit/(60*run)-(run_end_time-start_time)/60))
            if end_flag:
                break
        return np.mean(scores), np.mean(steps)
    