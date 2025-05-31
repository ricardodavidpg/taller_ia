import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from abstract_agent import Agent
from dqn_cnn_model import DQN_CNN_Model
from replay_memory import ReplayMemory
import random

class DQNAgent(Agent):
    def __init__(self, env, model, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma, epsilon_i, epsilon_f, epsilon_anneal_steps, episode_block, device):
        super().__init__(env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma, epsilon_i, epsilon_f, epsilon_anneal_steps, episode_block, device)
        # Inicializar policy_net en device
        self.policy_net = model.to(device)
        
        # Configurar función de pérdida MSE y optimizador Adam
        self.loss = nn.MSELoss()
        self.optim = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
    def select_action(self, state, current_steps, train=True):
      epsilon = self.compute_epsilon(current_steps)
      if (train and random.random() < epsilon):
        action = self.env.action_space.sample()
      else:
        phi_state = self.state_processing_function(state).unsqueeze(0)
        q = self.policy_net(phi_state)
        action = torch.argmax(q, dim=1).item()
      return action

    def update_weights(self):
      # 1) Comprobar que hay al menos batch_size muestras en memoria
      if len(self.memory) >= self.batch_size:
          self.optim.zero_grad()

          # 2) Muestrear minibatch y convertir a tensores (states, actions, rewards, dones, next_states
          satate_list, actions_list, reward_list, dones_list, next_state_list = zip(*self.memory.sample(self.batch_size))
          state_batch = torch.stack(satate_list).to(self.device)                                                            # Shape: (batch_size, 4, 84, 84)
          actions_batch = torch.stack(actions_list).to(self.device)                                                         # Shape: (batch_size, 1)
          rewards_batch = torch.stack(reward_list).to(self.device)                                                          # Shape: (batch_size, 1)
          dones_batch = torch.stack(dones_list).to(self.device)                                                             # Shape: (batch_size, 1)
          next_state_batch = torch.stack(next_state_list).to(self.device)                                                   # Shape: (batch_size, 4, 84, 84)  
          
          # 3) Calcular q_state_batch = policy_net(state_batch)
          q_state_batch = self.policy_net(state_batch)                                                                      # Shape: (batch_size, 4)                                                                                                          
          q_current_batch = q_state_batch.gather(dim=1, index=actions_batch)                                                # Shape: (batch_size, 1)
          
          # 4) Con torch.no_grad(): calcular q_max_next_state = policy_net(next_states).max(dim=1)[0] * (1 - dones)
          with torch.no_grad():
              q_next_state_batch = self.policy_net(next_state_batch).max(dim=1)[0].unsqueeze(1)                             # Shape: (batch_size, 1) 
              # 5) Calcular target = rewards + gamma * max_q_next_state                 
              q_target_batch = rewards_batch + self.gamma * (q_next_state_batch * (1 - dones_batch))                        # Shape: (batch_size, 1)
              
          # 6) Computar loss MSE entre q_current y target, backprop y optimizer.step()
          loss = self.loss(q_current_batch, q_target_batch)                                                           
          loss.backward()
          self.optim.step()
