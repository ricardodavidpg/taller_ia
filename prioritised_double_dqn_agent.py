import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prioritised_replay_memory import PrioritisedReplayMemory, Transition
import numpy as np
from abstract_agent import Agent
import random
import gc

class PrioritisedDoubleDQNAgent(Agent):
    def __init__(self, gym_env, model_a, model_b, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_steps, episode_block, sync_target, alpha_p, beta_p_init, beta_p_max, beta_p_anneal_steps, epsilon_p, device):
        
        super().__init__(gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma, epsilon_i, epsilon_f, epsilon_anneal_steps, episode_block, device)
        # Inicializar online_net (model_a) y target_net (model_b) en device
        self.online_net = model_a.to(device)
        self.target_net = model_b.to(device)
        
        # Configurar función de pérdida MSE y optimizador Adam
        self.loss = nn.MSELoss(reduction='none')  
        self.optim = optim.Adam(self.online_net.parameters(), lr=learning_rate)

        # Almacenar sync_target
        self.sync_target = sync_target

        # Inicializar contador de pasos para sincronizar target
        self.sync_counter = sync_target

        self.alpha_p = alpha_p
        self.beta_p_init = beta_p_init
        self.beta_p = beta_p_init
        self.beta_p_max = beta_p_max
        self.beta_p_anneal_steps = beta_p_anneal_steps
        
        self.epsilon_p = epsilon_p

        # Se sustituye la memoria por PriorisedReplayMemory
        self.memory = PrioritisedReplayMemory(memory_buffer_size, obs_processing_func, alpha_p, epsilon_p)
      
    def select_action(self, state, current_steps, train=True):
      epsilon = self.compute_epsilon(current_steps)
      if (train and random.random() < epsilon):
        action = self.env.action_space.sample()
      else:
        phi_state = self.state_processing_function(state).unsqueeze(0)
        q = self.online_net(phi_state)
        action = torch.argmax(q, dim=1).item()
      return action
            
    def update_weights(self):
      # 1) Comprobar que hay al menos batch_size muestras en memoria
      if len(self.memory) >= self.batch_size:
          self.optim.zero_grad()

          # 2) Muestrear minibatch y convertir a tensores (states, actions, rewards, dones, next_states
          trasitions, transitions_probabilities, transitions_index = self.memory.sample(self.batch_size)
          satate_list, actions_list, reward_list, dones_list, next_state_list = zip(*trasitions)
          state_batch = torch.stack(satate_list)                                                           # Shape: (batch_size, 4, 84, 84)
          actions_batch = torch.stack(actions_list)                                                        # Shape: (batch_size, 1)
          rewards_batch = torch.stack(reward_list)                                                          # Shape: (batch_size, 1)
          dones_batch = torch.stack(dones_list)                                                            # Shape: (batch_size, 1)
          next_state_batch = torch.stack(next_state_list)                                                  # Shape: (batch_size, 4, 84, 84)  
          transitions_probabilities = transitions_probabilities                                           # Shape: (batch_size) 
          transitions_index = transitions_index                                                            # Shape: (batch_size)

          # 3) Calcular q_state_batch = online_net(state_batch)
          q_state_batch = self.online_net(state_batch)                                                                      # Shape: (batch_size, 4)                                                                                                          
          q_current_batch = q_state_batch.gather(dim=1, index=actions_batch)                                                # Shape: (batch_size, 1)
          
          # 4) Calcular target Double DQN:
          with torch.no_grad():
          #    a) best_actions = online_net(next_states).argmax(…)
              best_actions = self.online_net(next_state_batch).max(dim=1)[1].unsqueeze(1)                                   # Shape: (batch_size, 1) 
          #    b) q_next = target_net(next_states).gather(… best_actions)                
              q_next = self.target_net(next_state_batch).gather(dim=1, index = best_actions)                                # Shape: (batch_size, 1)
          #    c) target_q = rewards + gamma * q_next * (1 - dones)
              q_target_batch = rewards_batch + self.gamma * (q_next * (1 - dones_batch))       
              
          transitions_weights = (len(self.memory) * transitions_probabilities).pow(-self.beta_p).unsqueeze(1)               # Shape: (batch_size, 1)
          normalized_weights = (transitions_weights / transitions_weights.max())                                            # Shape: (batch_size, 1)
          td_error = q_target_batch - q_current_batch
          self.memory.update_priorities(transitions_index, torch.abs(td_error.squeeze(1)).detach()) 
          self.update_beta()

          # 5) Computar loss MSE entre q_current y target, backprop y optimizer.step()    
          loss = self.loss(q_current_batch, q_target_batch) 
          weighted_loss = (normalized_weights * loss).mean()                                                 
          weighted_loss.backward()
          self.optim.step()
          
          # 6) Decrementar contador y si llega a 0 copiar online_net → target_net
          self.sync_counter -= 1
          if(self.sync_counter == 0):
             self.target_net.load_state_dict(self.online_net.state_dict())
             self.sync_counter = self.sync_target
          
          del loss, weighted_loss, td_error  # o usá names más específicos
            
    def saveModel(self):
      torch.save(self.online_net.state_dict(), "PrioritisedDDQNAgent.dat") 

    def update_beta(self): 
      frac = min(1.0, self.total_steps / self.beta_p_anneal_steps) 
      self.beta_p = self.beta_p_init + (self.beta_p_max - self.beta_p_init) * frac

      