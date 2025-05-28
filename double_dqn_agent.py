import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_memory import ReplayMemory, Transition
import numpy as np
from abstract_agent import Agent
import random

class DoubleDQNAgent(Agent):
    def __init__(self, gym_env, model_a, model_b, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma,
                 epsilon_i, epsilon_f, epsilon_anneal_steps, episode_block, device, sync_target = 1000):
        
        super().__init__(gym_env, obs_processing_func, memory_buffer_size, batch_size, learning_rate, gamma, epsilon_i, epsilon_f, epsilon_anneal_steps, episode_block, device)
        # Guardar entorno y función de preprocesamiento
        # Inicializar online_net (model_a) y target_net (model_b) en device
        # Configurar función de pérdida MSE y optimizador Adam para online_net
        # Crear replay memory de tamaño buffer_size
        # Almacenar batch_size, gamma, parámetros de epsilon y sync_target
        # Inicializar contador de pasos para sincronizar target
        pass
    
    def select_action(self, state, current_steps, train=True):
      # Calcular epsilon decay según step (entre eps_start y eps_end en eps_steps)
      # Si train y con probabilidad epsilon: acción aleatoria
      # En otro caso: usar greedy_action
      pass
    
    def update_weights(self):
        # 1) Verificar que haya al menos batch_size transiciones en memoria
        # 2) Muestrear minibatch y convertir estados, acciones, recompensas, dones y next_states a tensores
        # 3) Calcular q_current: online_net(states).gather(…)
        # 4) Calcular target Double DQN:
        #    a) best_actions = online_net(next_states).argmax(…)
        #    b) q_next = target_net(next_states).gather(… best_actions)
        #    c) target_q = rewards + gamma * q_next * (1 - dones)
        # 5) Computar loss MSE entre q_current y target_q, backprop y optimizer.step()
        # 6) Decrementar contador y si llega a 0 copiar online_net → target_net
        pass
            