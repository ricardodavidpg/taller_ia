import random
from collections import namedtuple
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))

# Ejemplo uso
# nueva_tupla = Transition(state, action, reward, done, next_state)

class ReplayMemory:

    def __init__(self, capacity, state_processing_function):
        """
        Inicializa la memoria de repetición con capacidad fija.
        Params:
         - capacity (int): número máximo de transiciones a almacenar.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.state_processing_function = state_processing_function
        self.device = 'cpu'

    def add(self, state, action, reward, done, next_state):
        """
        Agrega una transición a la memoria.
        Si la memoria está llena, sobreescribe la transición más antigua.
        """
        
        with torch.no_grad(): 
          state_tensor = self.state_processing_function(state, self.add_device)                                   # Shape: (4,84,84)
          action_tensor = torch.tensor(action, dtype=torch.int64, device=self.add_device).unsqueeze(0)            # Shape: (1,)  
          reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.add_device).unsqueeze(0)          # Shape: (1,)
          done_tensor = torch.tensor(float(done), dtype=torch.float32, device=self.add_device).unsqueeze(0)       # Shape: (1,)
          next_state_tensor = self.state_processing_function(next_state, self.add_device)                         # Shape: (4,84,84)   
          
          new_transition = Transition(state_tensor, action_tensor, reward_tensor, done_tensor, next_state_tensor)
          if len(self.memory) < self.capacity:
            self.memory.append(new_transition)
          else:
            self.memory[self.position] = new_transition
          self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
      """
      Devuelve un batch aleatorio de transiciones.
      Params:
       - batch_size (int): número de transiciones a muestrear.
      Returns:
       - lista de Transition de longitud batch_size.
      """
      assert batch_size <= len(self), "El tamaño del batch debe ser menor o igual que la cantidad de elementos en la memoria."
      return random.sample(self.memory, batch_size)
      
    def __len__(self):
      """
      Devuelve el número actual de transiciones en memoria.
      """
      return len(self.memory)
    
    def clear(self):
      """
      Elimina todas las transiciones de la memoria.
      """
      self.memory = []
      self.position = 0
