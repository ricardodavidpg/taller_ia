from collections import namedtuple
import torch
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))

# Ejemplo uso
# nueva_tupla = Transition(state, action, reward, done, next_state)

class PrioritisedReplayMemory:

    def __init__(self, capacity, state_processing_function, alpha_p):
        """
        Inicializa la memoria de repetición con capacidad fija.
        Params:
         - capacity (int): número máximo de transiciones a almacenar.
        """
        self.capacity = capacity
        self.memory = []
        self.transition_priorities = []
        self.position = 0
        self.state_processing_function = state_processing_function
        self.device = 'cpu'
        self.alpha_p = alpha_p  # Importancia de la priorización

    def add(self, state, action, reward, done, next_state):
        """
        Agrega una transición a la memoria.
        Si la memoria está llena, sobreescribe la transición más antigua.
        """
        with torch.no_grad(): 
          state_tensor = self.state_processing_function(state, self.device)                                   # Shape: (4,84,84)
          action_tensor = torch.tensor(action, dtype=torch.int64, device=self.device).unsqueeze(0)            # Shape: (1,)  
          reward_tensor = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(0)          # Shape: (1,)
          done_tensor = torch.tensor(float(done), dtype=torch.float32, device=self.device).unsqueeze(0)       # Shape: (1,)
          next_state_tensor = self.state_processing_function(next_state, self.device)                         # Shape: (4,84,84)   
          
          new_transition = Transition(state_tensor, action_tensor, reward_tensor, done_tensor, next_state_tensor)
          if len(self.memory) < self.capacity:
            self.memory.append(new_transition)
            self.transition_priorities.append(max(self.transition_priorities, default=1))
          else:
            self.memory[self.position] = new_transition
            self.transition_priorities[self.position] = max(self.transition_priorities, default=1)
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

      # Sumatoria de las prioridades de transición para calcular la probabilidad de selección
      
      priority_sum = np.sum(np.array(self.transition_priorities) ** self.alpha_p) 

      # Calcular la probabilidad de selección para cada transición
      all_probabilities = np.array([(priority ** self.alpha_p)/priority_sum for priority in self.transition_priorities])

      # Seleccionar índices de transiciones basados en la probabilidad
      selected_index_array = np.random.choice(len(self.memory), batch_size, p=all_probabilities)
      selected_index = torch.tensor(selected_index_array, dtype=torch.int64, device=self.device)     
                
      # Seleccionar las probabilidades de los elementos seleccionados
      selected_probabilities = torch.tensor(all_probabilities[selected_index], dtype=torch.float32, device=self.device)
     
      # Seleccionar las transiciones correspondientes a las mejores probabilidades
      selected_transitions = [self.memory[i] for i in selected_index_array]

      return (selected_transitions, selected_probabilities, selected_index)
      
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
      self.transition_priorities = []
      self.position = 0
