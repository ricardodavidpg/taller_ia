import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))

# Ejemplo uso
# nueva_tupla = Transition(state, action, reward, done, next_state)

class ReplayMemory:

    def __init__(self, capacity):
        """
        Inicializa la memoria de repetición con capacidad fija.
        Params:
         - capacity (int): número máximo de transiciones a almacenar.
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0        

    def add(self, state, action, reward, done, next_state):
        """
        Agrega una transición a la memoria.
        Si la memoria está llena, sobreescribe la transición más antigua.
        """
        #TODO: Joaquin dijo que nos recomendaba pasar state y next_state por pa función fi que vamos a hacer más adelante
        new_transition = Transition(state, action, reward, done, next_state)
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
      if batch_size > self.__len__():
          raise ValueError("El tamaño del batch no puede ser mayor que el número de transiciones en memoria.")
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
