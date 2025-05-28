import torch.nn as nn
import torch.nn.functional as F


def conv2d_output_shape(
    input_size: tuple[int, int],
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
    dilation: int | tuple[int, int] = 1,
) -> tuple[int, int]:
    """
    Calcula (H_out, W_out) para una capa Conv2d con:
      - input_size: (H_in, W_in)
      - kernel_size, stride, padding, dilation: int o tupla (altura, ancho)
    Basado en:
      H_out = floor((H_in + 2*pad_h - dil_h*(ker_h−1) - 1) / str_h + 1)
      W_out = floor((W_in + 2*pad_w - dil_w*(ker_w−1) - 1) / str_w + 1)
    Fuente: Shape section en torch.nn.Conv2d :contentReference[oaicite:0]{index=0}
    """
    # Unifica todos los parámetros a tuplas (h, w)
    def to_tuple(x):
        return (x, x) if isinstance(x, int) else x

    H_in, W_in = input_size
    ker_h, ker_w = to_tuple(kernel_size)
    str_h, str_w = to_tuple(stride)
    pad_h, pad_w = to_tuple(padding)
    dil_h, dil_w = to_tuple(dilation)

    H_out = (H_in + 2*pad_h - dil_h*(ker_h - 1) - 1) // str_h + 1
    W_out = (W_in + 2*pad_w - dil_w*(ker_w - 1) - 1) // str_w + 1

    return H_out, W_out


class DQN_CNN_Model(nn.Module):
    def __init__(self,  obs_shape, n_actions):
        super(DQN_CNN_Model, self).__init__()
        
        in_chanels, _, _ = obs_shape #(canales_entrada, alto_entrada, ancho_entrada)

        self.conv1 = nn.Conv2d(in_chanels, 16, kernel_size=8, stride=4) #Convolucional1 - 16 filtros, tamaño 8x8, stride 4
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)         #Convolucional2 - 32 filtros, tamaño 4x4, stride 2
        
        self.fc1 = nn.Linear(32 * 9 * 9, 256)                           #Capa lineal1 - 32*9*9 entradas, 256 salidas
        self.output = nn.Linear(256, n_actions)                         #Capa de salida - 256 entradas, n_actions salidas

    def forward(self, obs):
        #Input shape: (batch_size, 4, 84, 84)                                          
        result = self.conv1(obs)                        #The first hidden layer convolves 16 8 × 8 filters with stride 4 
        #Conv1 output shape: (batch_size, 16, 20, 20)                 
        result = F.relu(result)                         #and applies a rectifier nonlinearity
        #ReLU1 output shape: (batch_size, 16, 20, 20)               
        result = self.conv2(result)                     #The second hidden layer convolves 32 4 × 4 filters with stride 2               
         #Conv2 output shape: (batch_size, 32, 9, 9)          
        result = F.relu(result)                         #again followed by a rectifier nonlinearity   
        #ReLU2 output shape: (batch_size, 32, 9, 9)            
        result = result.view(result.size(0), -1)
        #Flatten output shape: (batch_size, 32*9*9) 
        result = F.relu(self.fc1(result))               #The final hidden layer is fully-connected and consists of 256 rectifier units
        #FC1 output shape: (batch_size, 256)        
        result = self.output(result)                    #The output layer is a fullyconnected linear layer with a single output for each valid action    
        #Output shape: (batch_size, n_actions)
        return result                     