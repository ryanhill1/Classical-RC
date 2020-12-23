from pathlib import Path
import numpy as np
import scipy.io
from scipy import sparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Ryan Hill (rjh324@cornell.edu)
# ESN Implementation

class Reservoir(object):
    
    # [new_state] generates random input weights, normally distributed around zero. 
    def new_state(self, input_size, hidden_size): 
        return np.random.rand(hidden_size, input_size) - 0.5
    
    # [init_weight_matrix] generates the internal weight matrix, [W]. 
    # [W] is generated sparse with nonzero elements having a normal distribution 
    # centered around zero.
    def init_weight_matrix(self, hidden_size, sparsity, spectral_radius):
        W = sparse.rand(hidden_size, hidden_size, density=1-sparsity).todense() 
        W[np.where(W > 0)] -= 0.5 # center around 0
        eigenvalues = np.linalg.eig(W)[0]
        max_eig = max(abs(eigenvalues))
        W *= 1 / max_eig
        W *= spectral_radius # set max eigenvalue as spectral radius
        return W

    """
    Parameters:
        input_size = input dimension
        hidden_size = number of internal (hidden) units in the reservoir
        sparsity = percentage of elements in W_in equal to zero
        spectral_radius = maximal absolute eigenvalue of matrix (W)*
        a = input scaling (of W_in) (how "nonlinear" reservoir responses are)*
        leak = leaking rate (speed of reservoir update dynamics over time)*
    """
    def __init__(self, input_size, hidden_size=30, sparsity=0.7,
        spectral_radius=0.9, a=1.0, leak=0.2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        self.a = a
        self.leak = leak
        # reservoir input weights
        self.W_in = Reservoir.new_state(input_size, hidden_size)
        # reservoir matrix weights
        self.W = Reservoir.init_weight_matrix(hidden_size, sparsity, spectral_radius) 
        # vector of reservoir neuron activations
        self.state_x = np.zeros((hidden_size, 1)) 

    # [forward_res] updates the state representation [x] for given input [u]. 
    def forward_res(self, u):
        state_x = ((1-self.leak) * self.state_x + 
        self.leak * np.tanh(np.dot(self.W_in, u) + np.dot(self.W, self.state_x)))
        return state_x 
    
class ESN(nn.Module): 

  def __init__(self, input_size, output_size):
    super().__init__()
    self.reservoir = Reservoir(input_size)
    self.fc = nn.Linear(in_features=input_size, 
    out_features=output_size)  # linear layer
    # normalize resultant vector into probability distribution
    self.softmax = nn.Softmax(dim=-1) 

  # [forward_esn] calculates the high dimensional state representation H for an
  # input of shape [n_obs, t_step, n_var] where [n_obs] is the number of observations, 
  # [t_step] is the number of time steps and [n_var] is the number of variables.
  def forward_esn(self, inp):
    n_obs, t_step, n_var = inp.shape()
    for i in range(n_obs):
      u = inp[i]
      x_r = self.reservoir.forward_res(u)
      x_r = np.concatenate((inp, x_r)) # ValueError:
      # all the input arrays must have the same number of dimensions
    H = self.fc(x_r)
    H_soft = self.softmax(H)
    return H_soft

esn = ESN(1,1) # create an instance

# Training
training_data = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST'
    ,train=True
    ,download=True
    ,transform=transforms.Compose([
        transforms.ToTensor()
    ])
)

data_loader = torch.utils.data.DataLoader(
    training_data
    ,batch_size=10
)

torch.set_grad_enabled(False)

# Define a loss function and an optimizer
# criterion = nn.MSELoss(reduction='sum')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(esn.parameters(), lr=1e-5)
# optimizer = optim.SGD(network.parameters(), lr=1e-5, momentum=0.9)

# Train the network
esn = esn.float()
start = time.time()
print("starting timer")

# initialize at epoch 1
def train(epoch):
  running_loss = 0.0
  for i in range(len(training_data)):
    # get the inputs; data is a list of [inputs, labels]
    data_file = training_data[i]
    inputs, labels = data_file

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    inputs = inputs.unsqueeze(0)
    outputs = esn(inputs.float())
    # loss = criterion(outputs, labels.float()) # uncomment for MSE
    loss = criterion(outputs, labels)
    # print(loss)
    loss.backward()
    optimizer.step()

    # print statistics
    running_loss += loss.item()
    if i % 100 == 99:  # print every 100 mini-batches (1 epoch)
      print('epoch %d, loss: %.3f' %
        (epoch, running_loss / 100))
      # adam_loss.append(running_loss / 100)
      epoch += 1
      if running_loss <= (0.05 * 100):
        print('Finished Training')
        return
    train(epoch)

train(1)
end = time.time()
print("total time = ", end - start)  # time in seconds





    



  





