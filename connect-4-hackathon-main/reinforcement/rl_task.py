# %%
import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# %%

'''
Here, you will design a first Neural Network and get familiar with Pytorch.
We encourage you to take a first look in PyTorch's documentation: 
https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html

Make yourself familiar with:

The Input Layer
Activation Functions
Hidden Layers and
Output Layer

You might want to think how to structure those layers yourself first ;)
'''

class DQN(nn.Module):

    def __init__(self, obs_shape, n_actions):
        super().__init__()
        #Build your own Neural network!
        c, w, h = obs_shape  # channels, height, width
        
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, stride=1),  # out: (32, 4, 5)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1), # out: (64, 2, 3)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1), # out: (64, 1, 2)
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, c, w, h)
            conv_out = self.conv(dummy_input)
            conv_out_size = conv_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )
        
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)  # flatten all but batch dim
        return self.fc(x)


# Utility function
def preprocess_obs(obs):
    # obs: (3, width, height), convert to float32
    return obs.astype(np.float32)