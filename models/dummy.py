import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(
            nn.Conv2d(  1, 16, 3, 2), nn.ReLU(),
            nn.Conv2d( 16, 32, 3, 2), nn.ReLU(),
            nn.Conv2d( 32, 64, 3, 2), nn.ReLU(),
        )   
        self.classifier = nn.Sequential(
            nn.Linear( 256, 100), nn.ReLU(),
            nn.Linear( 100,  10)
        )

    def forward(self, x):
        x = self.base(x)
        output = self.classifier(torch.flatten(x, 1))
        return output