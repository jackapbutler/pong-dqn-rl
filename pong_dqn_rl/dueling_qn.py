"""Defines the Neural Network"""
import torchsummary
import torch.nn as nn


class DuelCNN(nn.Module):
    """
    CNN with Duel Algo.
    """

    def __init__(self, output_size: int):
        super(DuelCNN, self).__init__()

        # Representation layers
        self.representation = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=10, kernel_size=8, stride=4),
            nn.Conv2d(in_channels=10, out_channels=2, kernel_size=4, stride=2),
        )

        # Action layer
        self.action_layer = nn.Sequential(
            nn.Linear(in_features=96, out_features=50),
            nn.LeakyReLU(),
            nn.Linear(in_features=50, out_features=output_size),
        )

        # State Value layer
        self.value_layer = nn.Sequential(
            nn.Linear(in_features=96, out_features=50),
            nn.LeakyReLU(),
            nn.Linear(in_features=50, out_features=output_size),
        )

    def forward(self, x) -> float:
        x = self.representation(x)
        x = x.view(x.size(0), -1)  # Flatten every batch

        Ax = self.action_layer(x)
        Vx = self.value_layer(x)

        q = Vx + (Ax - Ax.mean())

        return q

    def show_model_info(self):
        """Displays the parameters and shapes of the network layers"""
        torchsummary.summary(self, (4, 64, 80))
