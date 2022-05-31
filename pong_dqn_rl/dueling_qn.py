"""Defines the Neural Network"""
import torchsummary
import torch.nn as nn


class DuelCNN(nn.Module):
    """
    CNN with Duel Algo.
    """

    def __init__(self, h: int, w: int, output_size: int):
        super(DuelCNN, self).__init__()

        # Representation layers
        self.representation = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=24, kernel_size=4, stride=2),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=4, stride=2),
            nn.BatchNorm2d(24),
            nn.Conv2d(in_channels=24, out_channels=8, kernel_size=2, stride=1),
            nn.BatchNorm2d(8),
        )

        # Action layer
        self.Alinear1 = nn.Linear(in_features=16, out_features=128)
        self.Alrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Alinear2 = nn.Linear(in_features=128, out_features=output_size)

        # State Value layer
        self.Vlinear1 = nn.Linear(in_features=16, out_features=128)
        self.Vlrelu = nn.LeakyReLU()  # Linear 1 activation funct
        self.Vlinear2 = nn.Linear(in_features=128, out_features=1)  # Only 1 node

    def forward(self, x) -> float:
        x = self.representation(x)
        x = x.view(x.size(0), -1)  # Flatten every batch

        Ax = self.Alrelu(self.Alinear1(x))
        Ax = self.Alinear2(Ax)  # No activation on last layer

        Vx = self.Vlrelu(self.Vlinear1(x))
        Vx = self.Vlinear2(Vx)  # No activation on last layer

        q = Vx + (Ax - Ax.mean())

        return q

    def show_model_info(self):
        """Displays the parameters and shapes of the network layers"""
        torchsummary.summary(self, (4, 64, 80))
