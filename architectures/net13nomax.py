import torch
from architectures.template import LayerBlock
from torch import nn

class Net13NoMax(nn.Module):
    def __init__(self, numberFeatures 6):  
        super(Net13NoMax, self).__init__()

        self.l1 = LayerBlock(1, numberFeatures, 5, False)  # 13 -> 9
        self.l2 = LayerBlock(numberFeatures, numberFeatures, 3, False)  # 9 -> 7
        self.l3 = LayerBlock(numberFeatures, numberFeatures, 3, False)  # 7 -> 5
        self.l4 = LayerBlock(numberFeatures, numberFeatures, 3, False)  # 5 -> 3

        self.flatten = nn.Flatten()

        # 🔹 Dynamically calculate input features for fc1
        self._initialize_fc1_input(numberFeatures)

        self.fc1 = nn.Linear(self.fc1_input_size, 64)  # Correct input size
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(64, 1)
        self.dropout2 = nn.Dropout(0.5)

        self.sigmoid = nn.Sigmoid()

    def _initialize_fc1_input(self, numberFeatures):
        """Helper function to dynamically compute fc1 input size"""
        with torch.no_grad():
            dummy_input = torch.randn(1, 1, 13, 13)  # Simulate batch size 1
            dummy_output = self.l1(dummy_input)
            dummy_output = self.l2(dummy_output)
            dummy_output = self.l3(dummy_output)
            dummy_output = self.l4(dummy_output)
            self.fc1_input_size = dummy_output.numel()  # Flattened size

    def forward(self, x):
        print("Received input shape:", x.shape)  # Debugging input shape

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        print("Shape before flatten:", x.shape)
        x = self.flatten(x)
        print("Shape after flatten:", x.shape)

        x = self.fc1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.dropout2(x)

        x = self.sigmoid(x)

        return x
