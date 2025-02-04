from architectures.template import LayerBlock
from torch import nn
import torch

class Net13NoMax(nn.Module):
    def __init__(self, numberFeatures):
        super(Net13NoMax, self).__init__()

        # Set device to GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.l1 = LayerBlock(1, numberFeatures, 5, False).to(self.device)
        self.l2 = LayerBlock(numberFeatures, numberFeatures, 3, False).to(self.device)
        self.l3 = LayerBlock(numberFeatures, numberFeatures, 3, False).to(self.device)
        self.l4 = LayerBlock(numberFeatures, numberFeatures, 3, False).to(self.device)

        self.flatten = nn.Flatten().to(self.device)

        # Define fc1 with dummy values, will be updated in forward()
        self.fc1 = nn.Linear(1, 64).to(self.device)  # Placeholder (will be replaced)
        self.fc1_initialized = False  # Track initialization

        self.dropout1 = nn.Dropout(0.5).to(self.device)
        self.fc2 = nn.Linear(64, 1).to(self.device)
        self.dropout2 = nn.Dropout(0.5).to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)

        self.numberLayers = 6

    def forward(self, x):
        x = x.to(self.device)  # Ensure input is on the same device

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        x = self.flatten(x)

        # **Update fc1 input size only once**
        if not self.fc1_initialized:
            input_size = x.shape[1]  # Get flattened feature size
            self.fc1 = nn.Linear(input_size, 64).to(self.device)  # Update layer
            self.fc1_initialized = True  # Mark as initialized

        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.sigmoid(x)

        return x
    
