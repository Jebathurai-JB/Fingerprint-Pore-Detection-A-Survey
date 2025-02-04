from architectures.template import LayerBlock
from torch import nn

class Net13NoMax(nn.Module):
    def __init__(self, numberFeatures):
        super(Net13NoMax, self).__init__()

        self.l1 = LayerBlock(1, numberFeatures, 5, False)
        self.l2 = LayerBlock(numberFeatures, numberFeatures, 3, False)
        self.l3 = LayerBlock(numberFeatures, numberFeatures, 3, False)
        self.l4 = LayerBlock(numberFeatures, numberFeatures, 3, False)

        self.flatten = nn.Flatten()

        # Placeholder, will be updated dynamically
        self.fc1 = None
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.numberLayers = 6

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        x = self.flatten(x)
        
        # *Dynamically set fc1 input size*
        if self.fc1 is None:
            input_size = x.shape[1]  # Dynamically get input features
            self.fc1 = nn.Linear(input_size, 64).to("cuda")  # Move to correct device

        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.sigmoid(x)

        return x
