from architectures.template import LayerBlock
from torch import nn

class Net13NoMax(nn.Module):
    def __init__(self, numberFeatures):  # Added dropout probability
        super(Net13NoMax, self).__init__()

        self.l1 = LayerBlock(1, numberFeatures, 5, False)  # 13 -> 9
        self.l2 = LayerBlock(numberFeatures, numberFeatures, 3, False)  # 9 -> 7
        self.l3 = LayerBlock(numberFeatures, numberFeatures, 3, False)  # 7 -> 5
        self.l4 = LayerBlock(numberFeatures, numberFeatures, 3, False)  # 5 -> 3
        
        # Flatten before passing to fully connected layers
        self.flatten = nn.Flatten()
    
        # Static ANN Layers with Dropout
        self.fc1 = nn.Linear(numberFeatures*3*3, 64)
        self.dropout1 = nn.Dropout(0.5)  # Dropout after first FC layer
        
        self.fc2 = nn.Linear(64, 1)
        self.dropout2 = nn.Dropout(0.5)  # Dropout before the final layer (optional)
        
        self.sigmoid = nn.Sigmoid()

        self.numberLayers = 6

    def forward(self, x):
        x = self.l1(x)
        print("After l1:", x.shape)
        
        x = self.l2(x)
        print("After l2:", x.shape)
        
        x = self.l3(x)
        print("After l3:", x.shape)
        
        x = self.l4(x)
        print("After l4:", x.shape)
        
        x = self.flatten(x)
        print("Shape after flatten:", x.shape)
        
        x = self.fc1(x)
        print(x.shape)
        
        x = self.dropout1(x)  # Apply dropout after first FC layer
        
        x = self.fc2(x)
        x = self.dropout2(x)  # Apply dropout before final layer (optional)
        
        x = self.sigmoid(x)

        return x
