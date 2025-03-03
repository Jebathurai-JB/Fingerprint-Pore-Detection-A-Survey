from torch import nn

class Net13NoMax(nn.Module):
    def __init__(self, numberFeatures):
        super(Net13NoMax, self).__init__()

        self.l1 = nn.Sequential(
            nn.Conv2d(1, numberFeatures, kernel_size=5, padding=1, stride=1),  # Only this layer has padding
            nn.BatchNorm2d(numberFeatures),
            nn.ReLU()
        )
        
        self.l2 = nn.Sequential(
            nn.Conv2d(numberFeatures, numberFeatures, kernel_size=3, stride=1),
            nn.BatchNorm2d(numberFeatures),
            nn.ReLU()
        )
        
        self.l3 = nn.Sequential(
            nn.Conv2d(numberFeatures, numberFeatures, kernel_size=3, stride=1),
            nn.BatchNorm2d(numberFeatures),
            nn.ReLU()
        )

        self.l4 = nn.Sequential(
            nn.Conv2d(numberFeatures, numberFeatures, kernel_size=3, stride=1),
            nn.BatchNorm2d(numberFeatures),
            nn.ReLU()
        )
        
        self.l5 = nn.Sequential(
            nn.Conv2d(numberFeatures, numberFeatures, kernel_size=3, stride=1),
            nn.BatchNorm2d(numberFeatures),
            nn.ReLU()
        )
        
        self.l6 = nn.Conv2d(numberFeatures, 1, kernel_size=3, stride=1)

        self.sigmoid = nn.Sigmoid()

        self.numberLayers = 5  # Updated based on the modifications

    

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.sigmoid(x)
        return x
