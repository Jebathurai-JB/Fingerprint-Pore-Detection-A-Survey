from architectures.template import LayerBlock
from torch import nn

class Net13Max(nn.Module):
    def __init__(self, numberFeatures):
        super(Net13Max, self).__init__()
        self.l1 = LayerBlock(1, numberFeatures, 5, True)    # 13 -> 9
        #self.l2 = LayerBlock(numberFeatures, numberFeatures, 3, True)   # 9 -> 5
        self.l2 = nn.Conv2d(numberFeatures, 1, kernel_size=3, padding="valid", stride=(1, 1))
        self.l3 = nn.MaxPool2d(kernel_size=3, stride=(1, 1))
        self.sigmoid = nn.Sigmoid() 

        self.numberLayers = 4

    def forward(self, x):
        x = self.l1(x)
        #x = self.l2(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.sigmoid(x)        
        return x


