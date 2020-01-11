import torch
import torch.nn as nn
import torch.nn.functional as F
from activations  import Mish

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        # Depthwise Convolutions
        self.layers = nn.Sequential(
                            nn.Conv2d(in_channels=1*3, out_channels=16*3, kernel_size=3, groups=1, stride=1, padding=1, bias=False),
                            Mish(),
                            nn.BatchNorm2d(num_features=16*3, eps=1e-3, momentum=0.99),

                            nn.Conv2d(in_channels=16*3, out_channels=96, kernel_size=1, groups=8, stride=1, padding=0, bias=False),
                            Mish(),
                            nn.BatchNorm2d(num_features=96, eps=1e-3, momentum=0.99),

                            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, groups=8, stride=1, padding=1, bias=False),
                            Mish(),
                            nn.BatchNorm2d(num_features=128, eps=1e-3, momentum=0.99),
                            nn.MaxPool2d(2, 2),

                            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=1, groups=16, stride=1, padding=0, bias=False),
                            Mish(),
                            nn.BatchNorm2d(num_features=192, eps=1e-3, momentum=0.99),

                            nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, groups=16, stride=1, padding=1, bias=False),
                            Mish(),
                            nn.BatchNorm2d(num_features=256, eps=1e-3, momentum=0.99),

                            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, groups=32, stride=1, padding=0, bias=False),
                            Mish(),
                            nn.BatchNorm2d(num_features=512, eps=1e-3, momentum=0.99),
                            nn.MaxPool2d(2, 2),

                            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, groups=64, stride=1, padding=1, bias=False),
                            Mish(),
                            nn.BatchNorm2d(num_features=512, eps=1e-3, momentum=0.99),

                            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, groups=16, stride=1, padding=0, bias=False),
                            Mish(),
                            nn.BatchNorm2d(num_features=256, eps=1e-3, momentum=0.99),

                            nn.Conv2d(in_channels=256, out_channels=192, kernel_size=3, groups=16, stride=1, padding=1, bias=False),
                            Mish(),
                            nn.BatchNorm2d(num_features=192, eps=1e-3, momentum=0.99),
                            nn.MaxPool2d(2, 2),
                            )


        #squeeze and excitation
        self.se_reduce = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1)
        self.se_expand = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=1)

        # fully connected layer
        self.fc = nn.Linear(in_features=192*4*4, out_features=10)


    def forward(self, x):
        x = self.layers(x)
        x_squeezed = F.adaptive_avg_pool2d(x, x.size(2))
        x_squeezed = self.se_expand(Mish()(self.se_reduce(x_squeezed)))
        x = torch.sigmoid(x_squeezed) * x
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
