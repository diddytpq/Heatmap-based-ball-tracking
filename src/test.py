from efficientnet_pytorch import EfficientNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch import optim

import time
import numpy as np

class EfficientNet_MultiLabel(nn.Module):
    def __init__(self, in_channels):
        super(EfficientNet_MultiLabel, self).__init__()
        self.network = EfficientNet.from_pretrained('efficientnet-b3', in_channels=in_channels)
        self.output_layer = nn.Linear(1000, 26)

    def forward(self, x):
        x = F.relu(self.network(x))
        x = torch.sigmoid(self.output_layer(x))
        return x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('GPU Use : ',torch.cuda.is_available())

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.randn(1, 9, 288, 512).to(device)
    model = EfficientNet_MultiLabel(in_channels=9).to(device)

    time_list =[]

    epoch = 50

    summary(model, (9, 288, 512),device='cuda')
    
    with torch.no_grad():
        for i in range(epoch):
            torch.cuda.synchronize()
            t0 = time.time()
            output = model(data)
            torch.cuda.synchronize()
            t1 = time.time()

            #print(i)

            time_list.append(t1-t0)
            #print('output size:', output.size())
    print("output_shape : ",output.size())
    print("avg time : ", np.mean(time_list))
    print("avg FPS : ", 1 / np.mean(time_list))
