import torch
import torch.nn as nn

from Pretrain.ConvLSTM import ConvLSTM


class SharedNet(nn.Module):
    def __init__(self):
        super(SharedNet, self).__init__()
        self.conv3d_1 = nn.Conv3d(in_channels=2, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu_1 = nn.ReLU()

        self.conv3d_2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu_2 = nn.ReLU()

        self.conv3d_3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=(0, 1, 1))
        self.relu_3 = nn.ReLU()

        self.conv3d_4 = nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu_4 = nn.ReLU()

        self.conv3d_5 = nn.Conv3d(in_channels=32, out_channels=2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.relu_5 = nn.ReLU()
    
    def forward(self, x):
        x = self.relu_1(self.conv3d_1(x))
        x = self.relu_2(self.conv3d_2(x))
        x = self.relu_3(self.conv3d_3(x))
        x = self.relu_4(self.conv3d_4(x))
        x = self.relu_5(self.conv3d_5(x))
        return x


class SharedNet_clstm(nn.Module):
    def __init__(self):
        super(SharedNet_clstm, self).__init__()
        self.clstm1 = ConvLSTM(input_size=(32, 32), input_dim=2, hidden_dim=[8, 16], kernel_size=(3, 3), num_layers=2, batch_first=True, bias=True, return_all_layers=False)
        self.relu1 = nn.ReLU()

        self.clstm2 = ConvLSTM(input_size=(32, 32), input_dim=16, hidden_dim=[32, 64], kernel_size=(3, 3), num_layers=2, batch_first=True, bias=True, return_all_layers=False)
        self.relu2 = nn.ReLU()

        self.clstm3 = ConvLSTM(input_size=(32, 32), input_dim=64, hidden_dim=[16, 2], kernel_size=(3, 3), num_layers=2, batch_first=True, bias=True, return_all_layers=False)
        self.relu3 = nn.ReLU()
    
    def forward(self, x):
        x, _ = self.clstm1(x)
        x = self.relu1(x[0])
        x, _ = self.clstm2(x)
        x = self.relu2(x[0])
        x, _ = self.clstm3(x)
        x = self.relu3(x[0])
        return x
