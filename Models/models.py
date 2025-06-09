import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # Use the last time step's output
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class MATCH(nn.Module):
    def __init__(self, d_long, n_base, out_len):
        super().__init__()
        self.long1 = conv_block(d_long, 32, kernel_size=3, padding=1)
        self.mask1 = conv_block(d_long, 8, kernel_size=3, padding=1)
        
        self.long2 = conv_block(40, 32, kernel_size=3, padding=1)
        self.mask2 = conv_block(8, 8, kernel_size=3, padding=1)
        
        self.long3 = conv_block(40, 32, kernel_size=3, padding=1)
        self.mask3 = conv_block(8, 8, kernel_size=3, padding=1)
        
        self.long4 = conv_block(40, 32, kernel_size=3)
        
        self.survival = nn.Sequential(
            nn.Linear(32 + n_base, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(64),
            nn.Linear(64,out_len)
        )

        self.long = nn.Linear(32+n_base, d_long)
            
    
    def forward(self, long, base, mask):
        x = self.long1(long)
        mask = self.mask1(mask)
        x = torch.cat((x,mask),dim=1)
        
        x = self.long2(x)
        mask = self.mask2(mask)
        x = torch.cat((x,mask),dim=1)
        
        x = self.long3(x)
        mask = self.mask3(mask)
        x = torch.cat((x,mask),dim=1)
        
        x = self.long4(x)
        
        x = F.adaptive_avg_pool1d(x,1).squeeze()

        x = torch.cat((x,base),dim=1)
        surv = self.survival(x)
        surv = torch.softmax(surv, dim=1)
        return surv
        

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, padding_mode='replicate', **kwargs),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(out_channels)
            )
    def forward(self, x):
            return self.convolution(x)