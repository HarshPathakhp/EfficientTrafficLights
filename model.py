import torch
import torch.nn as nn

class DuelCNN(nn.Module):
    def __init__(self, num_actions = 9, img_size = 100, in_channels = 2, num_blks = 3):
        super(DuelCNN, self).__init__()
        conv_list = []
        self.num_actions = num_actions
        self.in_channels = in_channels
        channel_seq = [32, 64, 128]
        begin_channel = self.in_channels
        for i in range(num_blks):
            conv_list.append(nn.Conv2d(begin_channel, channel_seq[i], kernel_size = 3, stride = 2))
            if(i != 0):
                conv_list.append(nn.BatchNorm2d(channel_seq[i]))
            conv_list.append(nn.LeakyReLU())
            begin_channel = channel_seq[i]
        self.conv = nn.Sequential(*conv_list)
        
        self.to_flatten = 128 * 11 * 11
        flatten_list = [nn.Linear(self.to_flatten, 256),nn.LeakyReLU()]
        self.flatten = nn.Sequential(*flatten_list)

        self.value_head = nn.Sequential(*[nn.Linear(256, 64), nn.LeakyReLU(), nn.Linear(64, 1), nn.LeakyReLU()])
        self.advantage_head = nn.Sequential(*[nn.Linear(256, 64), nn.LeakyReLU(), nn.Linear(64, self.num_actions), nn.LeakyReLU()])

    def forward(self, img):
        x = self.conv(img)
        x = x.view(img.shape[0], -1)
        flat = self.flatten(x)
        value = self.value_head(flat)
        adv = self.advantage_head(flat)
        if(img.shape[0] != 1):
            qvalues = value + (adv - torch.mean(adv, dim = 1, keepdim = True).repeat(1, self.num_actions))
        else:
            qvalues = value + (adv - torch.mean(adv, dim = 1))
        return qvalues



