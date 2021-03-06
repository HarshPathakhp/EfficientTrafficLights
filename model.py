import torch
import torch.nn as nn

class DuelCNN(nn.Module):
    def __init__(self, num_actions = 9, img_size = 100, in_channels = 2, num_blks = 4, num_phases = 4):
        super(DuelCNN, self).__init__()
        conv_list = []
        self.num_actions = num_actions
        self.in_channels = in_channels
        channel_seq = [32, 64, 128, 128]
        begin_channel = self.in_channels
        for i in range(num_blks):
            conv_list.append(nn.Conv2d(begin_channel, channel_seq[i], kernel_size = 3, stride = 2))
            conv_list.append(nn.LeakyReLU())
            begin_channel = channel_seq[i]
        self.conv = nn.Sequential(*conv_list)
        
        self.to_flatten = 128 * 5 * 5
        flatten_list = [nn.Linear(self.to_flatten, 256),nn.LeakyReLU()]
        self.flatten = nn.Sequential(*flatten_list)

        dense = [nn.Linear(num_phases, 16), nn.LeakyReLU(),
                nn.Linear(16,32), nn.LeakyReLU()]
        self.phase_dense = nn.Sequential(*dense)
        self.final_dense = nn.Sequential(*[nn.Linear(288, 128), nn.LeakyReLU(),
                                        nn.Linear(128,128), nn.LeakyReLU(),
                                        nn.Linear(128, 128), nn.LeakyReLU()])
        self.value_head = nn.Sequential(*[nn.Linear(128, 64), nn.LeakyReLU(),
                                         nn.Linear(64, 64), nn.LeakyReLU(),
                                         nn.Linear(64, 1), nn.LeakyReLU()])
        self.advantage_head = nn.Sequential(*[nn.Linear(128, 64), nn.LeakyReLU(),
                                             nn.Linear(64, 64), nn.LeakyReLU()],
                                             nn.Linear(64, self.num_actions), nn.LeakyReLU())

    def forward(self, img, phase):
        x = self.conv(img)
        x = x.view(img.shape[0], -1)
        flat = self.flatten(x)
        phase = self.phase_dense(phase)
        join = torch.cat((flat, phase), 1)
        join = self.final_dense(join)
        value = self.value_head(join)
        adv = self.advantage_head(join)
        if(img.shape[0] != 1):
            qvalues = value + (adv - torch.mean(adv, dim = 1, keepdim = True).repeat(1, self.num_actions))
        else:
            qvalues = value + (adv - torch.mean(adv, dim = 1))
        return qvalues


class VanillaDQN(nn.Module):
    def __init__(self, num_actions = 9, img_size = 100, in_channels = 2, num_phases = 4):
        super(VanillaDQN, self).__init__()
        conv_list = []
        self.num_actions = num_actions
        self.in_channels = in_channels
        self.num_phases = num_phases
        convlist = [nn.Conv2d(self.in_channels, 16, 3, stride = 1),
                    nn.LeakyReLU(),
                    nn.Conv2d(16, 32, 3, stride = 2),
                    nn.LeakyReLU(),
                    nn.Conv2d(32, 64, 3, stride = 2),
                    nn.LeakyReLU(),
                    nn.Conv2d(64, 128, 3, stride = 2),
                    nn.LeakyReLU()]
        self.dim = 11 * 11 * 128

        phasenet = [nn.Linear(self.dim + self.num_phases, 512),
                    nn.LeakyReLU(),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 128),
                    nn.LeakyReLU(),
                    nn.Linear(128, 64),
                    nn.LeakyReLU(),
                    nn.Linear(64, self.num_actions)]
        self.conv = nn.Sequential(*convlist)
        self.phase = nn.Sequential(*phasenet)
    def forward(self, img, phase):
        x = self.conv(img)
        x = x.view(img.shape[0], -1)
        x = torch.cat((x, phase), 1)
        return self.phase(x)

