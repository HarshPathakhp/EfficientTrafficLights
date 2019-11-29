import numpy as np
import torch
import torch.nn as nn
import math 
class Buffer:
    def __init__(self, max_size = 50000, batch_size = 16):
        self.datalist = []
        self.max_size = max_size
        self.batch_size = batch_size
    
    def add(self, sample):
        """
        add <st,pt,at,st+1,pt+1,rt+1> to list
        """
        if(len(self.datalist) == self.max_size):
            self.datalist.pop(0)
        cur_state = torch.tensor(sample[0]).float().unsqueeze(0)
        cur_state_phase = torch.tensor(sample[1]).float().unsqueeze(0)
        next_state = torch.tensor(sample[3]).float().unsqueeze(0)
        next_state_phase = torch.tensor(sample[4]).float().unsqueeze(0)
        action = torch.tensor(sample[2]).view(1).long()
        reward = torch.tensor(sample[5]).view(1).float()
        self.datalist.append((cur_state, cur_state_phase, action, next_state, next_state_phase, reward))
    
    def sample(self):
        indices = np.random.choice(self.length(), size = (self.batch_size))
        batch_from = []
        batch_from_phase = []
        batch_to = []
        batch_to_phase = []
        batch_actions = []
        batch_reward = []
        for i in indices:
            batch_from.append(self.datalist[i][0])
            batch_from_phase.append(self.datalist[i][1])
            batch_to.append(self.datalist[i][3])
            batch_to_phase.append(self.datalist[i][4])
            batch_actions.append(self.datalist[i][2])
            batch_reward.append(self.datalist[i][5])
        batch_from = torch.cat(batch_from, 0)
        batch_to = torch.cat(batch_to, 0)
        batch_from_phase = torch.cat(batch_from_phase, 0) / 60.0
        batch_to_phase = torch.cat(batch_to_phase, 0) / 60.0
        batch_actions = torch.cat(batch_actions, 0)
        batch_reward = torch.cat(batch_reward, 0)
        return [batch_from, batch_from_phase, batch_actions, batch_to, batch_to_phase, batch_reward]

    def length(self):
        return len(self.datalist)     

class PriorityBuffer:
    def __init__(self, max_size = 50000, batch_size = 16, discount_factor = 0.99, use_cuda = False):
        self.datalist = []
        self.max_size = max_size
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self.use_cuda = use_cuda
    def add(self, sample):
        """
        add <st,pt,at,st+1,pt+1,rt+1> to list
        """
        if(len(self.datalist) == self.max_size):
            self.datalist.pop(0)
        cur_state = torch.tensor(sample[0]).float().unsqueeze(0)
        cur_state_phase = torch.tensor(sample[1]).float().unsqueeze(0)
        next_state = torch.tensor(sample[3]).float().unsqueeze(0)
        next_state_phase = torch.tensor(sample[4]).float().unsqueeze(0)
        action = torch.tensor(sample[2]).view(1).long()
        reward = torch.tensor(sample[5]).view(1).float()
        self.datalist.append((cur_state, cur_state_phase, action, next_state, next_state_phase, reward))
    
    def sample(self, primary_net, target_net):
        """
        samples <batch_size> elements from minibatch
        """

        num_samples = len(self.datalist)
        use_cuda = self.use_cuda

        td_error = []
        for i in range(num_samples):
            cur_state = self.datalist[i][0]
            cur_state_phase = self.datalist[i][1]
            action = self.datalist[i][2]
            next_state = self.datalist[i][3]
            next_state_phase = self.datalist[i][4]
            reward = self.datalist[i][5]

            if(self.use_cuda):
                cur_state = cur_state.cuda()
                cur_state_phase = cur_state_phase.cuda()
                next_state = next_state.cuda()
                next_state_phase = next_state_phase.cuda()
                reward = reward.cuda()

            best_action = torch.argmax(primary_net(next_state, next_state_phase).view(-1)).item()
            next_state_estimate = target_net(next_state, next_state_phase).view(-1)[best_action]
            bootstrapped_output = reward + self.discount_factor * next_state_estimate 
            target_output = primary_net(cur_state, cur_state_phase).view(-1)[action]
            
            error = torch.abs(bootstrapped_output - target_output).item()	
            td_error.append([error, i])	

        td_error.sort()
        td_error.reverse()
        distribution = np.zeros(len(self.datalist)).astype(float)
        
        for i in range(num_samples):
            distribution[td_error[i][1]] = 1 / (i + 1)

        distribution = distribution / distribution.sum()
        indices = np.random.choice(num_samples, size = (self.batch_size), replace = False, p = distribution)
        
        batch_from = []
        batch_from_phase = []
        batch_to = []
        batch_to_phase = []

        batch_actions = []
        batch_reward = []
        for i in indices:
            batch_from.append(self.datalist[i][0])
            batch_from_phase.append(self.datalist[i][1])
            batch_to.append(self.datalist[i][3])
            batch_to_phase.append(self.datalist[i][4])
            batch_actions.append(self.datalist[i][2])
            batch_reward.append(self.datalist[i][5])
        batch_from = torch.cat(batch_from, 0)
        batch_to = torch.cat(batch_to, 0)
        batch_from_phase = torch.cat(batch_from_phase, 0) / 60.0
        batch_to_phase = torch.cat(batch_to_phase, 0) / 60.0
        batch_actions = torch.cat(batch_actions, 0)
        batch_reward = torch.cat(batch_reward, 0)
        return [batch_from, batch_from_phase, batch_actions, batch_to, batch_to_phase, batch_reward]

    def length(self):
        return len(self.datalist)



class EpsilonPolicy:
    def __init__(self, start_eps = 1, decay = 0.00008, min_epsilon = 0.001):
        self.eps = start_eps
        self.decay = decay
        self.min_epsilon = min_epsilon
    def select(self, qvalues, num_actions):
        """
        qvalues is of dimension (batch_size * num_actions). This is for each state
        """
        toss = np.random.rand()
        if toss < self.eps:
            action = (np.random.choice(num_actions))  
        else:
            argmax = np.argmax(qvalues)
            action = (argmax)
        self.eps = max(self.min_epsilon, self.eps * math.exp(-1 * self.decay))
        return action 

if(__name__ == "__main__"):

	pass
