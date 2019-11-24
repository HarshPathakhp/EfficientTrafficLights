import numpy as np
import torch
import torch.nn as nn

class PriorityBuffer:
    def __init__(self, max_size = 50000, batch_size = 16, discount_factor = 0.99):
        self.datalist = []
        self.max_size = max_size
        self.batch_size = batch_size
        self.discount_factor = discount_factor

    def add(self, sample):
        """
        add <st,at,st+1,rt+1> to list
        """
        if(len(self.datalist) == self.max_size):
            self.datalist.pop(0)
        self.datalist.append(sample)
    def sample(self, primary_net, target_net):
        """
        samples <batch_size> elements from minibatch
        """

        num_samples = len(datalist)

        td_error = []
        for i in range(num_samples):

        	cur_state = torch.tensor(self.datalist[i][0]).float().cuda().unsqueeze(0)
        	action = self.datalist[i][1]
        	next_state = torch.tensor(self.datalist[i][2]).float().cuda().unsqueeze(0)
        	reward = torch.tensor([self.datalist[i][3]])

        	best_action = torch.argmax(primary_net(next_state)).item()
        	next_state_estimate = target_net(next_state)[best_action]
        	bootstrapped_output = reward + self.discount_factor * next_state_estimate 
        	target_output = target_net(cur_state)[action]
        	
        	error = torch.abs(bootstrapped_output - target_output).item()	
        	td_error.append([error, i])	

        td_error.sort()
        td_error.reverse()
        distribution = np.zeros(len(self.datalist)).astype(float)
        
        for i in range(num_samples):
        	distribution[td_error[i][1]] = 1 / (i + 1)

        distribution = distribution / distribution.sum()
        indices = np.random.choice(num_samples, size = (self.batch_size) replace = False, p = distribution)

        return self.datalist[indices]
        
    def length(self):
        return len(self.datalist)




class EpsilonPolicy:
    def __init__(self, start_eps = 0.3, decay = 0.99, min_epsilon = 0.05):
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
        self.eps = max(self.min_epsilon, self.eps * self.decay)
        return action 