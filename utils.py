import numpy as np
class PriorityBuffer:
    def __init__(self, max_size = 50000, batch_size = 16):
        self.datalist = []
        self.max_size = max_size
        self.batch_size = batch_size
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