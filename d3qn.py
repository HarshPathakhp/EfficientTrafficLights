"""
Double Dueling Deep Q-Learning
"""

from model import DuelCNN
from env import SumoIntersection
import traci
from utils import PriorityBuffer
from utils import EpsilonPolicy
import torch
STOP_TIME = 99800
START_GREEN = 10
YELLOW = 3
NUM_ACTIONS = 9
PRETRAIN_STEPS = 1000
BATCH_SIZE = 64
BUFFER_SIZE = 50000
""" Notation for actions ->
<t1,t2,t3,t4> -> <t1,t2,t3,t4> 0
                <t1-5,t2,t3,t4> 1
                <t1,t2-5,t3,t4> 2
                <t1,t2,t3-5,t4> 3
                <t1,t2,t3,t4-5> 4
                <t1+5,t2,t3,t4> 5
                <t1,t2+5,t3,t4> 6
                <t1,t2,t3+5,t4> 7
                <t1,t2,t3,t4+5> 8
"""
class D3qn:
    def __init__(self, num_episodes = 1000, use_cuda = False, alpha = 0.01):
        self.env = SumoIntersection("./2way-single-intersection/single-intersection.net.xml", "./2way-single-intersection/single-intersection-vhvh.rou.xml", phases=[
                                traci.trafficlight.Phase(START_GREEN, "GGrrrrGGrrrr"),  
                                traci.trafficlight.Phase(YELLOW, "yyrrrryyrrrr"),
                                traci.trafficlight.Phase(START_GREEN, "rrGrrrrrGrrr"),   
                                traci.trafficlight.Phase(YELLOW, "rryrrrrryrrr"),
                                traci.trafficlight.Phase(START_GREEN, "rrrGGrrrrGGr"),   
                                traci.trafficlight.Phase(YELLOW, "rrryyrrrryyr"),
                                traci.trafficlight.Phase(START_GREEN, "rrrrrGrrrrrG"), 
                                traci.trafficlight.Phase(YELLOW, "rrrrryrrrrry")
                                ], use_gui=False)

        self.primary_model = DuelCNN(num_actions = 9)
        self.target_model = DuelCNN(num_actions = 9)
        self.replaybuffer = PriorityBuffer(max_size = BUFFER_SIZE, batch_size = BATCH_SIZE)
        self.num_eps = num_episodes
        self.use_cuda = use_cuda
        self.epsilon_policy = EpsilonPolicy()
        self.alpha = alpha
        if(self.use_cuda):
            self.primary_model.cuda()
            self.target_model.cuda()
    def get_phase_durations(self, action_id, current_duration):
        ret_phases = current_duration
        if(action_id == 1):
            ret_phases[0] -= 5
        elif(action_id == 2):
            ret_phases[1] -= 5
        elif action_id == 3:
            ret_phases[2] -= 5
        elif action_id == 4:
            ret_phases[3] -= 5
        elif action_id == 5:
            ret_phases[0] += 5
        elif action_id == 6:
            ret_phases[1] += 5
        elif action_id == 7:
            ret_phases[2] += 5
        elif action_id == 8:
            ret_phases[3] += 5
        return ret_phases
    def train(self):
        total_steps = 0
        for eps in range(self.num_eps):
            cur_state = self.env.reset()
            cur_action_phase = [START_GREEN for i in range(4)]
            while(self.env.time <= STOP_TIME):
                total_steps += 1
                cur_state_tensor = torch.from_numpy(cur_state).float()
                if(self.use_cuda):
                    cur_state_tensor = cur_state_tensor.cuda()
                qvalues = self.primary_model(cur_state_tensor)
                action_id = self.epsilon_policy.select(qvalues, NUM_ACTIONS)
                new_phases = self.get_phase_durations(action_id, cur_action_phase)
                new_state, reward = self.env.take_action(new_phases)
                self.replaybuffer.add((cur_state, action_id, new_state, reward))
                cur_state = new_state
                if(self.replaybuffer.length() > BATCH_SIZE and total_steps > PRETRAIN_STEPS):
                    samples = self.replaybuffer.sample(self.primary_model, self.target_model)
                    
                


                

                


        

