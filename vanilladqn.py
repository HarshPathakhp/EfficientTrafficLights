"""
Vanilla deep Q-learning
"""
from tqdm import tqdm as tqdm
from model import VanillaDQN
from env import SumoIntersection
import traci
from utils import PriorityBuffer
from utils import Buffer
from torch.autograd import Variable
from utils import EpsilonPolicy
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import sys
from tqdm import tqdm
STOP_TIME = 10000
START_GREEN = 20
YELLOW = 3
NUM_ACTIONS = 9
PRETRAIN_STEPS = 100
BATCH_SIZE = 128
BUFFER_SIZE = 20000
EPSILON = 0.4
REWARD_NORM = 1e3
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
class Dqn:
	def __init__(self, num_episodes = 2000, use_cuda = False, discount_factor = 0.99, use_priorities = False):
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
		self.use_priorities  = use_priorities
		self.use_cuda = use_cuda
		self.model = VanillaDQN(num_actions = NUM_ACTIONS)
		if(not self.use_priorities):
			self.replaybuffer = Buffer(max_size = BUFFER_SIZE, batch_size = BATCH_SIZE)
		else:
			self.replaybuffer = PriorityBuffer(max_size = BUFFER_SIZE, batch_size = BATCH_SIZE, use_cuda = self.use_cuda)
		self.num_eps = num_episodes
		self.discount_factor = discount_factor
		self.epsilon_policy = EpsilonPolicy(start_eps = EPSILON)
		if(self.use_cuda):
			self.model.cuda()
		self.criterion = nn.MSELoss()
		self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-4)
		
		self.writer = open("./Results_dqn/dqn_status.txt", "w")
		self.episode_writer = open("./Results_dqn/dqn_episode.txt", "w")
		self.epsilon_writer = open("./Results_dqn/dqn_epsilon.txt", "w")
	
	def get_phase_durations(self, action_id, current_duration):
		ret_phases = [i for i in current_duration]
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
		for eps in tqdm(range(self.num_eps)):
			cur_state = self.env.reset()
			cur_action_phase = [START_GREEN for i in range(4)]
			reward_sum = 0
			wait_sum = 0
			while(self.env.time <= STOP_TIME):
				total_steps += 1
				cur_action_phase_np = np.array(cur_action_phase)
				cur_action_phase_np = cur_action_phase_np / 60
				cur_state_tensor = torch.from_numpy(cur_state).float().unsqueeze(0)
				cur_phase_tensor = torch.from_numpy(cur_action_phase_np).float().unsqueeze(0)
				if(self.use_cuda):
					cur_state_tensor = cur_state_tensor.cuda()
					cur_phase_tensor = cur_phase_tensor.cuda()
				qvalues = self.model(cur_state_tensor, cur_phase_tensor)

				action_id = self.epsilon_policy.select(qvalues[0].detach().cpu().numpy(), NUM_ACTIONS)
				new_phases = self.get_phase_durations(action_id, cur_action_phase)
				new_state, reward = self.env.take_action(new_phases)
				reward_sum += reward
				if(reward != (int)(-1e6)):
					wait_sum += (-1 * reward)
				reward /= REWARD_NORM
				flag = 0
				for i in new_phases:
					if(not(i > 0 and i <= 60)):
						flag = 1
						break
				if(flag == 1):
					new_phases = cur_action_phase
				self.replaybuffer.add((cur_state, cur_action_phase, action_id, new_state, new_phases, reward))
				cur_state = new_state
				cur_action_phase = new_phases

				if(self.replaybuffer.length() > BATCH_SIZE and total_steps > PRETRAIN_STEPS):
					self.model.eval()
					samples = None
					if(not self.use_priorities):
						samples = self.replaybuffer.sample()
					else:
						print("PRIORITY NOT IMPLEMENTED YET ON DQN")
						sys.exit(-1)
					self.model.train()
					self.optimizer.zero_grad()
					
					batch_states_from = samples[0]
					batch_states_from_phase = samples[1]
					batch_states_to = samples[3]
					batch_states_to_phase = samples[4]
					batch_actions = samples[2]
					batch_stepreward = samples[5]
					if(self.use_cuda):
						batch_states_from = batch_states_from.cuda()
						batch_states_from_phase = batch_states_from_phase.cuda()

						batch_states_to = batch_states_to.cuda()
						batch_states_to_phase = batch_states_to_phase.cuda()

						batch_stepreward = batch_stepreward.cuda()
						batch_actions = batch_actions.cuda()

					q_theta = self.model(batch_states_from, batch_states_from_phase)
					q_theta_prime = self.model(batch_states_to, batch_states_to_phase)
					q_theta_prime = Variable(q_theta_prime.detach(), requires_grad = False)
					if(self.use_cuda):
						q_theta_prime = q_theta_prime.cuda()
					maxv, _ = torch.max(q_theta_prime, 1)
					qtarget = batch_stepreward + self.discount_factor * maxv
					q_s_a = q_theta.gather(1, batch_actions.view(-1,1))
					qtarget = qtarget.view(-1,1)
					tdloss = self.criterion(qtarget, q_s_a)
					self.writer.write("EPISODE: " + str(eps) + " STEP " + str(total_steps) + ": TDLOSS: " + str(tdloss.item()) + "\n")
					self.epsilon_writer.write("STEP: " + str(self.epsilon_policy.eps) + "\n")
					tdloss.backward()
					self.optimizer.step()
				
					self.writer.close()
					self.epsilon_writer.close()
					self.writer = open("./Results_dqn/dqn_status.txt", "a")
					self.epsilon_writer = open("./Results_dqn/dqn_epsilon.txt", "a")
		
			wait_sum /= self.env.num_vehicles
			print(self.env.num_vehicles)
			self.episode_writer.write("EPISODE " + str(eps) + ": " + "TOTAL REWARD: " + str(reward_sum) + ", AVGWAITTIME: " + str(wait_sum) + "\n")
			self.episode_writer.close()
			self.episode_writer = open("./Results_dqn/dqn_episode.txt", "a")
			traci.close()

if __name__ == "__main__":
	os.system("rm -rf Results_dqn")
	os.makedirs("./Results_dqn")
	dqn = Dqn(use_cuda = False, use_priorities = False)
	dqn.train()

