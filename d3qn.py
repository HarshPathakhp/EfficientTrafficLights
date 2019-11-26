"""
Double Dueling Deep Q-Learning
"""

from model import DuelCNN
from env import SumoIntersection
import traci
from utils import PriorityBuffer
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
BATCH_SIZE = 64
BUFFER_SIZE = 10000
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
	def __init__(self, num_episodes = 1000, use_cuda = False, alpha = 0.01, discount_factor = 0.99):
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
		self.use_cuda = use_cuda
		self.replaybuffer = PriorityBuffer(max_size = BUFFER_SIZE, batch_size = BATCH_SIZE, use_cuda = self.use_cuda)
		self.num_eps = num_episodes
		self.discount_factor = discount_factor
		self.epsilon_policy = EpsilonPolicy()
		self.alpha = alpha
		if(self.use_cuda):
			self.primary_model.cuda()
			self.target_model.cuda()
		self.criterion = nn.MSELoss()
		self.optimizer = optim.Adam(self.primary_model.parameters(), lr = 1e-4)
		self.writer = open("./Results/3dqn_status.log", "w")
		self.episode_writer = open("./Results/3dqn_episode.log", "w")
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
	def update_targetNet(self):
		dict_primary = {}
		for name, param in self.primary_model.named_parameters():
			if param.requires_grad:
				dict_primary[name] = param
		with torch.no_grad():
			for name, param in self.target_model.named_parameters():
				param.copy_((1 - self.alpha) * param.data + (self.alpha) * dict_primary[name].data)
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
				qvalues = self.primary_model(cur_state_tensor, cur_phase_tensor)
				action_id = self.epsilon_policy.select(qvalues[0].detach().cpu().numpy(), NUM_ACTIONS)
				new_phases = self.get_phase_durations(action_id, cur_action_phase)
				new_state, reward = self.env.take_action(new_phases)
				reward_sum += reward
				if(reward != (int)(-1e6)):
					wait_sum += (-1 * reward)
				reward /= 1e5
				#self.writer.write(str(new_phases) + " " + str(reward) + "\n")
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
					self.primary_model.eval()
					self.target_model.eval()
					samples = self.replaybuffer.sample(self.primary_model, self.target_model)
					
					self.primary_model.train()
					self.target_model.train()
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

					q_theta = self.primary_model(batch_states_from, batch_states_from_phase)
					q_theta_prime = self.target_model(batch_states_to, batch_states_to_phase)
					_,argmax_actions = torch.max(q_theta, 1)
					argmax_actions = argmax_actions.long()
					qprime_vals = q_theta_prime.gather(1, argmax_actions.view(-1,1))
					qprime_vals = qprime_vals.view(-1)
					qtarget = self.discount_factor * qprime_vals + batch_stepreward
					batch_actions = batch_actions.long()
					q_s_a = q_theta.gather(1, batch_actions.view(-1,1))
					qtarget = qtarget.view(-1,1)
					tdloss = self.criterion(q_s_a, qtarget)
					self.writer.write("EPISODE: " + str(eps) + " STEP " + str(total_steps) + ": TDLOSS: " + str(tdloss.item()) + "\n")
					
					tdloss.backward()
					self.optimizer.step()
					self.update_targetNet()
				self.writer.close()
				self.writer = open("./Results/3dqn_status.log", "a")
			wait_sum /= self.env.num_vehicles
			print(self.env.num_vehicles)
			self.episode_writer.write("EPISODE " + str(eps) + ": " + "TOTAL REWARD: " + str(reward_sum) + ", AVGWAITTIME: " + str(wait_sum) + "\n")
			self.episode_writer.close()
			self.episode_writer = open("./Results/3dqn_episode.log", "a")
			traci.close()
			
if __name__ == "__main__":
	os.system("rm -rf Results")
	os.makedirs("./Results")
	d3qn = D3qn(use_cuda = False)
	d3qn.train()




				


				

				


		

