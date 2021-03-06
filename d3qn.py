"""
Double Dueling Deep Q-Learning
"""
from tqdm import tqdm as tqdm
from model import DuelCNN
from env import SumoIntersection
import traci
from utils import PriorityBuffer
from utils import Buffer
from utils import EpsilonPolicy
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import sys
from tqdm import tqdm
from env import MAX_REWARD

STOP_TIME = 4000
START_GREEN = 20
YELLOW = 3
NUM_ACTIONS = 9
REWARD_NORM = 1e5
PRETRAIN_STEPS = 100
BATCH_SIZE = 128
BUFFER_SIZE = 5000
LEARNING_RATE = 1e-4
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
import matplotlib.pyplot as plt
fig = plt.figure(figsize = (15,15))
	
def plot_grad_flow(named_parameters, path):
	#os.makedirs(path, exist_ok = True)
	ave_grads = []
	layers = []
	for n, p in named_parameters:
		if(p.requires_grad) and ("bias" not in n):
			layers.append(n)
			ave_grads.append(p.grad.abs().mean())
	plt.plot(ave_grads, alpha=0.3, color="b")
	plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
	plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
	plt.xlim(xmin=0, xmax=len(ave_grads))
	plt.xlabel("Layers")
	plt.ylabel("average gradient")
	plt.title("Gradient flow")
	plt.grid(True)
	plt.savefig(path)

class D3qn:
	def __init__(self, num_episodes = 2000, use_cuda = False, alpha = 0.01, discount_factor = 0.99, use_priorities = False):
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
		self.primary_model = DuelCNN(num_actions = 9)
		self.target_model = DuelCNN(num_actions = 9)
		self.use_cuda = use_cuda
		self.num_actions = NUM_ACTIONS
		if(not self.use_priorities):
			self.replaybuffer = Buffer(max_size = BUFFER_SIZE, batch_size = BATCH_SIZE)
		else:
			self.replaybuffer = PriorityBuffer(max_size = BUFFER_SIZE, batch_size = BATCH_SIZE, use_cuda = self.use_cuda)
		self.num_eps = num_episodes
		self.discount_factor = discount_factor
		self.epsilon_policy = EpsilonPolicy()
		self.alpha = alpha
		if(self.use_cuda):
			self.primary_model.cuda()
			self.target_model.cuda()
		self.criterion = nn.MSELoss()
		self.optimizer = optim.Adam(self.primary_model.parameters(), lr = LEARNING_RATE)
		self.writer = open("./Results/3dqn_status.txt", "w")
		self.episode_writer = open("./Results/3dqn_episode.txt", "w")
		self.epsilon_writer = open("./Results/3dqn_epsilon.txt", "w")
		self.debug_writer = open("./Results/debug.txt", "w")
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
	def fill_validmoves(self, cur_phase):
		validmoves = [0 for i in range(self.num_actions)]
		for action_id in range(self.num_actions):
			new_phases = self.get_phase_durations(action_id, cur_phase)
			flag = 0
			for time in new_phases:
				if(not(time >= 5 and time <= 60)):
					validmoves[action_id] = 0
					flag = 1
					break
			if(flag == 0):
				validmoves[action_id] = 1
		return validmoves	
	def penalise_bad_moves(self, tensor_phase):
		ret = np.zeros((tensor_phase.shape[0], self.num_actions))
		tensor_phase *= 60
		for idx in range(tensor_phase.shape[0]):
			cur_phase = tensor_phase[idx,:].numpy().tolist()
			valid_moves = self.fill_validmoves(cur_phase)
			for j in range(len(valid_moves)):
				if(valid_moves[j] == 1):
					valid_moves[j] = 0
				else:
					valid_moves[j] = -99999
			ret[idx,:] = valid_moves
		tensor_phase /= 60
		return torch.tensor(ret).float()		

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
			self.debug_writer.write("-"*50 + "\n")
			while(self.env.time <= STOP_TIME):
				self.debug_writer.write(str(cur_action_phase) + "\n")
				total_steps += 1
				cur_action_phase_np = np.array(cur_action_phase)
				cur_action_phase_np = cur_action_phase_np / 60
				cur_state_tensor = torch.from_numpy(cur_state).float().unsqueeze(0)
				cur_phase_tensor = torch.from_numpy(cur_action_phase_np).float().unsqueeze(0)
				if(self.use_cuda):
					cur_state_tensor = cur_state_tensor.cuda()
					cur_phase_tensor = cur_phase_tensor.cuda()
				qvalues = self.primary_model(cur_state_tensor, cur_phase_tensor)
				valid_moves = self.fill_validmoves(cur_action_phase)
				self.debug_writer.write(str(valid_moves) + "\n")
				qvalues = qvalues[0].detach().cpu().numpy()
				new_qvalues = []
				new_actions = []
				for i in range(self.num_actions):
					if(valid_moves[i] == 1):
						new_actions.append(i)
						new_qvalues.append(qvalues[i])
				new_qvalues = np.array(new_qvalues)
				action_id = self.epsilon_policy.select(new_qvalues, len(new_qvalues))
				new_phases = self.get_phase_durations(new_actions[action_id], cur_action_phase)
				new_state, reward = self.env.take_action(new_phases)
				reward_sum += reward
				self.debug_writer.write(str(reward) + "\n")
				if(reward != (int)(-MAX_REWARD)):
					wait_sum += (-1 * reward)
				reward /= REWARD_NORM
				#self.writer.write(str(new_phases) + " " + str(reward) + "\n")
				flag = 0
				for i in new_phases:
					if(not(i > 0 and i <= 60)):
						flag = 1
						break
				if(flag == 1):
					self.debug_writer("oops\n")
					new_phases = cur_action_phase
				if(not self.use_priorities):
					self.replaybuffer.add((cur_state, cur_action_phase, new_actions[action_id], new_state, new_phases, reward))
				else:
					self.replaybuffer.add((cur_state, cur_action_phase, new_actions[action_id], new_state, new_phases, reward), self.primary_model, self.target_model)
				cur_state = new_state
				cur_action_phase = new_phases
				if(self.replaybuffer.length() > BATCH_SIZE and total_steps > PRETRAIN_STEPS):
					self.primary_model.eval()
					self.target_model.eval()
					if(not self.use_priorities):
						samples = self.replaybuffer.sample()
					else:
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

					illegal_costs = self.penalise_bad_moves(batch_states_to_phase)
					if(self.use_cuda):
						batch_states_from = batch_states_from.cuda()
						batch_states_from_phase = batch_states_from_phase.cuda()

						batch_states_to = batch_states_to.cuda()
						batch_states_to_phase = batch_states_to_phase.cuda()

						batch_stepreward = batch_stepreward.cuda()
						batch_actions = batch_actions.cuda()

					q_theta_first_state = self.primary_model(batch_states_from, batch_states_from_phase) 
					q_theta = self.primary_model(batch_states_to, batch_states_to_phase).detach().cpu()
					q_theta_prime = self.target_model(batch_states_to, batch_states_to_phase)
					_,argmax_actions = torch.max(q_theta + illegal_costs, 1)
					if(self.use_cuda):
						argmax_actions = argmax_actions.cuda()
					qprime_vals = q_theta_prime.gather(1, argmax_actions.view(-1,1))
					qprime_vals = qprime_vals.view(-1)
					qtarget = self.discount_factor * qprime_vals + batch_stepreward
					q_s_a = q_theta_first_state.gather(1, batch_actions.view(-1,1))
					qtarget = qtarget.view(-1,1)
					tdloss = self.criterion(q_s_a, qtarget)
					self.writer.write("EPISODE: " + str(eps) + " STEP " + str(total_steps) + ": TDLOSS: " + str(tdloss.item()) + "\n")
					self.epsilon_writer.write("STEP: " + str(self.epsilon_policy.eps) + "\n")
					tdloss.backward()
					#plot_grad_flow(self.primary_model.named_parameters(), "./Gradients/" + str(total_steps) + ".png")
					self.optimizer.step()
					self.update_targetNet()
				self.writer.close()
				self.writer = open("./Results/3dqn_status.txt", "a")
				self.epsilon_writer.close()
				self.epsilon_writer = open("./Results/3dqn_epsilon.txt", "a")
				self.debug_writer.close()
				self.debug_writer = open("./Results/debug.txt", "a")
		
			wait_sum /= self.env.time
			self.episode_writer.write("EPISODE " + str(eps) + ": " + "TOTAL REWARD: " + str(reward_sum) + ", AVGWAITTIME: " + str(wait_sum) + "\n")
			self.episode_writer.close()
			self.episode_writer = open("./Results/3dqn_episode.txt", "a")
			traci.close()
			torch.save(self.primary_model.state_dict(), "./Results/network_d3qn.pth")	
if __name__ == "__main__":
	os.system("rm -rf Gradients")
	os.makedirs("Gradients")
	#os.system("rm -rf Results")
	try:
		os.makedirs("./Results", exist_ok = True)
	except:
		pass
	gpu_avail = torch.cuda.is_available()
	d3qn = D3qn(use_cuda = gpu_avail, use_priorities = False)
	d3qn.train()


				


				

				


		

