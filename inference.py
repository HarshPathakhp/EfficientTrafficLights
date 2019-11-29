from env import SumoIntersection
import traci
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

model_path = sys.argv[1]
STOP_TIME = 4000
START_GREEN = 40
YELLOW = 3
num_actions = 9

env = SumoIntersection("./2way-single-intersection/single-intersection.net.xml", "./2way-single-intersection/single-intersection-vhvh.rou.xml", phases=[
								traci.trafficlight.Phase(START_GREEN, "GGrrrrGGrrrr"),  
								traci.trafficlight.Phase(YELLOW, "yyrrrryyrrrr"),
								traci.trafficlight.Phase(START_GREEN, "rrGrrrrrGrrr"),   
								traci.trafficlight.Phase(YELLOW, "rryrrrrryrrr"),
								traci.trafficlight.Phase(START_GREEN, "rrrGGrrrrGGr"),   
								traci.trafficlight.Phase(YELLOW, "rrryyrrrryyr"),
								traci.trafficlight.Phase(START_GREEN, "rrrrrGrrrrrG"), 
								traci.trafficlight.Phase(YELLOW, "rrrrryrrrrry")
								], use_gui=True)

def get_phase_durations(action_id, current_duration):
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

def fill_validmoves(cur_phase):
		validmoves = [0 for i in range(num_actions)]
		for action_id in range(num_actions):
			new_phases = get_phase_durations(action_id, cur_phase)
			flag = 0
			for time in new_phases:
				if(not(time >= 5 and time <= 60)):
					validmoves[action_id] = 0
					flag = 1
					break
			if(flag == 0):
				validmoves[action_id] = 1
		return validmoves

primary_model = DuelCNN(num_actions = 9)
primary_model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
primary_model.eval()

state = env.reset()
phase = [START_GREEN for i in range(4)]

os.system("rm -rf Inference")
os.makedirs("Inference")
f = open("Inference/run.txt", "w")
reward_sum = 0
while(env.time <= STOP_TIME):
	f.write(str(phase) + "\n")
	phase_np = np.array(phase)
	phase_np = phase_np / 60
	state_tensor = torch.from_numpy(state).float().unsqueeze(0)
	phase_tensor = torch.from_numpy(phase_np).float().unsqueeze(0)			
	qvalues = primary_model(state_tensor, phase_tensor)
	valid_moves = fill_validmoves(phase)
	
	qvalues = qvalues[0].detach().cpu().numpy()
	new_qvalues = []
	new_actions = []
	for i in range(num_actions):
		if(valid_moves[i] == 1):
			new_actions.append(i)
			new_qvalues.append(qvalues[i])
	
	new_qvalues = np.array(new_qvalues)
	action_id = np.argmax(new_qvalues)
	new_phase = get_phase_durations(new_actions[action_id], phase)
	new_state, reward = env.take_action(new_phase)

	state = new_state
	phase = new_phase

	reward_sum += reward
f.write("total reward is " + str(reward_sum) + "\n")
f.write("average waiting time is " + str(reward_sum/env.time) + "\n")
traci.close()
f.close()
