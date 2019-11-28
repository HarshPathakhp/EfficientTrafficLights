from tqdm import tqdm as tqdm
from env import SumoIntersection
import os
import traci
YELLOW = 3
STOP_TIME = 10000
class FixedBaseline():
	def __init__(self, num_eps = 1000, ph1 = 10, ph2 = 10, ph3 = 10, ph4 = 10):
		self.time1 = ph1
		self.time2 = ph2
		self.time3 = ph3
		self.time4 = ph4
		self.num_episodes = num_eps
		self.phases = [self.time1, self.time2, self.time3, self.time4]
		self.env = SumoIntersection("./2way-single-intersection/single-intersection.net.xml", "./2way-single-intersection/single-intersection-vhvh.rou.xml", phases=[
								traci.trafficlight.Phase(self.time1, "GGrrrrGGrrrr"),  
								traci.trafficlight.Phase(YELLOW, "yyrrrryyrrrr"),
								traci.trafficlight.Phase(self.time2, "rrGrrrrrGrrr"),   
								traci.trafficlight.Phase(YELLOW, "rryrrrrryrrr"),
								traci.trafficlight.Phase(self.time3, "rrrGGrrrrGGr"),   
								traci.trafficlight.Phase(YELLOW, "rrryyrrrryyr"),
								traci.trafficlight.Phase(self.time4, "rrrrrGrrrrrG"), 
								traci.trafficlight.Phase(YELLOW, "rrrrryrrrrry")
								], use_gui=False)
		self.episode_writer = open("./Results/fixed" + str(ph1) + "_" + str(ph2) + "_" + str(ph3) + "_" + str(ph4) + "_episode.txt", "w")
	def run(self):
		for eps in tqdm(range(self.num_episodes)):
			wait_sum = 0
			reward_sum = 0
			cur_state = self.env.reset()
			while(self.env.time <= STOP_TIME):
				new_state, reward = self.env.take_action(self.phases)
				reward_sum += reward
				wait_sum += -1*reward if (reward != int(-1e6)) else 0
			wait_sum /= self.env.time
			self.episode_writer.write("EPISODE " + str(eps) + ": " + "TOTAL REWARD: " + str(reward_sum) + ", AVGWAITTIME: " + str(wait_sum) + "\n")
			self.episode_writer.close()
			self.episode_writer = open("./Results/fixed" + str(self.time1) + "_" + str(self.time2) + "_" + str(self.time3) + "_" + str(self.time4) + "_episode.txt", "a")
			traci.close()
if __name__ == "__main__":
	#os.system("rm -rf ./Results_fixed")
	try:
		os.makedirs("./Results")
	except:
		pass
	fx = FixedBaseline(ph1 = 30, ph2 = 30, ph3 = 30, ph4 = 30)
	fx.run()
	fx2 = FixedBaseline(ph1 = 40, ph2 = 40, ph3 = 40, ph4 = 40)
	fx2.run()
