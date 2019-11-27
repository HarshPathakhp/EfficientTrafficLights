import os, sys
MAX_REWARD = 1e7

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci
import traci.constants as tc
import sumolib
import numpy as np
from gym import Env
from gym import spaces
import numpy as np
import pandas as pd
from  traffic_signal import TrafficSignal
np.set_printoptions(threshold=sys.maxsize)
temp = os.environ['SUMO_HOME'].find("share")
sumoBinary = os.environ['SUMO_HOME'][:temp] + "bin/"

class SumoIntersection:
    """
    SUMO environment for traffic signal control
    :param net_file: (str).net.xml file
    :param route_file (str).rou.xml file
    :param phases
    """
    def __init__(self, net_file, route_file, phases, use_gui = False, num_seconds = 2e4,max_depart_delay = 1e5, time_to_load_vehicles = 0, delta_time = 5, min_green = 5, max_green = 50):
        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_bin = os.path.join(sumoBinary, 'sumo-gui')
        else:
            self._sumo_bin = os.path.join(sumoBinary, 'sumo')
        traci.start([os.path.join(sumoBinary,'sumo'), '-n', self._net])
        self.ts_ids = traci.trafficlight.getIDList()
        self.lanes = set(traci.trafficlight.getControlledLanes(self.ts_ids[0]))
        self.lanes_per_ts = len(self.lanes)
        self.traffic_signals = dict()
        self.phases = phases
        self.num_green_phases = len(phases)//2
        self.vehicles = dict()
        self.last_measure = dict()
        self.last_reward = {i : 0 for i in self.ts_ids}
        self.sim_max_time = num_seconds
        self.time_to_load_vehicles = time_to_load_vehicles
        self.delta_time = delta_time
        self.max_depart_delay = max_depart_delay
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = 10
        self.run = 0
        self.vehicle_width = None
        self.vehicle_length = None
        self.existing_waiting_time = {}
        self.num_vehicles = 0
        self.time = 0
        traci.close()
    
    def reset(self):
        self.run += 1
        self.metrics = []
        sumo_cmd = [self._sumo_bin, '-n', self._net, '-r', self._route, '--max-depart-delay', str(self.max_depart_delay), '--waiting-time-memory', '10000', '--random']
        if self.use_gui:
            sumo_cmd.append('--start')
        traci.start(sumo_cmd)
        for ts in self.ts_ids:
            self.traffic_signals[ts] = TrafficSignal(self, ts, self.delta_time, self.min_green, self.max_green, self.phases, self.yellow_time)
            self.last_measure[ts] = 0
        self.vehicles = dict()
        for _ in range(self.time_to_load_vehicles):
            self._sumo_step()
        self.existing_waiting_time = {}
        self.num_vehicles = 0
        self.time = 0
        return self._compute_observation()

    def _sumo_step(self):
        traci.simulationStep()
        self.time += 1
    
    def _compute_observation(self):
        controlled_lanes = self.lanes
        state_matrix = np.zeros((2,100,100))
        for lane in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            for v in vehicles:
                pos = traci.vehicle.getPosition(v)
                speed = traci.vehicle.getSpeed(v)
                maxspeed = traci.vehicle.getMaxSpeed(v)
                state_matrix[0,99 - int(pos[1]/3), int(pos[0]/3)] = 1
                state_matrix[1,99 - int(pos[1]/3), int(pos[0]/3)] = speed / maxspeed
        return state_matrix

    def take_action(self, phase_durations):
        """
        parameters - phase_duration : list of length 4 for each green phase denoting green light to run for
        """
        sum = 0
        for i in phase_durations:
            if(not (i > 0 and i <= 60)):
                return self._compute_observation(), (int)(-MAX_REWARD)
        for i in phase_durations:
            sum += i + self.yellow_time
        traci.trafficlight.setPhase(self.ts_ids[0], 0)
        self.traffic_signals['t'].set_new_logic(phase_durations)
        traci.trafficlight.setPhase(self.ts_ids[0], 0)
        waiting_time_map = {}
        waiting_time_map = self.util_update_waits(waiting_time_map)
        for time in phase_durations:
            for _ in range(time):
                self._sumo_step()
                waiting_time_map = self.util_update_waits(waiting_time_map)
            for _ in range(self.yellow_time):
                self._sumo_step()
                waiting_time_map = self.util_update_waits(waiting_time_map)
        wait_time = 0
        for k,v in waiting_time_map.items():
            if(self.existing_waiting_time.get(k,-1) != -1):
                wait_time += (v - self.existing_waiting_time.get(k))
            else:
                self.num_vehicles += 1
                wait_time += v
        final_vehicle_list = traci.vehicle.getIDList()
        self.existing_waiting_time = {}
        for veh in final_vehicle_list:
            self.existing_waiting_time[veh] = waiting_time_map[veh]
        reward = -1 * wait_time
        return self._compute_observation(), reward
    
    def util_update_waits(self, to_update_wait_map):
        cur_vehicles = traci.vehicle.getIDList()
        for veh in cur_vehicles:
            to_update_wait_map[veh] = traci.vehicle.getAccumulatedWaitingTime(veh)
        return to_update_wait_map
    
if __name__ == "__main__":
    si = SumoIntersection("./2way-single-intersection/single-intersection.net.xml", "./2way-single-intersection/single-intersection-vhvh.rou.xml", phases=[
                                        traci.trafficlight.Phase(32, "GGrrrrGGrrrr"),  
                                        traci.trafficlight.Phase(2, "yyrrrryyrrrr"),
                                        traci.trafficlight.Phase(32, "rrGrrrrrGrrr"),   
                                        traci.trafficlight.Phase(2, "rryrrrrryrrr"),
                                       traci.trafficlight.Phase(32, "rrrGGrrrrGGr"),   
                                        traci.trafficlight.Phase(2, "rrryyrrrryyr"),
                                       traci.trafficlight.Phase(32, "rrrrrGrrrrrG"), 
                                        traci.trafficlight.Phase(2, "rrrrryrrrrry")
                                        ], use_gui=False)

    #HOW TO RUN
    state_1 = si.reset()
    r = []
    for i in range(20):
        state_2, r1 = si.take_action([10,10,10,10])
        r.append(r1)
    print(r)
    traci.close()
    #state_2 = si.reset()
    #for i in range(100):
    #    for j in range(100):
    #        x = 'H' if state_2[i,j,0] == 1 else '-'
    #        print(x, end = " ")
    #    print()
    #print("\n"*3)
    #for i in range(100):
    #    for j in range(100):
    #        x = 'H' if state_3[i,j,0] == 1 else '-'
    #        print(x, end = " ")
    #    print()
    #
    #traci.close()
