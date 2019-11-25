import os, sys
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

class TrafficSignal:
    def __init__(self, env, ts_id, delta_time, min_green, max_green, phases,yellow_time):
        self.id = ts_id
        self.env = env
        self.time_on_phase = 0
        self.delta_time = delta_time
        self.min_green = min_green
        self.max_green = max_green
        self.green_phase = 0
        self.yellow_time = yellow_time
        self.num_green_phases = len(phases)//2
        self.lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.id)))
        self.cur_phase = phases
        logic = traci.trafficlight.Logic("0", 0, 0, phases = self.cur_phase)
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.id, logic)
    
    def _phase(self):
        return traci.trafficlight.getPhaseDuration(self.id)
    
    def set_new_logic(self, durations):
        """change the tls duration logic"""
        new_phase = [traci.trafficlight.Phase(durations[0], "GGrrrrGGrrrr"),  
                                        traci.trafficlight.Phase(self.yellow_time, "yyrrrryyrrrr"),
                                        traci.trafficlight.Phase(durations[1], "rrGrrrrrGrrr"),   
                                        traci.trafficlight.Phase(self.yellow_time, "rryrrrrryrrr"),
                                        traci.trafficlight.Phase(durations[2], "rrrGGrrrrGGr"),   
                                        traci.trafficlight.Phase(self.yellow_time, "rrryyrrrryyr"),
                                        traci.trafficlight.Phase(durations[3], "rrrrrGrrrrrG"), 
                                        traci.trafficlight.Phase(self.yellow_time, "rrrrryrrrrry")]
        logic = traci.trafficlight.Logic("0", 0, 0, phases = new_phase) 
        traci.trafficlight.setCompleteRedYellowGreenDefinition(self.id, logic)
        return
    
