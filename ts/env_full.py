import random
import numpy as np
from collections import deque
import gym
from numpy.lib.polynomial import _poly_dispatcher



MC_POS = np.array([500, 500])
MC_V = 5
MC_CHARGING_POWER = 5
WORST_REWARD = -8000
TIME_INTERVAL = 150




class Networks:
    def __init__(self, file: str):
        # self.time = 0
        self.number_of_nodes = None
        self.remaining_energy = None
        self.min_threshold = 540
        self.max_threshold = 8000
        self.min_E = None
        self.max_E = None
        self.ecr = None  # energy consumption rate
        self.remaining_time = None
        self.coords = None  # coordinate
        self.distance_matrix = None
        # self.init_remaining_time = None
        self._initialize(file)



    def _initialize(self, file: str):
        # the first place is for base station
        f = open(file)
        self.number_of_nodes = int(f.readline().split()[0])
        remaining_energy = [1]
        ecr = [1]
        coords = [np.array([int(x) for x in f.readline().split()])]
        distance_matrix = np.zeros((self.number_of_nodes+1, self.number_of_nodes+1))



        for i in range(1, self.number_of_nodes + 1):
            data = f.readline().split()
            coords.append(np.array([float(data[0]), float(data[1])]))
            ecr.append(float(data[2]))
            remaining_energy.append(float(data[3]))
            # print(i, pos, energy_consumption_rate, E_remain)



        self.remaining_energy = np.array(remaining_energy)
        self.min_E = np.array([self.min_threshold for _ in range(self.number_of_nodes + 1)])
        self.max_E = np.array([self.max_threshold for _ in range(self.number_of_nodes + 1)])
        self.ecr = np.array(ecr)
        self.coords = np.array(coords)
        self.remaining_time = np.divide(self.remaining_energy, self.ecr)
        # self.init_remaining_time = copy.copy(self.remaining_time)


        fn = lambda pos1, pos2: np.sqrt(sum(t**2 for t in (pos1 - pos2)))
        for i in range(self.number_of_nodes+1):
            for j in range(i+1, self.number_of_nodes + 1):
                distance_matrix[i, j] = fn(self.coords[i], self.coords[j])
                distance_matrix[j, i] = distance_matrix[i, j]
        self.distance_matrix = distance_matrix


    def update(self, time: float, charged_node=None):
        self.remaining_time[1:] -= time
        self.remaining_energy -= self.ecr * time
        if charged_node is not None:
            self.remaining_time[charged_node] += time
        # for i in range(1, len(self.remaining_time)):
        #     if self.remaining_energy[i] < self.min_E[i]:
        #         # print(i, self.remaining_energy[i])
        #         # self.remaining_time[1:] += time
        #         # if charged_node is not None: self.remaining_time[charged_node] -= time
        #         return True
        # return False
    def get_dead_nodes(self):
        cnt = 0


        for i in range(1, len(self.remaining_energy)):
            if self.remaining_energy[i] < self.min_E[i]:
                cnt += 1
        return cnt



MC_ENERGY = 42000
MC_MOVING_ENERGY = 10


class MC:
    def __init__(self, pos_id: int, pos: np.ndarray, velocity: int, charging_power: float):
        self.pos_id = pos_id
        self.pos = pos
        self.v = velocity
        self.charging_power = charging_power
        self.energy = MC_ENERGY
        self.energy_in_moving = 0.0
        self.energy_on_receiving = 0.0



    def move_to(self, pos_id: int, net: Networks):
        time = net.distance_matrix[self.pos_id, pos_id] / self.v
        self.energy -= net.distance_matrix[self.pos_id, pos_id] * MC_MOVING_ENERGY
        self.energy_in_moving += net.distance_matrix[self.pos_id, pos_id] * MC_MOVING_ENERGY
        self.pos_id = pos_id
        return time



    def charge(self, pos_id: int, ratio: float, net: Networks):
        delta_E = (net.max_E[pos_id] - net.remaining_energy[pos_id]) * ratio
        consume_energy = delta_E / 0.8
        self.energy -= consume_energy
        self.energy_on_receiving += consume_energy
        # print(net.remaining_energy[pos_id])
        net.remaining_energy[pos_id] += delta_E
        # print(net.remaining_energy[pos_id])
        net.remaining_time[pos_id] += delta_E / net.ecr[pos_id]
        return consume_energy / self.charging_power


RETURN_TO_BS = 0
RETURN_TO_BS_REWARD = 100


class Environment(gym.Env):
    """
    *s tate: n*1 : remaining time
    * action: 2*1 : node , charging ratio
    * alpha = mean(max_remaining_time) / min_remaining_time
    """
    def __init__(self, file: str, seed=0):
        self.seed = seed
        random.seed(self.seed)
        self.file = file
        self.net = Networks(self.file)
        self.mc = MC(0, MC_POS, MC_V, MC_CHARGING_POWER)
        self.state = None
        self.done = False
        self.action = None
        self.min_remaining_time = None
        self.avg_remaining_time = None
        self.alpha = np.mean(np.divide(self.net.max_E[1:], self.net.ecr[1:])) / np.amin(self.net.remaining_time[1:])
        self.beta = 1 / (self.alpha + 1)
        self.start = []
        self.cur_step = 0
        self._max_episode_steps = 50
        self.state = self.net.remaining_energy[1:]
        # self.observation_space = (self.net.number_of_nodes,)
        # self.action_shape = (2,)
        self.action_space = gym.spaces.Discrete(self.net.number_of_nodes + 1)
        self.observation_space = gym.spaces.Box(low=np.finfo(np.float32).min,high=np.finfo(np.float32).max,shape=((self.net.number_of_nodes + 1),),dtype=np.float32)
        
    def check(self):
        if self.mc.energy < 0:
            return True
        if self.net.get_dead_nodes() > 0:
            return True
        return False


    def reset(self):
        self.cur_step = 0

        self.net = Networks(self.file)
        self.mc = MC(0, MC_POS, MC_V, MC_CHARGING_POWER)
        self.state = None
        self.done = False
        self.action = None
        self.min_remaining_time = None
        self.avg_remaining_time = None
        self.alpha = np.mean(np.divide(self.net.max_E[1:], self.net.ecr[1:])) / np.amin(self.net.remaining_time[1:])
        self.beta = 1 / (self.alpha + 1)
        self.start = []
        self.cur_step = 0
        self._max_episode_steps = 50
        self.state = self.net.remaining_time[1:]


        self.action = None
        self.net.remaining_time = np.concatenate(([1], self.net.remaining_time[1:]))
        self.net.remaining_energy = np.multiply(self.net.remaining_time, self.net.ecr)
        self.min_remaining_time = np.amin(self.state)
        self.avg_remaining_time = np.mean(self.state)
        a = np.concatenate([[self.mc.energy / MC_ENERGY],self.state / (np.max(self.state) - np.min(self.state))])   
        return a


    def step(self, action):
        if action == self.action:
            self.net.update(TIME_INTERVAL)
            if self.check():
                a = np.concatenate([[self.mc.energy / MC_ENERGY],self.state / (np.max(self.state) - np.min(self.state))])
                # a = np.concatenate([[self.mc.energy],self.state])
                return a,WORST_REWARD,True,{"time":0}


            next_state = self.net.remaining_time[1:]
            min_remaining_time = np.amin(self.net.remaining_time[1:])
            avg_remaining_time = np.mean(self.net.remaining_time[1:])
            reward = -100


            # reward = self.beta * (avg_remaining_time - self.avg_remaining_time) + \
            #         (1-self.beta)*(min_remaining_time - self.min_remaining_time)

            self.min_remaining_time = min_remaining_time
            self.avg_remaining_time = avg_remaining_time
            self.state = next_state
            self.action = action

            a = np.concatenate([[self.mc.energy / MC_ENERGY],self.state / (np.max(self.state) - np.min(self.state))])
            # a = np.concatenate([[self.mc.energy],self.state ])
            return a, reward, False,{"time":TIME_INTERVAL}


        if action == RETURN_TO_BS:
            self.action = action
            times = self.mc.move_to(action,self.net)
            
            if self.check():
                a = np.concatenate([[self.mc.energy / MC_ENERGY],self.state / (np.max(self.state) - np.min(self.state))])
                # a = np.concatenate([[self.mc.energy],self.state])
                return a,WORST_REWARD,True,{"time":0}
            self.net.update(times)
            reward = - self.mc.energy / MC_ENERGY * 100
            self.mc.energy = MC_ENERGY
            self.state = self.net.remaining_time[1:]

            min_remaining_time = np.amin(self.net.remaining_time[1:])
            avg_remaining_time = np.mean(self.net.remaining_time[1:])

            # reward = self.beta * (avg_remaining_time - self.avg_remaining_time) + \
            #         (1-self.beta)*(min_remaining_time - self.min_remaining_time)

            
            
            self.min_remaining_time = min_remaining_time
            self.avg_remaining_time = avg_remaining_time
            a = np.concatenate([[self.mc.energy / MC_ENERGY],self.state / (np.amax(self.state) - np.min(self.state))])
            # a = np.concatenate([[self.mc.energy],self.state])
            return a,reward,False,{"time":times}
        else:
            # self.cur_step += 1
            # if self.cur_step >= self._max_episode_steps:
            #     return None, self.net.get_dead_nodes() * WORST_REWARD , True,{}
            times = 0


            moving_time = self.mc.move_to(action, self.net)


            charging_time = self.mc.charge(action, 1.0, self.net)


            # print(f"charge time : {charging_time}")
            # print(f"speed : {self.net.ecr[1]}")
            # print(charging_time)
            # self.net.update(charging_time, node)
            # if done:
            #     # return None, WORST_REWARD, times, True, None
            #     return self.state, WORST_REWARD, True, {'time':times}
            times += moving_time + charging_time
            self.net.update(times, action)
            if self.check():
                a = np.concatenate([[self.mc.energy / MC_ENERGY],self.state / (np.max(self.state) - np.min(self.state))])
                # a = np.concatenate([[self.mc.energy],self.state])
                return a,WORST_REWARD,True,{"time":0}


            next_state = self.net.remaining_time[1:]
            min_remaining_time = np.amin(self.net.remaining_time[1:])
            avg_remaining_time = np.mean(self.net.remaining_time[1:])
            # reward = self.beta * (avg_remaining_time - self.avg_remaining_time) + \
            #         (1-self.beta)*(min_remaining_time - self.min_remaining_time)

            # reward = 0.5* (charging_time * self.mc.charging_power / self.net.max_E[action]) + \
            #         (0.5)*(100/(moving_time * MC_V  * MC_MOVING_ENERGY))
            reward = charging_time / moving_time 
            # print(reward)

            self.min_remaining_time = min_remaining_time
            self.avg_remaining_time = avg_remaining_time
            self.state = next_state
            self.action = action
            a = np.concatenate([[self.mc.energy / MC_ENERGY],self.state / (np.max(self.state) - np.min(self.state))])
            # a = np.concatenate([[self.mc.energy],self.state])
            return a, reward, False,{"time":times}
    def can_charge_next(self,node):
        energy = self.net.distance_matrix[self.mc.pos_id, node] * MC_MOVING_ENERGY
        delta_E = (self.net.max_E[node] - self.net.remaining_energy[node]) / 0.8
        energy += delta_E
        energy += self.net.distance_matrix[RETURN_TO_BS, node] * MC_MOVING_ENERGY
        if self.mc.energy >= energy:
            return True
        return False
    def print_info(self):
        print(f"energy in moving : {self.mc.energy_in_moving}\nenergy on receing : {self.mc.energy_on_receiving}\nenergy efficiency : {self.mc.energy_on_receiving / (self.mc.energy_in_moving + self.mc.energy_on_receiving / 0.8)}")



if __name__ == '__main__':
    env = Environment('../data/u20.txt')
    # print(env.observation_space.shape)
    print(env.reset())
    print(env.step(12))
    print(env.step(10))