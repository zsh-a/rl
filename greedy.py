import numpy as np
from env_full import Environment



def main():
    np.random.seed(0)
    env = Environment('data/u20.txt')
    obs = env.reset()
    times = 0
    rewards = 0
    trace = []
    while True:
        action = np.argmin(obs[1:]) + 1
        if env.can_charge_next(action) is False:
            action = 0
        obs,reward,done,info = env.step(action)
        times += info['time']
        if done:
            break
        if len(trace) == 0 or trace[-1] != action:
            trace.append(action)
        rewards += reward
    print(f"rewards : {rewards} run times : {times}")
    print(trace)
    env.print_info()


if __name__ == '__main__':
    main()