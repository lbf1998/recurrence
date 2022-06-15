import time
import numpy as np
import make_env

env = make_env.make_env('simple_spread')
obs = env.reset()
print(env.observation_space)
print(env.action_space)

steps = 0
print(steps)
print(obs)

for _ in range(25):
    steps += 1
    print(steps)
    action_n = [np.array([0, 1, 0, 1, 0, 1, 1, 1, 1],dtype=np.float32),
                np.array([0, 10, 0, 0, 0], dtype=np.float32),
                np.array([0, 0, 0, 0, 0], dtype=np.float32),
                np.array([0, 0, 0, 0, 0], dtype=np.float32),
                np.array([0, 0, 0, 0, 0], dtype=np.float32),
                np.array([0, 0, 0, 0, 0], dtype=np.float32)]
    next_obs_n, reward_n, done_n, _ = env.step(action_n)
    print(next_obs_n)
    print(reward_n)
    print(done_n)
    env.render()
    time.sleep(1000)
    if all(done_n):
        break

env.close()