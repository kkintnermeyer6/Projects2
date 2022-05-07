import gym
import numpy as np
import random

env = gym.make('MountainCar-v0')

print(env.observation_space.low)
print(env.observation_space.high)

A = np.array([10,100])

stateSize = (env.observation_space.high - env.observation_space.low) * A
stateSize = np.round(stateSize,0).astype(int) + 1
actionSize = env.action_space.n

qtable = np.zeros((stateSize[0],stateSize[1],actionSize))
learningRate = 0.7
discountRate = 0.9
epsilon = 0.8
decay_rate = 0.1

numSteps = 2500
numEpisodes = 5000


for episode in range(numEpisodes):

	state = env.reset()
	done = False

	for _ in range(numSteps):
	
		state_adj = (state - env.observation_space.low) * A
		state_adj = np.round(state_adj,0).astype(int)
		
		if episode >= numEpisodes - 5:
			epsilon = 0.0
			env.render()
		
		if random.uniform(0,1) < epsilon:
			action = env.action_space.sample()
		else:
			action = np.argmax(qtable[state_adj[0],state_adj[1],:])

		newState, reward, done, info = env.step(action)

		state2_adj = (newState - env.observation_space.low) * A
		state2_adj = np.round(state2_adj,0).astype(int)		
		
		qtable[state_adj[0],state_adj[1],action] = qtable[state_adj[0],state_adj[1],action] + learningRate * (reward + discountRate*np.max(qtable[state2_adj[0],state2_adj[1],:])-qtable[state_adj[0],state_adj[1],action])
		
		state = newState
		
		if done == True:
			break

	epsilon = np.exp(-decay_rate*episode)



env.close()
