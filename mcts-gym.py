import numpy as np
import random
import ray
import copy
import gym

class MCTSTree(object):

	def __init__(self):
		self.children = []
		self.parent = None
		self.state_action = None
		self.reward = -np.inf

	def backpropagate(self,r):
		self.reward = r
		cur_tree = self.parent
		
		while cur_tree != None and r > cur_tree.reward:
			cur_tree.reward = r
			cur_tree = cur_tree.parent

	def treePrint(self):
		cur_tree = self
		while cur_tree.children != []:
			print cur_tree.state_action, cur_tree.reward
			best = np.argmax([c.reward for c in cur_tree.children])
			cur_tree = cur_tree.children[best]

	def argmax(self):
		result = []
		cur_tree = self
		while cur_tree.children != []:
			result.append(cur_tree.state_action)
			best = np.argmax([c.reward for c in cur_tree.children])
			cur_tree = cur_tree.children[best]

		return result


num_workers = 5
ray.init(start_ray_local=True, num_workers=num_workers)

# Function for initializing the gym environment.
def env_initializer():
  return gym.make("Pong-v0")

# Function for reinitializing the gym environment in order to guarantee that
# the state of the game is reset after each remote task.
def env_reinitializer(env):
  env.reset()
  return env

# Create a reusable variable for the gym environment.
ray.reusables.env = ray.Reusable(env_initializer, env_reinitializer)

#randomly plays out from the current state taking an initial action
@ray.remote
def randomPlayout(action_list, remaining_time):

	acc_reward = 0

	env = ray.reusables.env
	print "Inside", action_list

	for a in action_list:
		observation, reward, done, info = env.step(a)
		acc_reward = acc_reward + reward

		#break if terminal
		if done:
			return acc_reward

	for t in range(0, remaining_time):

		observation, reward, done, info = env.step(env.action_space.sample())

		acc_reward = acc_reward + reward

		#break if terminal
		if done:
			return acc_reward

	return acc_reward


#performs the tree search
def treeSearch(mctsTree,
			   action_list,
			   current_time,
			   receding_horizon=10,
			   playout_iters=10,
			   search_list=[]):

	#base case, no more time remaining
	if receding_horizon == 0:
		return

	env = ray.reusables.env

	search_list = []

	#expand each node
	for a in np.random.permutation(env.action_space.n):

		action_arg = copy.copy(action_list)
		action_arg.append(a)
		print "Outside", action_arg

		playouts = []
		for i in range(0,playout_iters):
			playouts.append(randomPlayout.remote(action_arg,current_time))

		rewards = ray.get(playouts)
		expected_reward = np.mean(rewards)
		delta = (np.max(rewards) - np.min(rewards))/np.sqrt(2*len(rewards)) #hoeffding's inequality

		#data structure update
		subTree = MCTSTree()
		mctsTree.children.append(subTree)
		subTree.parent = mctsTree
		subTree.state_action = a
		subTree.backpropagate(expected_reward)

		print expected_reward+delta, action_list

		search_list.append((expected_reward+delta, subTree, action_arg))

	#apply UCT rule
	search_list.sort(reverse=True)
	s = search_list.pop()

	treeSearch(s[1],
			       s[2],
			       current_time-1,
			       receding_horizon -1,
			       playout_iters,
			       search_list)


m = MCTSTree()

import datetime

now = datetime.datetime.now()
treeSearch(m, [], 100)
print datetime.datetime.now()-now

print m.argmax()


