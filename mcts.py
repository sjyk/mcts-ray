import numpy as np
import random
import ray
import copy

from rlpy.Domains.GridWorld import GridWorld

MAP_FOLDER = '/Users/sanjayk/Documents/research/RLPy/rlpy/Domains/GridWorldMaps/'


num_workers = 5
ray.init(start_ray_local=True, num_workers=num_workers)

#Plays a single action
def play(state, action, current_reward):
	g = GridWorld(mapname=MAP_FOLDER+'4x5.txt')
	g.s0()
	g.state = state.copy()
	r, ns, terminal, actions = g.step(action)
	return ns, actions, current_reward + r


#Plays a single action, gets an expected state
def getNextState(state, action, trials):
	
	results = np.zeros((trials,2))

	for t in range(0, trials):
		g = GridWorld(mapname=MAP_FOLDER+'4x5.txt')
		g.s0()
		g.state = state.copy()
		r, ns, terminal, actions = g.step(action)
		results[t,:] = ns

	agg_state = np.median(results, axis=0)

	g = GridWorld(mapname=MAP_FOLDER+'4x5.txt')

	return agg_state, g.isTerminal(agg_state), g.possibleActions(agg_state)
	#return np.mean(results)


#randomly plays out from the current state taking an initial action
@ray.remote
def randomPlayout(state, action, current_reward, remaining_time):

	cur_state = state
	acc_reward = current_reward
	cur_possible_actions = np.array([action])

	for t in range(0, remaining_time):

		action = np.random.choice(cur_possible_actions)

		cur_state, cur_possible_actions, acc_reward = play(cur_state, action, acc_reward)

	return acc_reward


#performs the tree search
def treeSearch(init_state, 
			   init_actions, 
			   init_reward, 
			   init_time,
			   mctsTree,
			   playout_iters=10,
			   query_iters=10):

	#initialize
	cur_state = init_state
	acc_reward = init_reward
	cur_possible_actions = init_actions
	current_time = init_time

	#base case, no more time remaining
	if current_time == 0:
		return

	#expand each node
	for a in cur_possible_actions:

		playouts = []
		for i in range(0,playout_iters):
			playouts.append(randomPlayout.remote(cur_state,a,acc_reward,current_time))

		expected_reward = np.mean(ray.get(playouts))

		#data structure update
		subTree = MCTSTree()
		mctsTree.children.append(subTree)
		subTree.parent = mctsTree
		subTree.state_action = (init_state, a)
		subTree.backpropagate(expected_reward)

		print current_time, cur_state, a, expected_reward

		
		new_state, terminal, new_actions = getNextState(cur_state, a, query_iters)

		#recurse
		if not terminal:
			treeSearch(new_state, 
					   new_actions,
					   expected_reward,
					   current_time-1,
					   subTree,
					   playout_iters,
					   query_iters)
		else:
			return



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
			print cur_tree.children
			cur_tree = cur_tree.children[best]


g = GridWorld(mapname=MAP_FOLDER+'4x5.txt')
s, t, a = g.s0()

m = MCTSTree()
treeSearch(s,a,0,15, m)
m.treePrint()


