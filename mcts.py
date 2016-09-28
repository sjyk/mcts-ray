import numpy as np
import random
import ray
import copy

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


class GridWorld(object):

	# Constants in the map
	EMPTY, BLOCKED, START, GOAL, PIT, AGENT = range(6)

	#actions
	ACTIONS = np.array([[-1, 0], [+1, 0], [0, -1], [0, +1]])
	actions_num = 4
	GOAL_REWARD = +1
	PIT_REWARD = -1
	STEP_REWARD = -.001

	def __init__(self, gmap, noise=0.1):
		#self.
		self.map = gmap
		self.start_state = np.argwhere(self.map == self.START)[0]
		self.ROWS, self.COLS = np.shape(self.map)
		self.statespace_limits = np.array(
            [[0, self.ROWS - 1], [0, self.COLS - 1]])
		self.NOISE = noise


	def s0(self):
		self.state = self.start_state.copy()
		return self.state, self.isTerminal(), self.possibleActions()

	def isTerminal(self, s=None):
		if s is None:
			s = self.state
		if self.map[s[0], s[1]] == self.GOAL:
			return True
		if self.map[s[0], s[1]] == self.PIT:
			return True
		return False

	def possibleActions(self, s=None):
		if s is None:
			s = self.state
		possibleA = np.array([], np.uint8)
		for a in xrange(self.actions_num):
			ns = s + self.ACTIONS[a]
			if (
                    ns[0] < 0 or ns[0] == self.ROWS or
                    ns[1] < 0 or ns[1] == self.COLS or
                    self.map[int(ns[0]), int(ns[1])] == self.BLOCKED):
				continue
			possibleA = np.append(possibleA, [a])
		return possibleA

	def step(self, a):
		r = self.STEP_REWARD
		ns = self.state.copy()
		if np.random.rand(1,1) < self.NOISE:
            # Random Move
			a = np.random.choice(self.possibleActions())

        # Take action
		ns = self.state + self.ACTIONS[a]

        # Check bounds on state values
		if (ns[0] < 0 or ns[0] == self.ROWS or
			ns[1] < 0 or ns[1] == self.COLS or
			self.map[ns[0], ns[1]] == self.BLOCKED):
			ns = self.state.copy()
		else:
            # If in bounds, update the current state
			self.state = ns.copy()

        # Compute the reward
		if self.map[ns[0], ns[1]] == self.GOAL:
			r = self.GOAL_REWARD
		if self.map[ns[0], ns[1]] == self.PIT:
			r = self.PIT_REWARD

		terminal = self.isTerminal()
		return r, ns, terminal, self.possibleActions()

MAP_NAME = '/Users/sanjayk/Documents/research/RLPy/rlpy/Domains/GridWorldMaps/4x5.txt'
gmap = np.loadtxt(MAP_NAME, dtype=np.uint8)
domain = GridWorld(gmap)

num_workers = 1
ray.init(start_ray_local=True, num_workers=num_workers)

#Plays a single action (remote version)
@ray.remote
def playr(state, action, current_reward):
	g = copy.copy(domain)
	g.s0()
	g.state = state.copy()
	r, ns, terminal, actions = g.step(action)
	return ns, actions, current_reward + r


def play(state, action, current_reward):
	g = copy.copy(domain)
	g.s0()
	g.state = state.copy()
	r, ns, terminal, actions = g.step(action)
	return ns, actions, current_reward + r, terminal


#Plays a single action, gets an expected state
def getNextState(state, action, trials):
	
	resultsa = np.zeros((trials,2))

	results = []

	for _ in range(0, trials):
		results.append(playr.remote(state, action,0))

	#must be a better way to do this
	results = ray.get(results)
	for t in range(0, trials):
		resultsa[t,:] = results[t][0]


	agg_state = np.median(np.array(resultsa), axis=0)

	g = GridWorld(gmap)

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

		cur_state, cur_possible_actions, acc_reward, term = play(cur_state, action, acc_reward)

		#break if terminal
		if term:
			break

	return acc_reward


#performs the tree search
def treeSearch(init_state, 
			   init_actions, 
			   init_reward, 
			   init_time,
			   mctsTree,
			   playout_iters=10,
			   query_iters=10,
			   traversed_set=set()):

	#initialize
	cur_state = init_state
	cur_state.flags.writeable = False
	acc_reward = init_reward
	cur_possible_actions = init_actions
	current_time = init_time

	#base case, no more time remaining
	if current_time == 0:
		return

	#expand each node
	for a in cur_possible_actions:

		#if we have already seen it
		if (cur_state.data, a) in traversed_set:
			continue

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

			new_set = copy.copy(traversed_set)
			new_set.add((cur_state.data, a))

			treeSearch(new_state, 
					   new_actions,
					   expected_reward,
					   current_time-1,
					   subTree,
					   playout_iters,
					   query_iters,
					   new_set)
		else:
			return


g = copy.copy(domain)
s, t, a = g.s0()

m = MCTSTree()

import datetime

now = datetime.datetime.now()
treeSearch(s,a,0,15, m)
print datetime.datetime.now()-now

print m.argmax()


