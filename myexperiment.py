#!/usr/bin/env python

from segmentcentroid.envs.GridWorldEnv import GridWorldEnv
from segmentcentroid.tfmodel.GridWorldNNModel import GridWorldNNModel
from segmentcentroid.planner.value_iteration import ValueIterationPlanner
from segmentcentroid.planner.traj_utils import *

import numpy as np
import copy

import tensorflow as tf

# Moving CCW
def inRoom(state):
	row = 8
	col = 8
	if (state[1] <= col and state[0]<=row):
		return 1
	if (state[1] <= col and state[0]>row):
		return 2 
	if (state[1] > col and state[0]>row):
		return 3
	if (state[1] > col and state[0]<=row):
		return 4

#two rooms start end 

def runPolicies(demonstrations=100,
		super_iterations=2000,
		sub_iterations=0,
		learning_rate=1e-2,
		env_noise=0.3):

	m  = GridWorldNNModel(4, statedim=(2,1))

	MAP_NAME = 'resources/GridWorldMaps/myexperiment1.txt'
	gmap = np.loadtxt(MAP_NAME, dtype=np.uint8)
	full_traj = []
	vis_traj = []

	for i in range(0,demonstrations):
		print("Traj",i)
		g = GridWorldEnv(copy.copy(gmap), noise=env_noise)

		g.generateRandomStartGoal() 
		start = np.argwhere(g.map == g.START)[0]
		goal = np.argwhere(g.map == g.GOAL)[0]
		room_start = inRoom(start)
		room_goal = inRoom(goal)
		#generate trajectories start in same room and end different room
		while (room_start==room_goal):
				g.generateRandomStartGoal()
				start = np.argwhere(g.map == g.START)[0]
				goal = np.argwhere(g.map == g.GOAL)[0]
				room_start = inRoom(start)
				room_goal = inRoom(goal)


		print(np.argwhere(g.map == g.START), np.argwhere(g.map == g.GOAL))

		v = ValueIterationPlanner(g)
		traj = v.plan(max_depth=100)
		
		new_traj = []
		for t in traj:
			a = np.zeros(shape=(4,1))

			s = np.zeros(shape=(2,1))

			a[t[1]] = 1

			s[:,0] = t[0]
			#s[2:4,0] = np.argwhere(g.map == g.START)[0]
			#s[4:6,0] = np.argwhere(g.map == g.GOAL)[0]

			new_traj.append((s,a))

		full_traj.append(new_traj)
		vis_traj.extend(new_traj)

	#raise ValueError("")

	#g.visualizePlan(vis_traj,blank=True, filename="resources/results/exp1-trajs.png")


	m.sess.run(tf.initialize_all_variables())

	with tf.variable_scope("optimizer"):
		opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

		m.train(opt, full_traj, super_iterations, sub_iterations)

	actions = np.eye(4)


	g = GridWorldEnv(copy.copy(gmap), noise=0.1)
	g.generateRandomStartGoal()

	for i in range(m.k):
		states = g.getAllStates()
		policy_hash = {}
		trans_hash = {}

		for s in states:

			t = np.zeros(shape=(2,1))

			t[:,0] = s
			#t[2:4,0] = np.argwhere(g.map == g.START)[0]
			#t[4:6,0] = np.argwhere(g.map == g.GOAL)[0]


			l = [ np.ravel(m.evalpi(i, [(t, actions[j,:])] ))  for j in g.possibleActions(s)]

			if len(l) == 0:
				continue

			#print(i, s,l, m.evalpsi(i,ns))
			action = g.possibleActions(s)[np.argmax(l)]

			policy_hash[s] = action

			print("Transition: ",m.evalpsi(i, [(t, actions[1,:])]), t)
			trans_hash[s] = np.ravel(m.evalpsi(i, [(t, actions[1,:])]))

		g.visualizePolicy(policy_hash, trans_hash, blank=True, filename="resources/results/myexp1-policy"+str(i)+".png")

runPolicies()