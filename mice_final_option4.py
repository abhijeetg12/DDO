#!/usr/bin/env python
'''
Changes to be made in this file: 
-- Fix the matplotlib visualisation error 
-- How to put options in the pdf file? 
--Including the option frame in the graph? 
--Colour coordination
-- Center of gravit of the options to be fixed. 

'''

#from segmentcentroid.envs.MiceEnv import GridWorldEnv
from segmentcentroid.tfmodel.MiceModel import GridWorldModel
# from segmentcentroid.planner.value_iteration import ValueIterationPlanner
# from segmentcentroid.planner.traj_utils import *
'''
mice_final_option2.py also includes the code for saving options.
Moreover this code attempts to bring the hierarchical policies as the base policies for the actions sets as well.  
This code includes the actions angles in the part of the code. 
NOSE_INDEX = 0
LEFT_EAR_INDEX = 1
RIGHT_EAR_INDEX = 2
BASE_NECK_INDEX = 3
LEFT_FRONT_PAW_INDEX = 4
RIGHT_FRONT_PAW_INDEX = 5
CENTER_SPINE_INDEX = 6
LEFT_REAR_PAW_INDEX = 7
RIGHT_REAR_PAW_INDEX = 8
BASE_TAIL_INDEX = 9
MID_TAIL_INDEX = 10
TIP_TAIL_INDEX = 11

The connections to be made in the visualisation are :
Tail Connections: 11-10-9
Nose to spine: 0-3-6
Arms connections: 4-6-5 , 7-6
angles calculator 

VISUALISATION CONNECTIONS: 
0-3-4-3-5-3-6-7-6-8-6-9-10-11
'''
import numpy as np
import copy
import pdb
import time 

import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
import h5py
import ffmpeg

np.random.seed(0)

############# Importing the data, the length of the data from pose1 file is ? ###############
filename = 'Data/pose1.h5'
filename2= 'Data/pose2.h5'
f = h5py.File(filename, 'r')
f2= h5py.File(filename2, 'r')
	
# List all groups
#print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

#print (list(f.keys())[0])

# # Get the data
data = list(f[a_group_key])
data2 = list(f2[a_group_key])
#print (data)
data=f['poseest']['points'].value
data2=f2['poseest']['points'].value

data=data.astype('float64')
data2= data2.astype('float64')
data=data[20000:]
data=np.vstack((data,data2[2000:]))
data=data[:5000]
l=len(data)



############### centralising the data #################
data_0=data
#mean_data=data[6]#np.mean(data, axis=1)
#print (mean_data)
for i in range(len(data)):
    #data[i,:,0]-=data[i,6]
    #data[i,:,1]-=data[i,6]
	data[i,:]-=data[i,6]

################ Generating action data ################
#### 0: up 1: left 2: down 3:right



'''
Modifying the action dimension here, from (l,4,1) to (l,12,1)
COM_action is acting as the action here. 
Action definition has to be adjusted to a weighted difference between subsequent moves
a(t)=alpha*(s(t+1)-s(t))+beta*(s(t+2)-s(t+1)+...)
Let's put a 10 moves buffer to maintain dimensions in the loop 
'''

l=len(data)
COM_action=np.zeros((l,12,2))
COM_action=COM_action.astype('float64')
#print (np.shape(COM_action))
for i in range(l-100):
    # distance = mean_data[i+1]-mean_data[i]
    # print (distance[0], distance[1])
    # if (np.abs(distance[0])>=np.abs(distance[1])):
    #     if (distance[0]>0):
    #         COM_action[i,3,0]=1
    #     else:
    #         COM_action[i,1,0]=1
    # else:
    #     if (distance[1]>0):
    #         COM_action[i,0,0]=1
    #     else:
    #         COM_action[i,2,0]=1
    COM_action[i]=data[i+1]-data[i]
    #print(COM_action[i], "COM action")
    #print (np.shape(COM_action[i])) (12,2)
    
print ("\n Done normalizing data, and actions generated \n")

'''
We need to divide up the trajectories in sets. 
'''
full_traj=[]
#the full trajectory imeension will be set at 10,10,2,12,2 
episode_len=10
no_episode=10
for j in range(0,no_episode):
	episode=[]
	for i in range(episode_len):
		#instant=[]
		instant=((data[j*episode_len+i], COM_action[j*episode_len+i].flatten('F')))
		#instant.append(COM_action[i])
		episode.append(instant)
	full_traj.append(episode)

'''actions are array([[-0.25,  1.25],
       [-0.25,  0.25],
       [-0.25,  0.25],
       [-0.25,  1.25],
       [-1.25,  1.25],
       [ 1.75, -0.75],
       [-0.25, -0.75],
       [-0.25, -0.75],
       [-0.25, -0.75],
       [ 0.75, -0.75],
       [-0.25,  0.25],
       [ 0.75, -0.75]]))]] whereas it is discrete array([1.],[0],[0],[0]) for the other experiment1.py code
'''
#print (np.shape(full_traj),full_traj)

demonstrations=10
super_iterations=20000#3000#10000
sub_iterations=0
learning_rate=10


#k=4 in  this case. number of primitive options
m  = GridWorldModel(4, statedim=(12,2))
m.sess.run(tf.initialize_all_variables())

with tf.variable_scope("optimizer"):
	opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	#define he optimizer,  put the full trajectorty, 1000, 0 
	closs,tloss=m.train(opt, full_traj, super_iterations, sub_iterations)
	np.save('options/closs', closs)
	np.save('options/tloss', tloss)
	print (closs,len(closs) ,'this is closs')
	plt.plot(range(len(closs)),closs)
	plt.savefig('options/closs.png')
	plt.plot(range(len(tloss)),tloss)
	plt.savefig('options/tloss.png')


'''So how do we generate the visualised options?
We can look at a state, and then apply the  respective options policy from that state.
so how is this done for the gridworld data? It just computes the max of action probabilities over the entire gridworld. 
Instead of doing that we need to provide the same option policy over continues states until it actally terminates. 
How do we do this? 

1. Find a few good states in the state space. 
2. Iterate over the numerb of options and do the sam ething as before
3. till the termination poliy is reached iterate of v evalpi for the same state space
4. Evalpi will need modifications for the same. 
5. But what are the possible action for each state? all actions are possible.
6. What is the termination policy? 

#actiondim does no match
'''
actions = np.eye(12)

policy_hash = {}
trans_hash = {}
len_option=10
init_state=1050
state=data[init_state] #1000


#Generating random actions that decide which state to take. 
#This way the length of th eoptions will also be elongated. 
'''
Options generation
The action dimension is (12,2)
'''
action_set=[]
for a in range(12):
	for b in range(2):
		action=np.zeros((12,2))
		action[a][b]=1

		action_set.append(action)

for k in range(100):
	action_set.append(np.random.randint(2, size=(12,2)))
print (action_set, "Action set")
print (np.shape(action_set), 'action_Set')
action_set=np.load("actions_modified_neg.npy")


Option_viz=[]
for i in range(m.k):
	opn_i=[]

	for j in range(len_option):

	
		print (i, 'is the option evaluation')
		#print (state)
		#l=[np.ravel(m.evalpi(i,[(state, data[1000+k])])) for k in [0,1,2,3] ]

		#l=[np.ravel(m.evalpi(i,[(state,actions[k,:])])) for k in [0,1,2,3]]
		l=[np.ravel(m.evalpi(i,[(state, action_set[k])])) for k in range(len(action_set)) ]
		print (l)
		action = [np.argmax(l)]
		print (action)
		state = state+.1*action_set[int(action[0])] # reducing the impace of one action in mouse movement
		state -= state[6] #Normalising the state of the mouse

		print (state)

		opn_i.append(state)

	Option_viz.append(opn_i)
	state=data[init_state]
#Now the only thing is left is to discretize action and visualize
print (np.shape(Option_viz), "This is the option viz dimension")
#### Visualization Procedure #####
##################################
#plt([datai[0,0],datai[0,1]],[datai[1,0], datai[1,1]])
def _update_plot(i, fig, scat):
    datai=Option_viz[0][i]#data[i]  [:]
    print (np.shape(datai), 'shape of datai')
    scat.set_offsets(datai)
    #plt.plot(datai[(0,3,4,3,5,3,6,7,6,8,6,9,10,11),0], datai[(0,3,4,3,5,3,6,7,6,8,6,9,10,11),1], 'ro-')#[datai[0,0],datai[0,1]],[datai[1,0],data[1,1]])

    print('Frames: %d' %i)

    return scat,

def _update_plot1(i, fig, scat):
    datai=Option_viz[1][i]#data[i][:]
    scat.set_offsets(datai)#(([0, i],[50, i],[100, i]))
    #plt.plot(datai[(0,3,4,3,5,3,6,7,6,8,6,9,10,11),0], datai[(0,3,4,3,5,3,6,7,6,8,6,9,10,11),1], 'ro-')
    print('Frames: %d' %i)

    return scat,


def _update_plot2(i, fig, scat):
    datai=Option_viz[2][i]#data[i][:]
    scat.set_offsets(datai)#(([0, i],[50, i],[100, i]))
    print('Frames: %d' %i)

    return scat,


def _update_plot3(i, fig, scat):
    datai=Option_viz[3][i]#data[i][:]
    scat.set_offsets(datai)#(([0, i],[50, i],[100, i]))
    print('Frames: %d' %i)

    return scat,


fig =  plt.figure()                

x = [0, 50, 100]
y = [0, 0, 0]

ax = fig.add_subplot(111)
ax.grid(True, linestyle = '-', color = '0.75')
ax.set_xlim([-50,50])
ax.set_ylim([-50,50])  

scat = plt.scatter(x, y)
scat.set_alpha(0.8)

Writer = animation.writers['ffmpeg']
writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

anim = animation.FuncAnimation(fig, _update_plot, fargs = (fig, scat),frames = len_option, interval = 50)
anim.save('options/option1_4.mp4', writer=writer)

anim = animation.FuncAnimation(fig, _update_plot1, fargs = (fig, scat),frames = len_option, interval = 50)
anim.save('options/option2_4.mp4', writer=writer)

anim = animation.FuncAnimation(fig, _update_plot2, fargs = (fig, scat),frames = len_option, interval = 50)
anim.save('options/option3_4.mp4', writer=writer)

anim = animation.FuncAnimation(fig, _update_plot3, fargs = (fig, scat),frames = len_option, interval = 50)
anim.save('options/option4_4.mp4', writer=writer)

#plt.show()  


# anim = animation.FuncAnimation(fig, _update_plot1, fargs = (fig, scat),frames = len_option, interval = 50)
# plt.show()  


#code for visualising the generated output trajectories

	# actions = np.eye(4)

############ This part of the code is only for visualization. 

	# g = GridWorldEnv(copy.copy(gmap), noise=0.0)
	# #np.savetxt( 'arrays_storage/g.txt',g,fmt='%s')
	
	# g.generateRandomStartGoal()

	# for i in range(m.k):
	# 	states = g.getAllStates()
	# 	np.savetxt( 'arrays_storage/states.txt',states,fmt='%s')

	# 	policy_hash = {}
	# 	trans_hash = {}

	# 	for s in states: # looping through all the states. 

	# 		t = np.zeros(shape=(8,9))

	# 		t[s[0],s[1]] = 1
	# 		#t[2:4,0] = np.argwhere(g.map == g.START)[0]
	# 		#t[4:6,0] = np.argwhere(g.map == g.GOAL)[0]

	# 		#np.ravel returns the elements of the combined set of elements. 
	# 		l = [ np.ravel(m.evalpi(i, [(t, actions[j,:])] ))  for j in g.possibleActions(s)]
	# 		#np.savetxt( 'arrays_storage/l.txt',l,fmt='%s')
 
	# 		if len(l) == 0:
	# 			continue

	# 		#print(i, s,l, m.evalpsi(i,ns))
	# 		action = g.possibleActions(s)[np.argmax(l)]

	# 		policy_hash[s] = action

	# 		#print("Transition: ",m.evalpsi(i, [(t, actions[1,:])]), t)
	# 		trans_hash[s] = np.ravel(m.evalpsi(i, [(t, actions[1,:])]))
	# 	#np.savetxt( 'arrays_storage/policy_hash.txt',policy_hash,fmt='%s')
	# 	#np.savetxt( 'arrays_storage/trans_hash.txt',trans_hash,fmt='%s')

	# 	g.visualizePolicy(policy_hash, trans_hash, blank=True, filename="resources/results/exp1-policy"+str(i)+".png")

#runPolicies()

'''
Current error in the code is 
ValueError: Cannot feed value of shape (10, 12, 2) for Tensor 'Placeholder_1:0', which has shape '(?, 4)'
What is the placeholder? 
This stems from, 
m.train(opt, full_traj, super_iterations, sub_iterations)
This is a batch sampling error.
Looking at the sampling function, 
 in sampleBatch
    weights = self.fb.fit([trajectory])
forwardbackward algorithm is exectued from the samplebatch function. 

self.init_iter(i, traj)
self.pi[:,h] = np.clip(self.model.evalpi(h,X),1e-6,1)

'''

# Logs
'''
1) 
Number of actions:100
Lenght of actions: 30 
Alpha (factor of action to state): 0.01
super iterations: 50000
init_state=1000, 1050


'''