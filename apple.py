import numpy as np 
actions=np.load("actions_modified.npy")
actions=0.1*actions
actions_1=-1*actions
act_fin = np.concatenate((actions, actions_1))
act_fin_hf=0.5*act_fin
act_fin_s = np.concatenate((act_fin_hf, act_fin_hf))

np.save('actions_modified_neg.npy', act_fin_s)
#print (len(act_fin))