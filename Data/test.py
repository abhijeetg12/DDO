import h5py
import numpy as np 
import matplotlib.pyplot as plt 
filename = 'pose1.h5'
f = h5py.File(filename, 'r')

# List all groups
print("Keys: %s" % f.keys())
a_group_key = list(f.keys())[0]

print (list(f.keys())[0])

# # Get the data
data = list(f[a_group_key])
print (data)
data=f['poseest']['points'].value

data=np.asarray(data)
organ=7
X=data[:,organ,0]
Y=data[:,organ,1]

# print np.shape(X)
plt.scatter(Y[1000:2000], X[1000:2000])

organ=8
X=data[:,organ,0]
Y=data[:,organ,1]
plt.scatter(Y[1000:2000], X[1000:2000],c='r')


organ=9
X=data[:,organ,0]
Y=data[:,organ,1]
plt.scatter(Y[1000:2000], X[1000:2000],c='k')

plt.show()
