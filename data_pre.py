import numpy as np

import h5py
import ffmpeg

def get_data():
############# Importing the data, the length of the data from pose1 file is ? ###############
	filename = 'Data/pose1.h5'
	filename2= 'Data/pose2.h5'
	filename3= 'Data/pose3.h5'
	filename4= 'Data/pose4.h5'
	f = h5py.File(filename, 'r')
	f2= h5py.File(filename2, 'r')
	f3= h5py.File(filename3, 'r')
	f4= h5py.File(filename4, 'r')
	# List all groups
	#print("Keys: %s" % f.keys())
	a_group_key = list(f.keys())[0]

	#print (list(f.keys())[0])

	# # Get the data
	data = list(f[a_group_key])
	data2 = list(f2[a_group_key])
	data3 = list(f3[a_group_key])
	data4 = list(f4[a_group_key])
	#print (data)
	data=f['poseest']['points'].value
	data2=f2['poseest']['points'].value
	data3=f3['poseest']['points'].value
	data4=f4['poseest']['points'].value

	data=data.astype('float64')
	data2= data2.astype('float64')
	data3=data3.astype('float64')
	data4= data4.astype('float64')

	data=data[20000:]
	data=np.vstack((data,data2[2000:]))
	data=np.vstack((data,data3[2000:]))
	data=np.vstack((data,data4[2000:]))
	#data=data[5000:]
	l=len(data)
	print ('The length of the data is ', l)
	return data

