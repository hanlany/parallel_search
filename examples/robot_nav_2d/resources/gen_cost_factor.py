import numpy as np
import os
import pdb

dim1 = {'hrt201n': [305, 294], 'den501d': [338, 320], 'den520d': [257, 256], 'brc203d': [391, 274]}
scale1 = 2

dim2 = {'ht_chantry':  [141, 162]}
scale2 = 4


for map_name, dims in dim1.items():
	print("Map: {} | dims: {}".format(map_name, np.multiply(dims,scale1)))
	cost_mat = np.random.uniform(low=1, high=100, size=np.multiply(dims,scale1))
	np.savetxt(os.path.join(map_name, map_name +'_cost_factor.map'), cost_mat, fmt='%d')

for map_name, dims in dim2.items():
	print("Map: {} | dims: {}".format(map_name, np.multiply(dims,scale2)))
	cost_mat = np.random.uniform(low=1, high=100, size=np.multiply(dims,scale2))
	np.savetxt(os.path.join(map_name, map_name +'_cost_factor.map'), cost_mat, fmt='%d')