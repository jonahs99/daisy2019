import numpy as np

import car_configs
import model
import score

TRACK_DIR = '../data/'
tracks = ['track_{}.csv'.format(i) for i in range(1, 11)]

for track_file in tracks:
	track = np.loadtxt(TRACK_DIR + track_file, skiprows=1)

	best = {'time': float('inf')}

	for config in car_configs.gen():
		instr = model.optimize(config, track)
		
		stats = score.score(config, track, instr)
		
		if stats['time'] < best['time']:
			best = stats
			best_instr = instr

	print(track_file)
	print(best)

exit()

import matplotlib.pyplot as plt
#plt.plot(track)
plt.plot(best_instr[:,0])
plt.show()

print(best)	
