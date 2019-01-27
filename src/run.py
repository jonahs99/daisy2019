import numpy as np

import car_configs
import model
import score

TRACK_DIR = '../data/'
track_files = ['track_{}.csv'.format(i) for i in range(1, 9)]
tracks = [np.loadtxt(TRACK_DIR + track_file, skiprows=1) for track_file in track_files]
tracks_w = [1, 0.25, 0.25, 0.25, 0.5, 0.5, 1, 1]

SOLUTION_DIR = '../soln/'

configs = car_configs.gen()
configs = [[4, 5, 5, 2, 1, 1]]

n_configs = len(configs)
print('Evaluating {} car configs...'.format(n_configs))

best_time = float('inf')
for i, config in enumerate(configs[:]):
	all_instr = []

	total_time = 0
	for track, track_w in zip(tracks, tracks_w):
		instr, time, gas, tire, v = model.optimize(config, track)
		total_time += track_w * time
		all_instr.append(instr)

	if total_time < best_time:
		best_time = total_time
		best_config = config
		best_instr = all_instr

	print(config, total_time)

	if ((i+1) % 10 == 0):
		print('{}/{} done.'.format(i+1, n_configs))
		print('Best: {} with time {}'.format(best_config, best_time))

	#print(config)
	#print(total_time)

print(best_config)
print(best_time)
print([np.sum(instr[:,1]) for instr in best_instr])

for i, instr in enumerate(best_instr):
	np.savetxt(SOLUTION_DIR + 'instructions_{}.csv'.format(i+1), instr, delimiter=',', fmt='%f',
		header='a,pit_stop', comments='') 


np.savetxt(SOLUTION_DIR + 'car.csv', (best_config,), delimiter=',', fmt='%d',
	header='tire,gas,handling,speed,acceleration,breaking', comments='')

exit()

import matplotlib.pyplot as plt
#plt.plot(track)
plt.plot(best_instr[:,0])
plt.show()

print(best)	
