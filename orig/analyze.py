import matplotlib.pyplot as plt
import numpy as np

import model

TRACK_DIR = '../data/'
track_files = ['track_{}.csv'.format(i) for i in range(1, 9)]
tracks = [np.loadtxt(TRACK_DIR + track_file, skiprows=1) for track_file in track_files]

SOLUTION_DIR = '../soln/'

for i in range(8):
	track = tracks[i]
	instr = np.loadtxt(SOLUTION_DIR + 'instructions_{}.csv'.format(i+1), delimiter=',', skiprows=1)
	config = model.make_config(np.loadtxt(SOLUTION_DIR + 'car.csv', delimiter=',', dtype=int, skiprows=1))

	v_max = model.max_speed(config, track)
	v = model.vel(instr[:,0])

	t = np.sum(2 / (v[1:] + v[:-1]))

	print(t, np.sum(instr[:,1]))

	plt.figure()

	plt.plot(v_max)
	plt.plot(v)

	plt.show()
