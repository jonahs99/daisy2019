import math
import numpy as np


EPS = 1e-6

props = {}
props['acceleration'] =	[10,	15,		20,		25,		30	]
props['breaking'] =		[-10,	-15,	-20,	-25,	-30	]
props['speed'] =		[10,	20,		30,		40,		50	]
props['gas'] = 			[500,	750,	1000,	1250,	1500]
props['tire'] =			[500,	750,	1000,	1250,	1500]
props['handling'] =		[9,		12,		15,		18,		21	]

def optimize(config, track):
	TIRE = props['tire'][config[0]-1]
	GAS = props['gas'][config[1]-1]
	HANDLE = props['handling'][config[2]-1] 
	TOP = props['speed'][config[3]-1]
	ACC = props['acceleration'][config[4]-1] 
	BRK = props['breaking'][config[5]-1] 

	#radius = np.loadtxt(TRACK_FILE, skiprows=1)
	radius = track

	# Convert to max speeds

	v_max = np.sqrt(np.abs(radius) * HANDLE / 1e6)
	v_max = np.minimum(v_max, TOP)
	v_max[radius == -1] = TOP

	v_max[1:] = np.minimum(v_max[:-1], v_max[1:])

	v_max[0] = 0

	# Optimal travel

	speed = np.zeros(v_max.shape)

	for i in range(1, len(speed)):
		speed[i] = min(math.sqrt( speed[i-1] ** 2 + 2 * ACC ), v_max[i])

	for i in range(len(speed) - 2, -1, -1):
		speed[i] = min(math.sqrt( speed[i+1] ** 2 - 2 * BRK ), speed[i])

	acc = (speed[1:]**2 - speed[:-1]**2) / 2
	acc = np.append(acc, [0])

	# Costs

	gas = np.cumsum((0.1*np.maximum(acc, 0)**2))
	tire = np.cumsum((0.1*np.minimum(acc, 0)**2))

	# Scale to save gas money

	c_gas = math.sqrt(GAS / gas[-1])
	c_tire = math.sqrt(TIRE / tire[-1])
	c = min(c_gas, c_tire, 1)
	acc *= c
	acc *= (1-EPS)

	instr = np.zeros((len(acc), 2))
	instr[:,0] = acc

	return instr

'''
	# Save Instructions

	OUTPUT_PATH = "soln/"

	outarr = np.zeros((len(radius), 2))
	outarr[:,0] = acc

	np.savetxt(OUTPUT_PATH + '/instructions.csv', outarr, delimiter=',', header='a,pit_stop')

	#np.savetxt(OUTPUT_PATH + '/car.csv', ([TIRE, GAS, HANDLE, TOP, ACC, BRK],), delimiter=',',
	#	header='tire,gas,handling,speed,acceleration,breaking')

	# Plot
	import matplotlib.pyplot as plt

	n = len(radius)
	n = 55

	plt.subplot(2, 1, 1)

	plt.plot(v_max[:n])
	plt.plot(speed[:n])

	plt.plot(acc[:n], '.-')

	plt.subplot(2, 1, 2)

	plt.plot(gas[:n])
	plt.plot(tire[:n])

	plt.show()
'''
