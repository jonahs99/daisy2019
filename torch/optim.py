import torch
import torch.optim as optim
import numpy as np
import math

import matplotlib.pyplot as plt

EPS = 1e-8

def optimize(config, track):
	
	n = len(track)

	v_max = np.minimum(np.sqrt(np.abs(track) * config['handling'] / 1e6), config['speed'])
	v_max[track == -1] = config['speed']
	v_max[1:] = np.minimum(v_max[:-1], v_max[1:])
	v_max[0] = 0

	v_psbl = v_max.copy()

	for i in range(1, n):
		v_psbl[i] = min(
			math.sqrt(v_psbl[i-1] ** 2 + 2 * config['acceleration']*(1-EPS) ), v_psbl[i])
	for i in range(n - 2, -1, -1):
		v_psbl[i] = min(
			math.sqrt(v_psbl[i+1] ** 2 - 2 * config['breaking']*(1-EPS) ), v_psbl[i])

	v_psbl = torch.from_numpy(v_psbl)

	v_scl = torch.ones(v_psbl.size(), dtype=torch.double) * 0.5
	v_scl.requires_grad_(True)

	optimizer = optim.SGD([v_scl], lr=0.0004)

	times = []

	for i in range(20000):
		optimizer.zero_grad()
	
		v_scaled = v_psbl * torch.clamp(v_scl, min=0, max=1)

		acc = accel(v_scaled)
		gas = torch.sum(torch.clamp(acc, min=0)**2 * 0.1)
		tire = torch.sum(torch.clamp(acc, max=0)**2 * 0.1)

		#print(gas.data, tire.data)

		c = torch.clamp(
			torch.min(torch.sqrt(config['gas'] / gas), torch.sqrt(config['tire'] / tire)),
			min=0, max=1)

		v = v_scaled * c

		t = time(v)
		t.backward()
	
		optimizer.step()

		times.append(t.item())
	
		#print('time {}'.format(t.data))

		#if (i % 100 == 0):
		#	plt.plot(v_scaled.detach().numpy() * c.detach().numpy())

	plt.show()
	plt.plot(times)
	plt.show()

	instr = accel(v.detach()).numpy()
	

	return v_max, v_psbl.numpy(), v, instr, t.item()

def accel(v):
	return (v[1:]**2 - v[:-1]**2) / 2

def time(v):
	return torch.sum(2 / (v[1:] + v[:-1]))

# Testing

config = {
	'tire': 1250,
	'gas': 1500,
	'handling': 21,
	'speed': 20,
	'acceleration': 10,
	'breaking': -10,
}

track_files = ['../data/track_{}.csv'.format(i) for i in range(1,9)]
track_w = [1, 0.25, 0.25, 0.25, 0.5, 0.5, 1, 1]

weighted_time = 0

for i, (track_file, w) in enumerate(zip(track_files, track_w)):
	track = np.loadtxt(track_file, skiprows=1)

	v_max, v_psbl, v, acc, t = optimize(config, track)

	instr = np.zeros((len(track), 2))
	instr[:-1,0] = acc

	np.savetxt('../torch-soln/instructions_{}.csv'.format(i+1), instr, header='a,pit_stop', comments='', fmt='%f', delimiter=',')

	print(track_file, t)

	weighted_time += w * t

print(weighted_time)
