import math
import numpy as np


EPS = 1e-6


def make_config(config):	
	props = {}
	props['acceleration'] =	[10,	15,		20,		25,		30	]
	props['breaking'] =		[-10,	-15,	-20,	-25,	-30	]
	props['speed'] =		[10,	20,		30,		40,		50	]
	props['gas'] = 			[500,	750,	1000,	1250,	1500]
	props['tire'] =			[500,	750,	1000,	1250,	1500]
	props['handling'] =		[9,		12,		15,		18,		21	]
	return {prop: props[prop][config[i]-1] for i, prop in
		enumerate(['tire', 'gas', 'handling', 'speed', 'acceleration', 'breaking'])}

def max_speed(config, track):	
	v_max = np.sqrt(np.abs(track) * config['handling'] / 1e6)
	v_max = np.minimum(v_max, config['speed'])
	v_max[track == -1] = config['speed']
	v_max[1:] = np.minimum(v_max[:-1], v_max[1:])

	v_max[0] = 0

	return v_max

def optimize(config, track):
	config = make_config(config)

	# Convert to max speeds

	v_max = max_speed(config, track)
	
	# Prep

	unconstr_v = v_max.copy()
	unconstrain(config, unconstr_v) 

	# Try pitstops

	stats = []
	for n_pits in range(10):
		acc, pit_locs = opt_with_pits(config, unconstr_v, n_pits)
		acc *= (1-EPS)

		v = np.zeros(acc.shape)
		for i, a in enumerate(acc[:-1]):
			discr = v[i]**2 + 2 * a
			v[i+1] = math.sqrt(discr) if discr > 0 else 0

		time = 0
		for i, a in enumerate(acc[:-1]):
			time += ((v[i+1] - v[i]) / a) if abs(a) > EPS else 1 / v[i]

		time += 30 * n_pits
	
		#for i, s in enumerate(v[490:510]):
		#	print(i+490, s)
	
		gas = np.sum((0.1*np.maximum(acc, 0)**2))
		tire = np.sum((0.1*np.minimum(acc, 0)**2))

		instr = np.zeros((len(acc), 2))
		instr[:,0] = acc
		instr[:,1][pit_locs] = 1

		if len(stats) and time >= stats[-1][1]:
			break

		stats.append( (instr, time, gas, tire, v) )

	return stats[-1]

def unconstrain(config, v):
	for i in range(1, len(v)):
		v[i] = min(math.sqrt( v[i-1] ** 2 + 2 * config['acceleration']*(1-EPS) ), v[i])
	for i in range(len(v) - 2, -1, -1):
		v[i] = min(math.sqrt( v[i+1] ** 2 - 2 * config['breaking']*(1-EPS) ), v[i])

def accel(v):
	a = np.zeros(v.shape)
	a[:-1] = (v[1:]**2 - v[:-1]**2) / 2
	return a

def vel(acc):	
	v = np.zeros(acc.shape)
	for i, a in enumerate(acc[:-1]):
		discr = v[i]**2 + 2 * a
		v[i+1] = math.sqrt(discr) if discr > 0 else 0
	return v

def opt_with_pits(config, unconstr_v, n_pits):
	l = len(unconstr_v)

	# compute pit locations
	acc = accel(unconstr_v)
	
	gas_cum = np.cumsum((0.1*np.maximum(acc, 0)**2))
	tire_cum = np.cumsum((0.1*np.minimum(acc, 0)**2)) 

	timeline = (gas_cum + tire_cum) / (gas_cum[-1] + tire_cum[-1])

	#print(timeline[-100:])

	pit_locs = [ np.argmax( timeline >= i / (n_pits+1) ) for i in range(n_pits + 1) ]
	pit_locs.append(l) 
	pit_locs = np.array(pit_locs)

	#print(pit_locs)

	#pit_locs = np.append(np.arange(0, l, l // (n_pits+1)), [l])[:n_pits+2]
	#pit_locs[-1] = l

	#print(pit_locs)
	
	v = unconstr_v.copy() # entire track
	v[pit_locs[1:-1]] = 0

	unconstrain(config, v)

	acc = accel(v)

	for start, end in zip(pit_locs[:-1], pit_locs[1:]):
		reduce(config, acc[start:end], v[start:end])

	acc[pit_locs[1:-1] - 1] = config['breaking']
	
	return acc, pit_locs[1:-1]

def pos(a):
	return np.maximum(a, 0)

def neg(a):
	return np.minimum(a, 0)

def reduce(config, acc, v):	

	import matplotlib.pyplot as plt
		
	count = 0
	gas_arr, tire_arr = [], []

	shape = v.shape
	v = np.append(v, [0])

	lr = 0.001

	while True:
		gas = np.sum((0.1*np.maximum(acc, 0)**2))
		tire = np.sum((0.1*np.minimum(acc, 0)**2)) 
	
		if (gas < config['gas'] and tire < config['tire']):
			break

		v[-1] = v[-2]

		dgas, dtire, dt = np.zeros(shape), np.zeros(shape), np.zeros(shape)
		dgas[1:] = 0.001 * v[1:-1] * ( -pos(v[2:]**2 - v[1:-1]**2) + pos(v[1:-1]**2 - v[:-2]**2) )
		dtire[1:] = 0.001 * v[1:-1] * ( -neg(v[2:]**2 - v[1:-1]**2) + neg(v[1:-1]**2 - v[:-2]**2) )	

		dt[1:] = np.minimum(-2/(v[:-2] + v[1:-1])**2 - 2/(v[1:-1] + v[2:])**2, -0.1) # dt/dv
		dt[0] = float('inf')	
	
		if (tire - config['tire'] > gas - config['gas']):
			shift_t = (-dtire / dt) * 0.8
		else:
			shift_t = -dgas / dt * 0.8
		#shift_t = -dtire/dt + -dgas/dt	
	
		#plt.plot(dgas)
		#plt.plot(dtire)
		##plt.plot(dt)
		#plt.plot(v)
		#plt.plot(neg(shift_t/dt))
		#plt.show()
		#assert(False)
	
		shift_v = neg(shift_t / dt)
		shift_v[shift_v > np.min(shift_t) * 0.6] = 0
		v[:-1] += lr * shift_v
		v[:-1] = pos(v[:-1])

		np.copyto(acc, accel(v[:-1]))	
	
		if count > 20000:
			#print(gas)
			plt.plot(gas_arr)
			plt.plot(tire_arr)
			plt.plot([0, len(gas_arr)], [config['gas']] * 2)
			plt.plot([0, len(gas_arr)], [config['tire']] * 2)
			plt.show()

		gas_arr.append(gas)
		tire_arr.append(tire)
		#plt.plot(v)
		count += 1
	'''
	while(
		np.sum((0.1*np.maximum(acc, 0)**2)) > config['gas'] or
		np.sum((0.1*np.minimum(acc, 0)**2)) > config['tire']):
		
		gas_arr.append(np.sum((0.1*np.maximum(acc, 0)**2)))

		#assert(count < 100)
		if count > 10000:
			plt.plot(gas_arr)
			plt.plot([0, 1000], [config['gas']] * 2)
			plt.show()
		
		#c = v/config['speed']
		c = np.ones(v.shape)
		v[1:-1] = np.minimum(c[1:-1] * np.sqrt( (v[:-2]**2 + v[2:]**2) / 2 + (1-c[1:-1]) * v[1:-1] ), v[1:-1])
		v[-1] = min(c[-1] * math.sqrt((v[-2]**2 + v[-1]**2) / 2) + (1-c[-1])*v[-1], v[-1]) if v[-1] > 0 else 0
	
		np.copyto(acc, accel(v))

		#plt.plot(v)

		count += 1
	'''
	#print(acc, gas)
	#assert(gas > 0)
	#assert(tire > 0)
	
#	c = min(math.sqrt(config['gas'] / gas), math.sqrt(config['tire'] / tire), 1)

#	acc *= c
'''
exit()

# Optimal travel

speed = np.zeros(v_max.shape)

for i in range(1, len(speed)):
	speed[i] = min(math.sqrt( speed[i-1] ** 2 + 2 * ACC ), v_max[i])

for i in range(len(speed) - 2, -1, -1):
	speed[i] = min(math.sqrt( speed[i+1] ** 2 - 2 * BRK ), speed[i])

acc = (speed[1:]**2 - speed[:-1]**2) / 2
acc = np.append(acc, [0])

assert(acc.shape == speed.shape)

# Costs

gas = np.cumsum((0.1*np.maximum(acc, 0)**2))
tire = np.cumsum((0.1*np.minimum(acc, 0)**2))

# Scale to save gas money

c_gas = math.sqrt(GAS / gas[-1])
c_tire = math.sqrt(TIRE / tire[-1])
c = min(c_gas, c_tire, 1) * (1 - EPS)
acc *= c

speed[0] = 0
for i, a in enumerate(acc[:-1]):
	speed[i+1] = math.sqrt(speed[i]**2 + 2*a)

# Compute stats

time = 0
for v, v_next, a in zip(speed[:-1], speed[1:], acc[:-1]):
	time += ((v_next - v) / a) if abs(a) > EPS else 1 / v

gas = np.cumsum((0.1*np.maximum(acc, 0)**2))
tire = np.cumsum((0.1*np.minimum(acc, 0)**2))

instr = np.zeros((len(acc), 2))
instr[:,0] = acc

return instr, time, gas, tire

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
