import math
import numpy as np

EPS = 1e-6

PIT_PENALTY = 30

# Tier constants

props = {}
props['acceleration'] =	[10,	15,		20,		25,		30	]
props['breaking'] =		[-10,	-15,	-20,	-25,	-30	]
props['speed'] =		[10,	20,		30,		40,		50	]
props['gas'] = 			[500,	750,	1000,	1250,	1500]
props['tire'] =			[500,	750,	1000,	1250,	1500]
props['handling'] =		[9,		12,		15,		18,		21	]

def score(config, track, instr):
	car = {}
	for i, param in enumerate(['tire', 'gas', 'handling', 'speed', 'acceleration', 'breaking']):
		car[param] = props[param][config[i] - 1]

	#instr_vals = np.loadtxt(DIR + 'instructions.csv', delimiter=',', skiprows=1)
	instr_vals = instr
	instr = {}
	instr['acc'] = instr_vals[:,0]
	instr['pit'] = instr_vals[:,1]

	radius = track

	# Simulate

	time = 0
	gas = car['gas']
	tire = car['tire']
	v = 0

	for x in range(len(radius)):
		if abs(v) < EPS and instr['pit'][x]:
			gas = car['gas']
			tire = car['tire']
			time += PIT_PENALTY	

		a = instr['acc'][x]

		if gas < 0:
			print('No gas {}'.format(gas))
			a = min(a, 0)

		discr = v**2 + 2*a

		v_next = 0 if discr < 0 else math.sqrt(v**2 + 2*a)

		delta_t = (v_next - v)/a if abs(a) > EPS else 1 / v

		v_max = math.sqrt((radius[x] * car['handling']) / 1000) if radius[x] >= 0 else float('inf')

		if tire < 0:
			print('Blew out at x={} !'.format(x))
			break

		if a > car['acceleration']:
			print('Acceleration too high')
			print(a - car['acceleration'])
			print(a)

		if v > car['speed']:
			print('Speed too high')
			print(v - car['speed'])
		if v_next > v_max:
			print('Crashed at x={} !'.format(x))
			print(v_next - v_max)
			break

		gas  -= 0.1 * max(a, 0) ** 2
		tire -= 0.1 * min(a, 0) ** 2

		v = v_next
		time += delta_t

	stats = {
		'time': time,
		'gas': gas,
		'tire': tire,
		'config': config
	}

	return stats

