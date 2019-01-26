import math
import numpy as np

TRACK_FILE = "../data/track_1.csv"

ACC = 30 # m/s^2
BRK = -30 # m/s^2
TOP = 30 # m/s
HANDLE = 21

radius = np.loadtxt(TRACK_FILE, skiprows=1)

# Convert to max speeds

v_max = np.sqrt(np.abs(radius) * HANDLE / 1e6)
v_max = np.minimum(v_max, TOP)
v_max[radius == -1] = TOP

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

# Save Instructions

OUTPUT_FILE = "instructions.csv"

outarr = np.zeros((len(radius), 2))
outarr[:,0] = acc

np.savetxt(OUTPUT_FILE, outarr, delimiter=',')

# Plot

exit()

import matplotlib.pyplot as plt

n = len(radius)

plt.subplot(2, 1, 1)

plt.plot(v_max[:n])
plt.plot(speed[:n])

plt.plot(acc[:n], '.-')

plt.subplot(2, 1, 2)

plt.plot(gas[:n])
plt.plot(tire[:n])

plt.show()

