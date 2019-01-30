# Winning solution for the 2019 Daisy Intelligence Hackathon

## About the code

The original solution submitted for the hackathon is contained in the `orig` folder. A cleaner version using PyTorch written after the hackathon is in the `torch` folder. This is the solution I describe below.

## Problem

The goal is to optimize the acceleration and breaking of a race car to minimize time spent, constrained at each position by a varying maximum speed and a limited amount of fuel which is consumed during acceleration. The exact values of these constraints depend on the chosen tier for each constaint. 

## Optimization

Given a particular car configuration, the problem is to use the given fuel and tire limits as a efficiently as possible. The strategy is to optimize within the space of legal solutions that use either the gas or the tire entirely. We also optimize on the vector of velocities rather than on the accelerations because changes to a single velocity have only local changes while acceleration changes are compounding.

So we first form the vector of maximum possible velocities based on the track and limits on acceleration/breaking.

Now we define a solution as a vector of numbers which scale the velocity from the maximum allowed at each point. This may use more than the allowed gas or tire, but we can always scale a solution to just use up one or the other. We evaluate the solution by taking these scaled values for the velocities and computing the time.
