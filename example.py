#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 22:38:10 2023

@author: yongruipoh
"""

import sys
import numpy as np
# sys.path.append('../')   # Search for Python files one parent directory above
from objects import *
### Check command-line arguments ###
if len(sys.argv) not in [2]:
    sys.exit("Usage: python example.py num_procs")

### Number of parallel processes ###
num_procs = int(sys.argv[1])

### Simulation parameters ###
E = 24816.7
Delta = 6.0815
g_L = -0.0155
B = 0.35
C = 7387.05
# D = 0.733
D = 0
t_list = np.linspace(0, 2E-6, 100)
b = np.array([0.,11700.,1.])
e = np.array([-1.,1.,-11700.])
b[0], e[0] = -b[0], -e[0]

### Initialise ###
filename = os.path.splitext(os.path.basename(__file__))[0]
dynamics = Dynamics(E, Delta, g_L, B, C, D, t_list, b=b, e=e)

### Obtain aligned result ###
print("*** ALIGNED RESULT ***")
print("\n")
dynamics.prepare_and_solve()
dynamics.solve_g_mag()

### Save results ###
dynamics.save("", filename)

### Load results ###
dynamics = Dynamics.load("", filename)

### Make plots ###
dynamics.plot_quantity("g_mag", "", filename)

print(f"Maximum magnetisation: {dynamics.g_mag.max()}")
print(f"Minimum magnetisation: {dynamics.g_mag.min()}")
print(f"Final magnetisation: {dynamics.g_mag[-1]}")

sys.stdout.flush()      # Flush all output from the process

