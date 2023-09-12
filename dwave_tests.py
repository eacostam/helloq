#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 18:19:34 2023

!pip install dwave-ocean-sdk

@author: ernesto.acosta
"""

from dwave.cloud import Client

# Replace 'YOUR_API_TOKEN' with your D-Wave API token
client = Client.from_config(token='DEV-9497f5036e9b82253b880fd0562ae8386487f9cd')
solver = client.get_solver()

qubo = {(0, 0): -1, (1, 1): -1, (0, 1): 2}

computation = solver.sample_qubo(qubo, num_reads=100)

samples = computation.samples
energies = computation.energies

print("Samples: " + str(samples))
print("Energies: " + str(energies))