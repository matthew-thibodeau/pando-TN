#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 12:02:06 2023

@author: matthewthibodeau
"""
import numpy as np
import os

ll_svd = []
ll_largest = []
for fname in os.listdir('data/mera_data'):
    if 'svd' in fname:
        x = np.load(f'data/mera_data/{fname}')
        ll_svd.append(x)
    elif 'largest' in fname:
        x = np.load(f'data/mera_data/{fname}')
        ll_largest.append(x)
        

