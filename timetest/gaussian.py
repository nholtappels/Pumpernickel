'''
Created on 24.11.2013

@author: Nick
'''

import math

def gaussian(value, mean, std):
    u = (value - mean) / abs(std)
    y = (1 / ((2 * math.pi) * abs(std)) ** 0.5) * math.exp(-u * u/2)
    return y