#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:04:16 2020

@author: alessandro
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm



### deprecated
def read_data(filename, separator=',',starting_line = 0, ending_line = 10**10):
    file = open(filename,'r')
    v = []
    line = file.readline().rstrip('\n').lstrip(' ')
    count = 0
    while(len(line) > 0 and count < ending_line):
        if count < starting_line:
            count += 1
            continue
        if line[0] == '#':
            continue
        
        time, ch = line.split(separator)
        couple = [int(time.strip(' ')), int(ch.strip(' '))]
        v.append(couple)
        line = file.readline()
        count += 1
    v = np.array(v)
    file.close()
    return v


def time_difference_histo(data, timestep, plot = True):
    t_diff = np.zeros(len(data) - 1)
    for i in range(0,len(data) - 1):
        t_diff[i] = data[i + 1,0] - data[i,0]
    Dt_max = np.max(t_diff)
    n_bins = Dt_max/timestep + 1
    bins = np.arange(0,timestep*n_bins,timestep)
    
    if plot:
        plt.figure()
        plt.hist(t_diff,bins=bins)
        plt.xlabel('Dt/tau')
        plt.ylabel('counts')
        plt.show()
        
    return t_diff,bins

def photon_rate_histo(data, timestep, plot = True):
    rates = np.zeros(int(data[-1,0]/timestep) + 1)
    for t in data[:,0]:
        rates[int(t/timestep)] += 1
    rates = np.array(rates)
    max_r = np.max(rates)
    bins = np.arange(0,max_r + 1)
    
    if plot:
        plt.figure()
        plt.hist(rates,bins=bins)
        plt.xlabel('number of photons')
        plt.ylabel('counts')
        plt.show()
        
    return rates,bins
        

def binary_to_int(data,n_bits):
    '''
    data must a be a string containing only 0 and 1
    '''
    v = []
    for i in range(int(len(data)/n_bits)):
        n = int(data[i*n_bits:(i + 1)*n_bits],2)
        v.append(n)
    return np.array(v)
    

def random_plot_3d(data, n_bits):
    '''
    data must a be a string containing only 0 and 1
    '''
    v = binary_to_int(data,n_bits)
    v = v[0:3*int(len(v)/3)]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(v[0::3],v[1::3],v[2::3])
    

def ACF(data,max_distance):
    acf = np.zeros(max_distance + 1)
    v = 2*binary_to_int(data,1) - 1
    acf[0] = 1.
    for i in range(1, max_distance + 1):
        s = 0.
        for j in range(i,len(v)):
            s += v[j]*v[j - i]
        acf[i] = s*1./(len(v) - i)
    return acf

def photon_random_string(data, criterion = 'photon_rate', window_lenght = 1):
    t_diff = np.zeros(len(data) - 1)
    for i in range(0,len(data) - 1):
        t_diff[i] = data[i + 1,0] - data[i,0]
    m = np.mean(t_diff)
    s = ''
    if criterion == 'time_difference':
        for t in t_diff:
            if t <= m:
                s += '0'
            else:
                s += '1'
        return s
    if criterion == 'photon_rate':
        timestep = window_lenght*m
        rates = np.zeros(int(data[-1,0]/timestep) + 1)
        for t in data[:,0]:
            rates[int(t/timestep)] += 1
        rates = np.array(rates)
        
        for r in rates:
            if r%2 == 0:
                s += '0'
            else:
                s += '1'
    return s
        
        
def pseudo_random_string(lenght):
    s = ''
    for i in range(lenght):
        r = np.random.uniform(0,1)
        if r <= 0.5:
            s += '0'
        else:
            s += '1'
    return s



###### coincidences

def update(c,c_l):
    update_keys = []
    if c_l[0] == 2:
        update_keys.append('2')
        if 3 in c_l:
            update_keys.append('23')
            if 4 in c_l:
                update_keys.append('234')
        if 4 in c_l:
            update_keys.append('24')
    elif c_l[0] == 3:
        update_keys.append('3')
        if 2 in c_l:
            update_keys.append('23')
            if 4 in c_l:
                update_keys.append('234')
        if 4 in c_l:
            update_keys.append('34')
    else:
        update_keys.append('4')
        if 2 in c_l:
            update_keys.append('24')
            if 3 in c_l:
                update_keys.append('234')
        if 3 in c_l:
            update_keys.append('34')
    
    for i in range(len(c)):
        if c['kind'][i] in update_keys:
            c['count'][i] += 1
    
    return c


def coincidences(data,tolerance):
    dtype = [('kind','U3'),('count',int)]
    v = [('2',0),('3',0),('4',0),('23',0),('34',0),('24',0),('234',0)]
    coinc = np.array(v,dtype=dtype)
    
    
    for i in tqdm(range(len(data))):
        t,ch = data[i]
        ch_list = [ch]
        j = 1
        while(i+j < len(data) and data[i+j,0] - t < tolerance):
            ch_list.append(data[i+j,1])
            j += 1
        coinc = update(coinc,ch_list)
    
    return coinc

def plot_coincidences(coinc):
    plt.figure()
    plt.bar(coinc['kind'],coinc['count'])
    plt.show()


def delay_distribution(data,neighbors = 10,tolerance = 100,biases=[0,0,0]):
    '''
    Triggers when the herald is detected, looks for the other two photons and saves the time delay
    '''
    
    v = []
    for i in tqdm(range(neighbors, len(data) - neighbors)):
        if data[i,1] != 4:
            continue
        t = data[i,0]
        delays = [0,0,0] # delay 4-2, delay 4-3, delay 4-4
        for j in range(-neighbors, neighbors):
            ch = data[i+j,1] - 2
            d = data[i+j,0] - t
            if ch == 2 and d <= 0:
                continue
            if np.abs(d - biases[ch]) > tolerance:
                continue
            if delays[ch] == 0:
                delays[ch] = d
            elif np.abs(d) < np.abs(delays[ch]):
                delays[ch] = d
        v.append(delays)
    return np.array(v)
    



