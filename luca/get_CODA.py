#!/usr/bin/env python
# -*- coding: utf8 -*- 

import numpy as np
import os
import matplotlib.pyplot as plt
import glob


def read_chain(base_path, fchain):
  # read CODAindex.txt
  findex = os.path.join(base_path, 'CODAindex.txt')
  #CODAidx = np.genfromtxt(findex, dtype = ['S5', 'i8', 'f8'])
  ofidx = open(findex, 'r')
  lines = ofidx.readlines()
  ofidx.close()
  nidx = len(lines)
  #print nidx

  var_name = []
  var_start = []
  var_end = []
  
  for i in range(0, nidx):
    line = lines[i].strip().split()
    var_name.append(line[0])
    var_start.append(int(line[1]))
    var_end.append(int(line[2]))

  # read CODAchains.txt
  CODAchains = np.genfromtxt(fchain)

  print fchain

  fig, ax = plt.subplots(ncols=2, nrows=nidx,figsize=(8,8))

  for i in range(0, nidx):
    print ' %s' %var_name[i]
    temp = CODAchains[var_start[i]-1:var_end[i],:]
    
    median = np.median( temp[:,1] )
    mean = np.mean( temp[:,1] )
    std = np.std( temp[:,1], ddof=1)
    print ' median = %.6f' %(median)
    print ' mean = %.6f' %(mean)
    print ' std = %.6f' %(std)

    #ax0.plot(temp[:,0], temp[:,1])
    ax[i,0].plot(temp[:,0], temp[:,1])
    ax[i,0].set_ylabel(var_name[i])
    
    k = np.ceil(2. * temp.shape[0]**(1./3.)).astype(int)
    #hist, bin_edges = np.histogram(temp[:,1], k, density = False)
    #ax1.hist(temp[:,1], k, normed=1, histtype='stepfilled', alpha=0.75)
    ax[i,1].hist(temp[:,1], k, normed=1, histtype='stepfilled', alpha=0.75)
    ax[i,1].set_xlabel(var_name[i])
    
  plt.show()
  

base_path = os.path.abspath('Poisson')

chains = glob.glob(os.path.join(base_path, 'CODAchain*.txt'))
print chains

for i in range(0, len(chains)):
  print
  read_chain(base_path, chains[i])




