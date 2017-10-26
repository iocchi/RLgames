#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys


if len(sys.argv) == 1:
    print "Use: plotscores.py <filename>"
    sys.exit(1)

    
#ylim = 1500 
ylim = -1 # nolimit
#goal_limit = 1200 # game G1
goal_limit = 1900 # game G2
RAFail = 7


fname = str(sys.argv[1]).replace('.dat','')
fname = fname.replace('data/','')
a = np.loadtxt('data/' + fname + '.dat', delimiter=",")
#a = np.loadtxt(fname + '.dat', delimiter=",")

# plt.plot(a[:,0])
# plt.title(fname)
# plt.xlabel('Iteration')
# plt.ylabel('Score')
# plt.show()

if (ylim>0):
    plt.ylim(ymax = ylim, ymin = 0)
plt.plot(a[:,1])
plt.title(fname)
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.savefig('fig/'+fname+'_r.png')
plt.show()

scv = a[:,0]
rv = a[:,1]  # reward vector
#sv = a[:,2]  # RA state vector
n = len(rv)

#for i in range(0,len(sv)):  # remove failure states
#    if (sv[i]==RAFail):
#        sv[i]=-1
    

d = 100 # size of interval
rr = [] # vector of avg rewards in interval
cg = [] # vector of percentage of goals reached in interval
for i in range(0,n/d):
    ai = rv[i*d:(i+1)*d]
    rr.append(np.mean(ai))
    cg.append(float((ai > goal_limit).sum()*100.0)/d)

if (ylim>0):
    plt.ylim(ymin = 0, ymax = ylim)
plt.plot(rr)
plt.title(fname)
plt.xlabel('Iteration/%d'%(d))
plt.ylabel('Avg Reward')
# plt.savefig('fig/'+fname+'_a.png')
plt.show()

#plt.ylim(ymin = -2, ymax = 7)
plt.plot(scv,'r.')
plt.title(fname)
plt.xlabel('Iteration')
plt.ylabel('Score')
# plt.savefig('fig/'+fname+'_g.png')
plt.show()
