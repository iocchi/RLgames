#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys


if len(sys.argv) == 1:
    print "Use: plotresults.py <filename>"
    sys.exit(1)

    
fname = str(sys.argv[1]).replace('.dat','')
fname = fname.replace('data/','')
a = np.loadtxt('data/' + fname + '.dat', delimiter=",")

sv = a[:,0]  # score vector
rv = a[:,1]  # reward vector
gv = a[:,2]  # goal reached vector

n = len(rv)

plt.plot(sv,'b')
plt.title(fname)
plt.xlabel('Iteration')
plt.ylabel('Score')
# plt.savefig('fig/'+fname+'_s.png')
plt.show()

plt.plot(rv,'r')
plt.title(fname)
plt.xlabel('Iteration')
plt.ylabel('Reward')
#plt.savefig('fig/'+fname+'_r.png')
plt.show()

plt.ylim(ymin = -0.2, ymax = 1.2)
plt.plot(gv,'g.')
plt.title(fname)
plt.xlabel('Iteration')
plt.ylabel('Goal')
#plt.savefig('fig/'+fname+'_g.png')
plt.show()


d = 100 # size of interval
ss = [] # vector of avg scores in interval
rr = [] # vector of avg rewards in interval
gg = [] # vector of percentage of goals reached in interval
for i in range(0,n/d):
    si = sv[i*d:(i+1)*d]
    ri = rv[i*d:(i+1)*d]
    gi = gv[i*d:(i+1)*d]
    ss.append(np.mean(si))
    rr.append(np.mean(ri))
    gg.append(float((gi == 1).sum()*100.0)/d)

#if (ylim>0):
#    plt.ylim(ymin = 0, ymax = ylim)
plt.plot(ss,'b')
plt.title(fname)
plt.xlabel('Iteration/%d'%(d))
plt.ylabel('Avg Score')
# plt.savefig('fig/'+fname+'_as.png')
plt.show()

#if (ylim>0):
#    plt.ylim(ymin = 0, ymax = ylim)
plt.plot(rr,'r')
plt.title(fname)
plt.xlabel('Iteration/%d'%(d))
plt.ylabel('Avg Reward')
# plt.savefig('fig/'+fname+'_ar.png')
plt.show()

plt.ylim(ymin = -0.2, ymax = 1.2)
plt.plot(gg,'g')
plt.title(fname)
plt.xlabel('Iteration/%d'%(d))
plt.ylabel('% Goals')
#plt.savefig('fig/'+fname+'_pg.png')
plt.show()

