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

try:
    sv = a[:,2]  # score vector
    rv = a[:,3]  # reward vector
    gv = a[:,4]  # goal reached vector
    ov = a[:,6]  # optimal (no exploration) vector
except:
    sv = a[:,0]  # score vector
    rv = a[:,1]  # reward vector
    gv = a[:,2]  # goal reached vector
    ov = a[:,4]  # optimal (no exploration) vector

maxscore = max(sv)
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
#plt.savefig('fig/'+fname+'_g.png'
plt.show()


d = 100 # size of interval
ss = [0] # vector of avg scores in interval
rr = [0] # vector of avg rewards in interval
gg = [0] # vector of percentage of goals reached in interval
rd = [0] # vector of reward every d iterations
sd = [0] # vector of score every d iterations
for i in range(0,n/d):
    si = sv[i*d:(i+1)*d]
    ri = rv[i*d:(i+1)*d]
    gi = gv[i*d:(i+1)*d]
    x = np.argwhere(ov[i*d:(i+1)*d]).flatten().tolist()
    #print "optimal runs:", x
    oi = x[len(x)-1] + i*d
    #print("optimal run: %d %f %d %d" %(oi, rv[oi], sv[oi], ov[oi]))
    ss.append(np.mean(si))
    rr.append(np.mean(ri))
    gg.append(float((gi == 1).sum()*100.0)/d)
    rd.append(rv[oi])
    sd.append(sv[oi])



#if (ylim>0):
#    plt.ylim(ymin = 0, ymax = ylim)
plt.plot(ss,'b')
plt.title(fname+" - Average score")
plt.xlabel('Iteration/%d'%(d))
plt.ylabel('Avg Score')
# plt.savefig('fig/'+fname+'_as.png')
plt.show()

rrmax = max(rr)

plt.ylim(ymin = 0, ymax = rrmax*1.05)
plt.title(fname+" - Average reward")
plt.plot(rr,'r')
plt.xlabel('Iteration/%d'%(d))
plt.ylabel('Avg Reward')
# plt.savefig('fig/'+fname+'_ar.png')
plt.show()

#plt.ylim(ymin = -0.2, ymax = 1.2)
plt.title(fname+" - % goals reached")
plt.plot(gg,'g')
plt.xlabel('Iteration/%d'%(d))
plt.ylabel('% Goals')
#plt.savefig('fig/'+fname+'_pg.png')
plt.show()

plt.title(fname+" - Best policy score")
plt.ylim(ymin = 0, ymax = maxscore*1.05)
plt.plot(sd,'b')
plt.xlabel('Iteration/%d'%(d))
plt.ylabel('Score')
# plt.savefig('fig/'+fname+'_as.png')
plt.show()

plt.title(fname+" - Best policy reward")
#if (ylim>0):
#    plt.ylim(ymin = 0, ymax = ylim)
plt.plot(rd,'r')
plt.xlabel('Iteration/%d'%(d))
plt.ylabel('Reward')
# plt.savefig('fig/'+fname+'_ar.png')
plt.show()


