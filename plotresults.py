#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse

# global vectors

sv = None  # score vector
rv = None  # reward vector
gv = None  # goal reached vector
ov = None  # optimal (no exploration) vector


def loaddata(filename):
    global sv,rv,gv,ov

    fname = str(filename).replace('.dat','')
    fname = fname.replace('data/','')
    a = np.loadtxt('data/' + fname + '.dat', delimiter=",")

    try:
        sv = a[:,2]  # score vector
        rv = a[:,3]  # reward vector
        gv = a[:,4]  # goal reached vector
        ov = a[:,6]  # optimal (no exploration) vector
    except: # old version
        sv = a[:,0]  # score vector
        rv = a[:,1]  # reward vector
        gv = a[:,2]  # goal reached vector
        ov = a[:,4]  # optimal (no exploration) vector



def plot1(fname):
    global sv,rv,gv

    plt.plot(sv,'b')
    plt.title(fname)
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    # plt.savefig('fig/'+fname+'_s.png')
    plt.show()

    top = 0
    bot = 100
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



def plotavg(fname):
    global sv,rv,gv

    maxscore = max(sv)
    n = len(rv)

    d = int(n/100) # size of interval

    xx = [0] # x axis vector
    ss = [0] # vector of avg scores in interval
    rr = [0] # vector of avg rewards in interval
    rrtop = [0]  # confidence interval
    rrbot = [0]  # confidence interval

    gg = [0] # vector of percentage of goals reached in interval
    rd = [0] # vector of reward every d iterations
    sd = [0] # vector of score every d iterations

    for i in range(0,int(n/d)):
        si = sv[i*d:min(n,(i+1)*d)]
        ri = rv[i*d:min(n,(i+1)*d)]
        gi = gv[i*d:min(n,(i+1)*d)]
        x = np.argwhere(ov[i*d:(i+1)*d]).flatten().tolist()
        #print "optimal runs:", x
        #oi = x[len(x)-1] + i*d
        #print("optimal run: %d %f %d %d" %(oi, rv[oi], sv[oi], ov[oi]))
        xx.append(i*d)
        ss.append(np.mean(si))
        rr.append(np.mean(ri))
        rrtop.append(np.mean(ri)+np.std(ri))
        rrbot.append(np.mean(ri)-np.std(ri))
        gg.append(float((gi == 1).sum()*100.0)/d)
        #rd.append(rv[oi])
        #sd.append(sv[oi])



    #if (ylim>0):
    #    plt.ylim(ymin = 0, ymax = ylim)
    plt.plot(xx,ss,'b')
    plt.title(fname+" - Average score")
    plt.xlabel('Iteration')
    plt.ylabel('Avg Score')
    # plt.savefig('fig/'+fname+'_as.png')
    plt.show()

    rrmax = max(rrtop)

    plt.ylim(ymin = 0, ymax = rrmax*1.05)
    plt.title(fname+" - Average reward")
    plt.fill_between(xx, rrtop, rrbot, facecolor='r', alpha=0.25)
    plt.plot(xx,rr,'r')
    plt.xlabel('Iteration')
    plt.ylabel('Avg Reward')
    # plt.savefig('fig/'+fname+'_ar.png')
    plt.show()

    #plt.ylim(ymin = -0.2, ymax = 1.2)
    plt.title(fname+" - % goals reached")
    plt.plot(xx,gg,'g')
    plt.xlabel('Iteration')
    plt.ylabel('% Goals')
    #plt.savefig('fig/'+fname+'_pg.png')
    plt.show()

    if False:
        plt.title(fname+" - Best policy score")
        plt.ylim(ymin = 0, ymax = maxscore*1.05)
        plt.plot(xx,sd,'b')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        # plt.savefig('fig/'+fname+'_as.png')
        plt.show()

        plt.title(fname+" - Best policy reward")
        #if (ylim>0):
        #    plt.ylim(ymin = 0, ymax = ylim)
        plt.plot(xx,rd,'r')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        # plt.savefig('fig/'+fname+'_ar.png')
        plt.show()


def plotdata(filename):
    global sv,rv,gv,ov

    loaddata(filename)

    plotavg(filename)





# main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot results')
    parser.add_argument('file', type=str, help='File name with data')
    #parser.add_argument('--reward', help='plot reward', action='store_true')
    #parser.add_argument('--score', help='plot score', action='store_true')

    args = parser.parse_args()

    plotdata(args.file)


