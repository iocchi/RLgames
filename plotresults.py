#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse


def loaddata(filename):

    try:
        fname = str(filename).replace('.dat','')
        fname = fname.replace('data/','')
        a = np.loadtxt('data/' + fname + '.dat', delimiter=",")
    except:
        print("Error in loading file ",fname)
        return None, None, None

    try:
        tm = np.array(a[:,1])  # time vector
        sv = a[:,2]  # score vector
        rv = a[:,3]  # reward vector
        gv = a[:,4]  # goal reached vector
#        ov = a[:,6]  # optimal (no exploration) vector
    except: # old version
        sv = a[:,0]  # score vector
        rv = a[:,1]  # reward vector
        gv = a[:,2]  # goal reached vector
        tm = range(0,len(rv))
#        ov = a[:,4]  # optimal (no exploration) vector

    return tm, rv, fname



def getplotdata(tm,data):
    x = [] # x axis vector
    y = [] # y axis vector 
    ytop = []  # confidence interval (top-edge)
    ybot = []  # confidence interval (bottom-edge)

    n = len(data)
    d = int(n/100) # size of interval

    for i in range(0,int(n/d)):
        di = data[i*d:min(n,(i+1)*d)]
        ti = tm[i*d:min(n,(i+1)*d)]
        if (len(ti)>0):
            x.append(np.mean(ti))
            y.append(np.mean(di))
            ytop.append(np.mean(di)+0.5*np.std(di))
            ybot.append(np.mean(di)-0.5*np.std(di))

    return x,y,ytop,ybot


def showplots(xx,yy,yytop,yybot,yylabel,save):

    colors = ['r','b','g','yellow','cyan','magenta']
    
    ytop = max(max(l) for l in yytop)

    plt.ylim(bottom = 0, top = ytop*1.2)
    plt.title("Average reward")
    plt.xlabel('Time')
    plt.ylabel('Avg Reward')

    for i in range(0,len(xx)):
        plt.fill_between(xx[i], yytop[i], yybot[i], facecolor=colors[i], alpha=0.25)
        plt.plot(xx[i],yy[i],colors[i],label=yylabel[i])

    plt.legend()

    if save is not None:
        plt.savefig(save)
        print('File saved: ',save)

    plt.show()


def plotdata(datafiles, save):
    xx = []
    yy = []
    yytop = []
    yybot = []
    yylabel = []

    for f in datafiles:
        tm,rv,fname = loaddata(f)
        if tm is not None:
            x,y,ytop,ybot = getplotdata(tm,rv)
            xx += [x]
            yy += [y]
            yytop += [ytop]
            yybot += [ybot]
            yylabel += [fname]

    if (len(xx)>0):
        showplots(xx,yy,yytop,yybot,yylabel,save)



# main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot results')
    #parser.add_argument('file', type=str, help='File name with data')
    #parser.add_argument('--reward', help='plot reward', action='store_true')
    #parser.add_argument('--score', help='plot score', action='store_true')
    parser.add_argument('-save', type=str, help='save figure on specified file', default=None)

    parser.add_argument('-datafiles', nargs='+', help='[Required] Data files to plot', required=True)

    args = parser.parse_args()

    plotdata(args.datafiles, args.save)


