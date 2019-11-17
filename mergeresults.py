#!/usr/bin/python

import numpy as np
import sys
import argparse
from collections import defaultdict


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



def merge(tm,data,c,y):
    for i in range(0,len(data)):
        ti = tm[i]
        di = data[i]
        c[ti] += 1
        y[ti] += di


def mergedata(datafiles, fileout):

    c = defaultdict(int)
    y = defaultdict(float)

    for f in datafiles:
        tm,rv,fname = loaddata(f)
        if tm is not None:
            merge(tm,rv,c,y)

    f = open(fileout,'w')
    for i in range(0,len(y)):
        if (c[i]>0):
            y[i] /= c[i]
        f.write('0,%d,0,%f,0,0\n' %(i,y[i]))    
    f.close()



# main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Merge results')
    parser.add_argument('-out', type=str, help='output data file', default='data/output.dat')
    parser.add_argument('-datafiles', nargs='+', help='[Required] Data files to plot', required=True)

    args = parser.parse_args()

    mergedata(args.datafiles, args.out)



