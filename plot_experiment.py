import os
import argparse


def plotmerge(exp,params,runid):
    ri = 0
    merged = ''
    e0 = None
    for e in exp:
        if e0 is None:
            e0 = e[0]
        cmdm = 'python3 mergeresults.py -datafiles '
        cmdp = 'python3 plotresults.py -datafiles '
        for i in e[1]:
            cmdm += 'data/%s_%s_%02d ' %(e[0],params,i)
            cmdp += 'data/%s_%s_%02d ' %(e[0],params,i)
        cmdm += ' -out data/%s_%s_m.dat' %(e[0],params)
        merged += 'data/%s_%s_m.dat ' %(e[0],params)
        cmdp += ' -save fig/%s_%s_m.png' %(e[0],params)
        print(cmdm)
        os.system(cmdm)
        print(cmdp)
        os.system(cmdp)    

    cmd = 'python3 plotresults.py -datafiles %s' %(merged)
    cmd += ' -save fig/%s.png' %e0
    print(cmd)
    os.system(cmd)    


def unused():
        cmdm = 'python3 mergeresults.py -datafiles '
        cmdp = 'python3 plotresults.py -datafiles '
        for i in runid[ri]:
            cmdm += 'data/%sO_%s_%02d ' %(e,params,i)
            cmdp += 'data/%sO_%s_%02d ' %(e,params,i)
        cmdm += ' -out data/%sO_%s_m.dat' %(e,params)
        print(cmdm)
        os.system(cmdm)    
        print(cmdp)
        os.system(cmdp)    

        cmd = 'python3 plotresults.py -datafiles data/%s_%s_m data/%sO_%s_m' %(e,params,e,params)
        cmd += ' -save fig/%s.png' %e        
        print(cmd)
        os.system(cmd)    

        ri += 1

def plotall(exp,params,runid):
    ri = 0
    for e in exp:
      for i in runid[ri]:
        cmd = 'python3 plotresults.py -datafiles data/%s_%s_%02d data/%sO_%s_%02d' %(e,params,i,e,params,i)
        print(cmd)
        os.system(cmd)    
      ri += 1


# main
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot results of experiment')
    parser.add_argument('name', type=str, help='Game name initial (B,S,M)')

    args = parser.parse_args()

    exp = []
    params = ''
    runid = [ [1,2,3], [1,2,3] ]

    if (args.name=='B'):
        exp.append(['B44NRA', [3,101,105]])  # 1
        exp.append(['B44NRAO', [1,2,3]])
        params = 'S_g0999_eAv_n500'
    if (args.name=='BRL'):
        exp.append(['B44NRARL', [1,2,3]])
        exp.append(['B44NRARLO', [1,2,3]])
        params = 'S_g0999_eAv_n200'
    elif (args.name=='S'):
        exp.append(['S3', [101,102,103]])
        exp.append(['S3O', [101,102,103]])
        params = 'S_g099_n20'        
    elif (args.name=='SD'):
        exp.append(['S3D', [101,102,103]])
        exp.append(['S3DO', [101,102,103]])
        params = 'S_g099_n20' 
    elif (args.name=='M'):
        exp.append(['M', [1,2,3]])
        exp.append(['MO', [1,2,3]])
        params = 'S_g0990_n20'
    elif (args.name=='MD'):
        exp.append(['MD', [1,2,3]])
        exp.append(['MDO', [1,2,3]])
        params = 'S_g0990_n20'

    if (len(exp)>0):
        #plotall()
        plotmerge(exp,params,runid)




