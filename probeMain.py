
# from tslm import combine,one_q,query_sales,query_uci,query_taxi,get_numerator, get_denominator, classify_any_and,classify_any_or,union_test,intersection_test,new_classify_or,new_classify_and,classify
import matplotlib.pyplot as plt
import matplotlib as mpl

from ddpwlm import taxi as taxiDD
from ddpwlm import sales as salesDD

from naive import query_taxi as taxiTS
from naive import query_sales as salesTS

from probe_algo import query_taxi as taxiP
from probe_algo import query_sales as salesP

import sys 

# example: python3 probeMain naive sales u=10 b=0.005 t=0
def parseArgs(args):
    algo = args[1]
    datasetPresent = (len(args) > 2 and (args[2] == 'sales' or args[2] == 'taxi'))
    dataset = args[2] if datasetPresent else 'taxi'
    beta = -1
    u=-1
    n=2
    t=0
    a=0

    args = args[3:] if datasetPresent else args[2:]
    for arg in args:
        if arg[0] == 'b':
            beta = float(arg[2:])
        elif arg[0] == 'u':
            u =  float(arg[2:])
        elif arg[0] == 't':
            t = int(arg[2])
        elif arg[0] == 'n':
            n = int(arg[2]) 
        elif arg[0] == 'a':
            a= float(arg[2:])

    return algo,dataset,beta,u,t,n,a

def getAlgorithmName(algo):
    if algo == 'naive':
       return "Naive Threshold Shift Laplace Mechanism"
    elif algo == 'probe':
       return "ProBE Mechanism"

def executeProbe(algo,dataset,alpha,beta,u,t,n):
    a = 0.15 if alpha == 0 else alpha
    if algo == 'probe':
        if dataset == 'sales':
            ep,fnr,fpr = salesP('2017-01-04 00:00:00','2017-01-05 00:00:00',beta,a,t)
        elif dataset == 'taxi':
            ep,fnr,fpr = taxiP('2020-03-01 00:00:00','2020-03-02 00:00:00',beta,a,t)
        return ep[n-1],fnr[n-1],fpr[n-1]
    else: 
        if dataset == 'sales':
            epTS,fnrTS,fprTS,epNaive,fnrNaive,fprNaive = salesTS('2017-01-04 00:00:00','2017-01-05 00:00:00',beta,u,t)
        elif dataset == 'taxi':
            epTS,fnrTS,fprTS,epNaive,fnrNaive,fprNaive = taxiTS('2020-03-01 00:00:00','2020-03-02 00:00:00',beta,u,t)
        return epNaive[n-1],fnrNaive[n-1],fprNaive[n-1]

algo,dataset,beta,u,t,n,alpha = parseArgs(sys.argv)

ep,fnr,fpr= executeProbe(algo,dataset,alpha,beta,u,t,n)
print("*********************************RESULTS*********************************")
print("Algorithm: ", getAlgorithmName(algo))
print("EX-POST PRIVACY LOSS EPSILON:", ep)
print("FALSE NEGATIVE RATE:", fnr)
print("FALSE POSITIVE RATE:", fpr)


