
# from tslm import combine,one_q,query_sales,query_uci,query_taxi,get_numerator, get_denominator, classify_any_and,classify_any_or,union_test,intersection_test,new_classify_or,new_classify_and,classify
import matplotlib.pyplot as plt
import matplotlib as mpl

from ppwlm import sales,taxi
from ddpwlm import taxi as taxiDD
from ddpwlm import sales as salesDD

from tslm import query_taxi as taxiTS
from tslm import query_sales as salesTS
import sys 

# example: python3 probeMain tslm sales u=10 b=0.005 m=3 f=2 e=0.001 t=0
def parseArgs(args):
    algo = args[1]
    datasetPresent = (len(args) > 2 and (args[2] == 'sales' or args[2] == 'taxi'))
    dataset = args[2] if datasetPresent else 'taxi'
    beta = -1
    u=-1
    m=-1
    f=-1
    e=-1
    n=2
    t=0
    args = args[3:] if datasetPresent else args[2:]
    for arg in args:
        if arg[0] == 'b':
            beta = float(arg[2:])
        elif arg[0] == 'u':
            u =  float(arg[2:])
        elif arg[0] == 'm':
            m = int(arg[2:])
        elif arg[0] == 'f':
            f = int(arg[2:])
        elif arg[0] == 'e':
            e = float(arg[2:])
        elif arg[0] == 't':
            t = int(arg[2])
        elif arg[0] == 'n':
            n = int(arg[2])   
    return algo,dataset,beta,u,m,f,e,t,n

def getAlgorithmName(algo):
    if algo == 'tslm':
       return "Probe-based Threshold Shift Laplace Mechanism"
    elif algo == 'naive':
       return "Naive Threshold Shift Laplace Mechanism"
    elif algo == 'ppwlm':
       return "Probe-based Multi-Step Predicate-wise Laplace Mechanism"
    elif algo == 'ddpwlm':
        return "Probe-based Multi-Step Predicate-wise Laplace Mechanism (Data-Dependent)"

def executeProbe(algo,dataset,beta,u,m,f,e,t,n):
    if algo == 'ppwlm':
        if dataset == 'sales':
            epProg,fnrProg,fprProg,entropyProg = sales('2017-01-04 00:00:00','2017-01-05 00:00:00',beta,u,m,e,t)
        elif dataset == 'taxi':
            epProg,fnrProg,fprProg,entropyProg = taxi('2020-03-01 00:00:00','2020-03-02 00:00:00',beta,u,m,e,t)
        return epProg[n-1],fnrProg[n-1],fprProg[n-1],entropyProg[n-1]
    elif algo == 'ddpwlm':
        if dataset == 'sales':
            epDD,fnrDD,fprDD,entropyDD = salesDD('2017-01-04 00:00:00','2017-01-05 00:00:00',beta,u,m,f,e,t)
        elif dataset == 'taxi':
            epDD,fnrDD,fprDD,entropyDD = taxiDD('2020-03-01 00:00:00','2020-03-02 00:00:00',beta,u,m,f,e,t)
        return epDD[n-1],fnrDD[n-1],fprDD[n-1],entropyDD[n-1]
    else: 
        if dataset == 'sales':
            epTS,fnrTS,fprTS,entropyTS,epNaive,fnrNaive,fprNaive,entropyNaive = salesTS('2017-01-04 00:00:00','2017-01-05 00:00:00',beta,u,t)
        elif dataset == 'taxi':
            epTS,fnrTS,fprTS,entropyTS,epNaive,fnrNaive,fprNaive,entropyNaive = taxiTS('2020-03-01 00:00:00','2020-03-02 00:00:00',beta,u,t)
        if algo == 'tslm':
            return epTS[n-1],fnrTS[n-1],fprTS[n-1],entropyTS[n-1]
        else:
            return epNaive[n-1],fnrNaive[n-1],fprNaive[n-1],entropyNaive[n-1]

algo,dataset,beta,u,m,f,e,t,n = parseArgs(sys.argv)
ep,fnr,fpr,entropy= executeProbe(algo,dataset,beta,u,m,f,e,t,n)
print("*********************************RESULTS*********************************")
print("Algorithm: ", getAlgorithmName(algo))
print("EX-POST PRIVACY LOSS EPSILON:", ep)
print("FALSE NEGATIVE RATE:", fnr)
print("FALSE POSITIVE RATE:", fpr)
print("MIN-ENTROPY:", entropy)


