import numpy as np
import math
from scipy.stats import entropy,norm
from decimal import Decimal, Context
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import os
import query
import datetime
import time
import random
from collections import OrderedDict
import pandas as pd
from statistics import mean,stdev,mode,median
import mysql.connector
from queries.queries_sales import sales_volume,average_transaction_value,distinct_customers,transaction_per_category,avg_total_per_category,avg_itemprice_per_category,category_count,average_item_number,sales_volume_women
from queries.queries_sales import sales_volume_marmara
from itertools import combinations
from queries.taxi_queries import query_taxi_count_location,query_taxi_count,query_taxi_fareamount,query_congestion_amount,query_taxi_tip_amount,query_taxi_tolls_amount,query_taxi_total_amount,query_taxi_count_payment_type,query_taxi_count_flag
from utils import intersection_test,union_test,classify_mix,get_denominator,get_numerator,getBi,new_classify_and,new_classify_or,getEntropyArr

def classify(countsByPartition,thresholds,selected):
    tp=[]
    tn=[]
    fp=[]
    fn=[]
    for i in range(len(countsByPartition)):
        if countsByPartition[i]>thresholds[i] and i in selected:
            tp.append(i)
        elif countsByPartition[i]>thresholds[i] and i not in selected:
            fn.append(i)
        elif countsByPartition[i]<=thresholds[i] and i in selected:
            fp.append(i)
        elif countsByPartition[i]<=thresholds[i] and i not in selected:
            tn.append(i)
    return tp,tn,fp,fn

def threshold_shift(countsByPartition, thresholds, uncertainRegion, failingRate, epsilonMax, l_sensitivity):
    ep = l_sensitivity*np.log( 0.5 / failingRate) /uncertainRegion*1.0
    selected = []
    epList = []
    uncertainRegionList = []
    if(ep<epsilonMax):

        lap_noises = np.random.laplace(0, 1.0*l_sensitivity/ep, len(countsByPartition))
        for i in range(len(countsByPartition)):
            epList.append(ep)
            uncertainRegionList.append(uncertainRegion)
            if(countsByPartition[i]+lap_noises[i]>thresholds[i]-uncertainRegion):
                selected.append(i)
    return selected, ep


def get_g(counts,ts):
    lists = []
    for i in range(len(counts)):
        lists.append([0] for j in range(len(counts[i])))
        lists[i] =[j for j in range(len(counts[i])) if counts[i][j] <= ts[i]]
    return intersection_test(lists)

def get_intersection_conj(counts,ts):
    lists = []
    for i in range(len(counts)):
        lists.append([0] for j in range(len(counts[i])))
        lists[i] =[j for j in range(len(counts[i])) if counts[i][j] > ts[i]]
    return intersection_test(lists)
    
def get_union(counts,ts):
    lists=[]*len(counts)
    for i in range(len(counts)):
        lists.append([0] for j in range(len(counts[i])))
        lists[i] =[j for j in range(len(counts[i])) if counts[i][j] > ts[i]]
    return union_test(lists)

def get_union_conj(counts,ts):
    lists=[]*len(counts)
    for i in range(len(counts)):
        lists.append([0] for j in range(len(counts[i])))
        lists[i] =[j for j in range(len(counts[i])) if counts[i][j] <= ts[i]]
    return union_test(lists)

def one_q(counts,t,failingRate,a,all_a,idx,l,epsilonMax,iter=100,alt=False,predicate_list=[],one=False):
    if(one):
        return threshold_shift(counts,[t]*len(counts), a,failingRate,epsilonMax,l)
    prod = get_numerator(a,idx,all_a)*l
    b1 = (prod/get_denominator(all_a,False,idx,s=[l]))*failingRate
    selected,ep = threshold_shift(counts,[t]*len(counts), a,b1,epsilonMax,l)
    return selected,ep

def one_q_eq(counts,t,failingRate,a,all_a,idx,l,epsilonMax,iter=100,alt=False,predicate_list=[],one=False):
    if one == True:
        selected,ep = threshold_shift(counts,[t]*len(counts), a,failingRate,epsilonMax,l)
        return selected,ep
    b1 = failingRate/len(all_a)
    selected,ep = threshold_shift(counts,[t]*len(counts), a,b1,epsilonMax,l)
    return selected,ep


def classify_mix(c,t,selected,type, all_preds,distrib=False):
    fn,fp,tn,tp = [],[],[],[]
    og_positives = [[] for i in range(len(c))]
    og_negatives = [[] for i in range(len(c))]
    og_final = []
    for i,count in enumerate(c):
        for j in range(len(count)):
            if(count[j] > t[i]):
                og_positives[i].append(j)
            else:
                og_negatives[i].append(j)
    if len(c) == 1:
        og_final=og_positives[0]
    elif len(c) == 3:
        if type == 1:
            if distrib==False:
                pos = union_test(og_positives[1:3])
                og_final = intersection_test([pos,og_positives[0]])
            else:
                pos = intersection_test(og_positives[0:2])
                pos2 = intersection_test([og_positives[0],og_positives[2]])
                og_final = union_test([pos,pos2])
        else:
            if distrib==False:
                pos = intersection_test(og_positives[0:2])
                og_final = union_test([pos,og_positives[2]])
            else:
                pos = union_test(og_positives[0:2])
                pos2 = union_test([og_positives[0],og_positives[2]])
                og_final = intersection_test([pos,pos2])
    elif len(c) == 4:
        if type == 0: # N N U
            pos = intersection_test(og_positives[0:3])
            og_final = union_test([pos,og_positives[3]])
        elif type == 1: # N U U
            pos = intersection_test(og_positives[0:2])
            pos2 = union_test(og_positives[2:4])
            og_final = union_test([pos,pos2])
    elif len(c) == 5:
        if type == 0: # N N N U
            pos = intersection_test(og_positives[0:4])
            og_final = union_test([pos,og_positives[4]])
        elif type == 1: # N N U U
            pos = intersection_test(og_positives[0:3])
            pos2 = union_test(og_positives[3:5])
            og_final = union_test([pos,pos2])
        elif type == 2: # N U U U
            pos = intersection_test(og_positives[0:2])
            pos2 = union_test(og_positives[2:5])
            og_final = union_test([pos,pos2])
    elif len(c) == 6:
        if type == 0: # N N N N U
            pos = intersection_test(og_positives[0:5])
            og_final = union_test([pos,og_positives[5]])
        elif type == 1: # N N N U U
            pos = intersection_test(og_positives[0:4])
            pos2 = union_test(og_positives[4:6])
            og_final = union_test([pos,pos2])
        elif type == 2: # N N U U U
            pos = intersection_test(og_positives[0:3])
            pos2 = union_test(og_positives[3:6])
            og_final = union_test([pos,pos2])
        elif type == 3: # N U U U U
            pos = intersection_test(og_positives[0:2])
            pos2 = union_test(og_positives[2:6])
            og_final = union_test([pos,pos2])

    for i in all_preds:
        if (i in selected) and (i not in og_final):
            fp.append(i)
        elif (i in selected) and (i in og_final):
            tp.append(i)
        elif (i not in selected) and (i not in og_final):
            tn.append(i)
        else:
            fn.append(i)
    #("NEW TP TN FP FN",tp,tn,fp,fn)
    return tp,tn,fp,fn

def combine(counts_total,t_total,failingRate,a_total,l_sensitivity_total,epsilonMax,predicate_list=[],vary=False):
    eps,fns,fps,epNA,fnNA,fpNA = [[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)]
    if vary == True:
        for i in range(100):
            #for n=3 we do N U  
            ep_total_opt = 0
            ep_total_eqb = 0
            combos = [list(c) for c in combinations(counts_total,3)]
            q1 = counts_total[0]
            q2 = counts_total[1]
            #for n=3 we do N U
            q3 = counts_total[2]
            q_i = [q1,q2,q3]
            idx = [q[1] for q in q_i]
            counts = [counts_total[j][0] for j in idx]
            ts = [t_total[j] for j in idx]
            aes = [a_total[j] for j in idx]
            ls = [l_sensitivity_total[j] for j in idx]
            preds = [predicate_list[j] for j in idx]
            #CONJ first Q1 N Q2
            opt,eqb,tn,tp = conj(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],epsilonMax,iter=1,aes=aes,og_idx=[0,1],predicate_list=preds[0:2],combine=True)
            ep12_opt = opt[0]
            ep12_eqb = eqb[0]

            ep_total_opt += ep12_opt
            ep_total_eqb += ep12_eqb

            selected12_opt = opt[9]
            selected12_eqb = eqb[9]

            preds12_before_opt = opt[13]
            preds12_before_eqb = eqb[13]

            preds12_after_opt = opt[14]
            preds12_after_eqb = eqb[14]
            #SINGLE Q3
            bi = getBi(aes,aes[2],2,failingRate)
            bi_n = failingRate/3
            selected3_opt,ep3_opt = one_q(counts[2],ts[2],bi,aes[2],aes,2,ls[2],epsilonMax,iter=1)
            selected3_eqb,ep3_eqb = one_q_eq(counts[2],ts[2],bi_n,aes[2],aes,2,ls[2],epsilonMax,iter=1)

            preds3_opt = [preds[2][k] for k in selected3_opt]
            preds3_eqb = [preds[2][k] for k in selected3_eqb]

            ep_total_opt += ep3_opt
            ep_total_eqb += ep3_eqb
            
            # Q12 U Q3
            selected_total_opt = [selected12_opt,selected3_opt]
            selected_final_opt = union_test(selected_total_opt)

            selected_total_eqb = [selected12_eqb,selected3_eqb]
            selected_final_eqb = union_test(selected_total_eqb)
            
            preds_after_shift_opt = union_test([preds12_after_opt,preds3_opt]) 
            preds_before_shift_opt = union_test([preds12_before_opt,preds[2]])

            preds_after_shift_eqb = union_test([preds12_after_eqb,preds3_eqb]) 
            preds_before_shift_eqb = union_test([preds12_before_eqb,preds[2]])

            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_opt,0,preds_before_shift_opt)
            positives = tp + fn
            negatives = tn + fp
            fps[2].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[2].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[2].append(ep_total_opt)

            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_eqb,0,preds_before_shift_eqb)
            
            positives = tp + fn
            negatives = tn + fp
            epNA[2].append(ep_total_eqb)
            fpNA[2].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fnNA[2].append(len(fn)/len(positives) if len(positives)> 0 else 0)   
        return mean(eps[2]),mean(fns[2]),mean(fps[2]), mean(epNA[2]),mean(fnNA[2]),mean(fpNA[2])

    for i in range(150):
        #for n =1 just one query
        q_i = random.choice(counts_total)
        idx =  q_i[1]
        counts = [q_i[0]]
        ts = [t_total[idx]]
        aes = [a_total[idx]]
        ls = [l_sensitivity_total[idx]]
        preds = [predicate_list[idx]]
        selected_opt,ep_opt = one_q(counts[0],ts[0],failingRate,aes[0],aes,0,ls[0],epsilonMax,iter=1,one=True)
        selected_eqb,ep_eqb = one_q_eq(counts[0],ts[0],failingRate,aes[0],aes,0,ls[0],epsilonMax,iter=1,one=True)
        
        
        preds_opt = [preds[0][k] for k in selected_opt]
        preds_eqb = [preds[0][k] for k in selected_eqb]
        tp,tn,fp,fn = new_classify_or(counts,[[ts[k]]*len(counts[k]) for k in range(len(ts))],[],preds_opt,preds[0],preds)
        positives = tp + fn
        negatives = tn + fp
        fps[0].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
        fns[0].append(len(fn)/len(positives) if len(positives)> 0 else 0)
        eps[0].append(ep_opt)

        fpNA[0].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
        fnNA[0].append(len(fn)/len(positives) if len(positives)> 0 else 0)
        epNA[0].append(ep_opt)


        #first for n=2 we do either U or N
        #randomly choose a pair of sub-queries
        combos = [list(c) for c in combinations(counts_total,2)]
        q_i = random.choice(combos)
        idx = [q[1] for q in q_i]
        
        counts = [counts_total[j][0] for j in idx]
        ts = [t_total[j] for j in idx]
        aes = [a_total[j] for j in idx]
        ls = [l_sensitivity_total[j] for j in idx]
        preds = [predicate_list[j] for j in idx]
        funcs = [disj,conj]
        if i < 75:
            opt,eqb,tn,tp= disj(counts,ts,failingRate,aes,ls,epsilonMax,iter=1, predicate_list=preds)
            eps[1].append(opt[0])
            epNA[1].append(eqb[0])
            fps[1].append(opt[4])
            fpNA[1].append(eqb[4])
            fns[1].append(opt[2])
            fnNA[1].append(eqb[2])
        else:
            opt,eqb,tn,tp= conj(counts,ts,failingRate,aes,ls,epsilonMax,iter=1, predicate_list=preds)
            eps[1].append(opt[0])
            epNA[1].append(eqb[0])
            fps[1].append(opt[4])
            fpNA[1].append(eqb[4])
            fns[1].append(opt[2])
            fnNA[1].append(eqb[2])
        #for n=3 we do N U  
        ep_total_opt = 0
        ep_total_eqb = 0
        combos = [list(c) for c in combinations(counts_total,3)]
        q_i = random.choice(combos)
        idx = [q[1] for q in q_i]
        counts = [counts_total[j][0] for j in idx]
        ts = [t_total[j] for j in idx]
        aes = [a_total[j] for j in idx]
        ls = [l_sensitivity_total[j] for j in idx]
        preds = [predicate_list[j] for j in idx]
        #CONJ first Q1 N Q2
        opt,eqb,tn,tp = conj(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],epsilonMax,iter=1,aes=aes,og_idx=[0,1],predicate_list=preds[0:2],combine=True)
        ep12_opt = opt[0]
        ep12_eqb = eqb[0]

        ep_total_opt += ep12_opt
        ep_total_eqb += ep12_eqb

        selected12_opt = opt[9]
        selected12_eqb = eqb[9]

        preds12_before_opt = opt[13]
        preds12_before_eqb = eqb[13]

        preds12_after_opt = opt[14]
        preds12_after_eqb = eqb[14]
        #SINGLE Q3
        bi = getBi(aes,aes[2],2,failingRate)
        bi_n = failingRate/3
        selected3_opt,ep3_opt = one_q(counts[2],ts[2],bi,aes[2],aes,2,ls[2],epsilonMax,iter=1)
        selected3_eqb,ep3_eqb = one_q_eq(counts[2],ts[2],bi_n,aes[2],aes,2,ls[2],epsilonMax,iter=1)

        preds3_opt = [preds[2][k] for k in selected3_opt]
        preds3_eqb = [preds[2][k] for k in selected3_eqb]

        ep_total_opt += ep3_opt
        ep_total_eqb += ep3_eqb
        
        # Q12 U Q3
        selected_total_opt = [selected12_opt,selected3_opt]
        selected_final_opt = union_test(selected_total_opt)

        selected_total_eqb = [selected12_eqb,selected3_eqb]
        selected_final_eqb = union_test(selected_total_eqb)
        
        preds_after_shift_opt = union_test([preds12_after_opt,preds3_opt]) 
        preds_before_shift_opt = union_test([preds12_before_opt,preds[2]])

        preds_after_shift_eqb = union_test([preds12_after_eqb,preds3_eqb]) 
        preds_before_shift_eqb = union_test([preds12_before_eqb,preds[2]])

        tp,tn,fp,fn = classify_mix(counts,ts,selected_final_opt,0,preds_before_shift_opt)
        positives = tp + fn
        negatives = tn + fp
        fps[2].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
        fns[2].append(len(fn)/len(positives) if len(positives)> 0 else 0)
        eps[2].append(ep_total_opt)

        tp,tn,fp,fn = classify_mix(counts,ts,selected_final_eqb,0,preds_before_shift_eqb)
        
        positives = tp + fn
        negatives = tn + fp
        epNA[2].append(ep_total_eqb)
        fpNA[2].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
        fnNA[2].append(len(fn)/len(positives) if len(positives)> 0 else 0)

        #for n=4 we do N U U or N N U
        ep_total_opt = 0
        ep_total_eqb = 0
        #randomly choose 4 queries
        combos = [list(c) for c in combinations(counts_total,4)]
        q_i = random.choice(combos)
        idx = [q[1] for q in q_i]
        counts = [counts_total[j][0] for j in idx]
        ts = [t_total[j] for j in idx]
        aes = [a_total[j] for j in idx]
        ls = [l_sensitivity_total[j] for j in idx]
        preds = [predicate_list[j] for j in idx]
        #each iteration randomly choose to run either N U U or N N U
        flag = random.choice([0,1])
        flag = 0 if i < 75 else 1
        if flag == 1:
            #first N
            opt,eqb,tn,tp = conj(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],epsilonMax,iter=1,aes=aes,og_idx=[0,1],predicate_list=preds[0:2],combine=True)
            ep12_opt = opt[0]
            ep12_eqb = eqb[0]

            ep_total_opt += ep12_opt
            ep_total_eqb += ep12_eqb

            selected12_opt = opt[9]
            selected12_eqb = eqb[9]

            preds12_before_opt = opt[13]
            preds12_before_eqb = eqb[13]

            preds12_after_opt = opt[14]
            preds12_after_eqb = eqb[14]

            #last U 
            opt,eqb,tn,tp = disj(counts[2:4],ts[2:4],failingRate,aes[2:4],ls[2:4],epsilonMax,iter=1,aes=aes,og_idx=[2,3],predicate_list=preds[2:4],combine=True)
            ep34_opt = opt[0]
            ep34_eqb = eqb[0]

            ep_total_opt += ep34_opt
            ep_total_eqb += ep34_eqb

            selected34_opt = opt[9]
            selected34_eqb = eqb[9]

            preds34_before_opt = opt[13]
            preds34_before_eqb = eqb[13]

            preds34_after_opt = opt[14]
            preds34_after_eqb = eqb[14]

            #U in the middle
            # Q12 U Q34
            selected_total_opt = [selected12_opt,selected34_opt]
            selected_final_opt = union_test(selected_total_opt)

            selected_total_eqb = [selected12_eqb,selected34_eqb]
            selected_final_eqb = union_test(selected_total_eqb)
            
            preds_after_shift_opt = union_test([preds12_after_opt,preds34_after_opt]) 
            preds_before_shift_opt = union_test([preds12_before_opt,preds34_before_opt])

            preds_after_shift_eqb = union_test([preds12_after_eqb,preds34_after_eqb]) 
            preds_before_shift_eqb = union_test([preds12_before_eqb,preds34_before_eqb])
            
            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_opt,1,preds_before_shift_opt)
            positives = tp + fn
            negatives = tn + fp
            fps[3].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[3].append(len(fn)/len(positives) if len(positives)> 0 else 0)

            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_eqb,1,preds_before_shift_eqb)
            positives = tp + fn
            negatives = tn + fp
            fpNA[3].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fnNA[3].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[3].append(ep_total_opt)
            epNA[3].append(ep_total_eqb)
        else:
            opt,eqb,tn,tp = conj(counts[0:3],ts[0:3],failingRate,aes[0:3],ls[0:3],epsilonMax,iter=1,aes=aes,og_idx=[0,1,2],predicate_list=preds[0:3],combine=True)
            ep12_opt = opt[0]
            ep12_eqb = eqb[0]

            ep_total_opt += ep12_opt
            ep_total_eqb += ep12_eqb

            selected12_opt = opt[9]
            selected12_eqb = eqb[9]

            preds12_before_opt = opt[13]
            preds12_before_eqb = eqb[13]

            preds12_after_opt = opt[14]
            preds12_after_eqb = eqb[14]
            #SINGLE Q4
            bi = getBi(aes,aes[3],3,failingRate)
            bi_n = failingRate/4
            selected4_opt,ep4_opt = one_q(counts[3],ts[3],bi,aes[3],aes,3,ls[3],epsilonMax,iter=1)
            selected4_eqb,ep4_eqb = one_q_eq(counts[3],ts[3],bi_n,aes[3],aes,3,ls[3],epsilonMax,iter=1)
            
            preds4_opt = [preds[3][k] for k in selected4_opt]
            preds4_eqb = [preds[3][k] for k in selected4_eqb]

            ep_total_opt += ep4_opt
            

            ep_total_eqb += ep4_eqb
            

            # Q123 U Q4
            selected_total_opt = [selected12_opt,selected4_opt]
            selected_final_opt = union_test(selected_total_opt)

            selected_total_eqb = [selected12_eqb,selected4_eqb]
            selected_final_eqb = union_test(selected_total_eqb)
            
            preds_after_shift_opt = union_test([preds12_after_opt,preds4_opt]) 
            preds_before_shift_opt = union_test([preds12_before_opt,preds[3]])

            preds_after_shift_eqb = union_test([preds12_after_eqb,preds3_eqb]) 
            preds_before_shift_eqb = union_test([preds12_before_eqb,preds[3]])

            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_opt,0,preds_before_shift_opt)
            positives = tp + fn
            negatives = tn + fp
            fps[3].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[3].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[3].append(ep_total_opt)
            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_eqb,0,preds_before_shift_eqb)
            positives = tp + fn
            negatives = tn + fp
            fpNA[3].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fnNA[3].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            epNA[3].append(ep_total_eqb)

        #for n=5 we do N N U U / N N N U / N U U U
        ep_total_opt = 0
        ep_total_eqb = 0
        #randomly choose 4 queries
        combos = [list(c) for c in combinations(counts_total,5)]
        q_i = random.choice(combos)
        idx = [q[1] for q in q_i]
        counts = [counts_total[j][0] for j in idx]
        ts = [t_total[j] for j in idx]
        aes = [a_total[j] for j in idx]
        ls = [l_sensitivity_total[j] for j in idx]
        preds = [predicate_list[j] for j in idx]
        
        #each iteration randomly choose to run either N N U U / N N N U / N U U U
        flag = random.choice([0,1,2])
        if i < 50:
            flag = 0
        elif i < 100:
            flag = 1
        else:
            flag =2
        if flag == 0: # N N U U
            #first N
            opt,eqb,tn,tp = conj(counts[0:3],ts[0:3],failingRate,aes[0:3],ls[0:3],epsilonMax,iter=1,aes=aes,og_idx=[0,1,2],predicate_list=preds[0:3],combine=True)
            ep12_opt = opt[0]
            ep12_eqb = eqb[0]

            ep_total_opt += ep12_opt
            ep_total_eqb += ep12_eqb

            selected12_opt = opt[9]
            selected12_eqb = eqb[9]

            preds12_before_opt = opt[13]
            preds12_before_eqb = eqb[13]

            preds12_after_opt = opt[14]
            preds12_after_eqb = eqb[14]

            #last U 
            opt,eqb,tn,tp = disj(counts[3:5],ts[3:5],failingRate,aes[3:5],ls[3:5],epsilonMax,iter=1,aes=aes,og_idx=[3,4],predicate_list=preds[3:5],combine=True)
            ep34_opt = opt[0]
            ep34_eqb = eqb[0]

            ep_total_opt += ep34_opt
            ep_total_eqb += ep34_eqb
           

            selected34_opt = opt[9]
            selected34_eqb = eqb[9]

            preds34_before_opt = opt[13]
            preds34_before_eqb = eqb[13]

            preds34_after_opt = opt[14]
            preds34_after_eqb = eqb[14]

            #U in the middle
            # Q123 U Q45
            selected_total_opt = [selected12_opt,selected34_opt]
            selected_final_opt = union_test(selected_total_opt)

            selected_total_eqb = [selected12_eqb,selected34_eqb]
            selected_final_eqb = union_test(selected_total_eqb)
            
            preds_after_shift_opt = union_test([preds12_after_opt,preds34_after_opt]) 
            preds_before_shift_opt = union_test([preds12_before_opt,preds34_before_opt])

            preds_after_shift_eqb = union_test([preds12_after_eqb,preds34_after_eqb]) 
            preds_before_shift_eqb = union_test([preds12_before_eqb,preds34_before_eqb])
            
            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_opt,1,preds_before_shift_opt)
            positives = tp + fn
            negatives = tn + fp
            fps[4].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[4].append(len(fn)/len(positives) if len(positives)> 0 else 0)

            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_eqb,1,preds_before_shift_eqb)
            
            positives = tp + fn
            negatives = tn + fp
            fpNA[4].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fnNA[4].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[4].append(ep_total_opt)
            epNA[4].append(ep_total_eqb)
        elif flag == 1: # N N N U
            opt,eqb,tn,tp = conj(counts[0:4],ts[0:4],failingRate,aes[0:4],ls[0:4],epsilonMax,iter=1,aes=aes,og_idx=[0,1,2,3],predicate_list=preds[0:4],combine=True)
            ep12_opt = opt[0]
            ep12_eqb = eqb[0]

            ep_total_opt += ep12_opt
            ep_total_eqb += ep12_eqb

            selected12_opt = opt[9]
            selected12_eqb = eqb[9]

            preds12_before_opt = opt[13]
            preds12_before_eqb = eqb[13]

            preds12_after_opt = opt[14]
            preds12_after_eqb = eqb[14]
            #SINGLE Q4
            bi = getBi(aes,aes[4],4,failingRate)
            bi_n = failingRate/5
            selected4_opt,ep4_opt = one_q(counts[4],ts[4],bi,aes[4],aes,4,ls[4],epsilonMax,iter=1)
            selected4_eqb,ep4_eqb = one_q_eq(counts[4],ts[4],bi_n,aes[4],aes,4,ls[4],epsilonMax,iter=1)
            
            preds4_opt = [preds[4][k] for k in selected4_opt]
            preds4_eqb = [preds[4][k] for k in selected4_eqb]

            ep_total_opt += ep4_opt
            ep_total_eqb += ep4_eqb
            

            # Q123 U Q4
            selected_total_opt = [selected12_opt,selected4_opt]
            selected_final_opt = union_test(selected_total_opt)

            selected_total_eqb = [selected12_eqb,selected4_eqb]
            selected_final_eqb = union_test(selected_total_eqb)
            
            preds_after_shift_opt = union_test([preds12_after_opt,preds4_opt]) 
            preds_before_shift_opt = union_test([preds12_before_opt,preds[3]])

            preds_after_shift_eqb = union_test([preds12_after_eqb,preds3_eqb]) 
            preds_before_shift_eqb = union_test([preds12_before_eqb,preds[3]])

            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_opt,0,preds_before_shift_opt)
            positives = tp + fn
            negatives = tn + fp
            fps[4].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[4].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[4].append(ep_total_opt)
            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_eqb,0,preds_before_shift_eqb)
            positives = tp + fn
            negatives = tn + fp
            fpNA[4].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fnNA[4].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            epNA[4].append(ep_total_eqb)
        else: # N U U U
            #first N
            opt,eqb,tn,tp = conj(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],epsilonMax,iter=1,aes=aes,og_idx=[0,1],predicate_list=preds[0:2],combine=True)
            ep12_opt = opt[0]
            ep12_eqb = eqb[0]

            ep_total_opt += ep12_opt
            ep_total_eqb += ep12_eqb

            selected12_opt = opt[9]
            selected12_eqb = eqb[9]

            preds12_before_opt = opt[13]
            preds12_before_eqb = eqb[13]

            preds12_after_opt = opt[14]
            preds12_after_eqb = eqb[14]

            #last U 
            opt,eqb,tn,tp = disj(counts[2:5],ts[2:5],failingRate,aes[2:5],ls[2:5],epsilonMax,iter=1,aes=aes,og_idx=[2,3,4],predicate_list=preds[2:5],combine=True)
            ep34_opt = opt[0]
            ep34_eqb = eqb[0]

            ep_total_opt += ep34_opt
            ep_total_eqb += ep34_eqb

            selected34_opt = opt[9]
            selected34_eqb = eqb[9]

            preds34_before_opt = opt[13]
            preds34_before_eqb = eqb[13]

            preds34_after_opt = opt[14]
            preds34_after_eqb = eqb[14]

            #U in the middle
            # Q12 U Q345
            selected_total_opt = [selected12_opt,selected34_opt]
            selected_final_opt = union_test(selected_total_opt)

            selected_total_eqb = [selected12_eqb,selected34_eqb]
            selected_final_eqb = union_test(selected_total_eqb)
            
            preds_after_shift_opt = union_test([preds12_after_opt,preds34_after_opt]) 
            preds_before_shift_opt = union_test([preds12_before_opt,preds34_before_opt])

            preds_after_shift_eqb = union_test([preds12_after_eqb,preds34_after_eqb]) 
            preds_before_shift_eqb = union_test([preds12_before_eqb,preds34_before_eqb])
            
            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_opt,2,preds_before_shift_opt)
            positives = tp + fn
            negatives = tn + fp
            fps[4].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[4].append(len(fn)/len(positives) if len(positives)> 0 else 0)

            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_eqb,2,preds_before_shift_eqb)
            positives = tp + fn
            negatives = tn + fp
            fpNA[4].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fnNA[4].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[4].append(ep_total_opt)
            epNA[4].append(ep_total_eqb)
        #for n=6 we do N N N N U / N N N U U / N N U U U / N U U U U
        ep_total_opt = 0
        ep_total_eqb = 0
        #randomly choose 4 queries
        combos = [list(c) for c in combinations(counts_total,6)]
        q_i = random.choice(combos)
        idx = [q[1] for q in q_i]
        counts = [counts_total[j][0] for j in idx]
        ts = [t_total[j] for j in idx]
        aes = [a_total[j] for j in idx]
        ls = [l_sensitivity_total[j] for j in idx]
        preds = [predicate_list[j] for j in idx]
        #each iteration randomly choose to run either N N N N U / N N N U U / N N U U U / N U U U U
        flag = random.choice([0,1,2,3])
        if i < 37:
            flag = 0
        elif i < 74:
            flag = 1
        elif i <111:
            flag =2
        else:
            flag=3
        if flag == 0: # N N N N U
            opt,eqb,tn,tp = conj(counts[0:5],ts[0:5],failingRate,aes[0:5],ls[0:5],epsilonMax,iter=1,aes=aes,og_idx=[0,1,2,3,4],predicate_list=preds[0:5],combine=True)
            ep12_opt = opt[0]
            ep12_eqb = eqb[0]

            ep_total_opt += ep12_opt
            ep_total_eqb += ep12_eqb

            selected12_opt = opt[9]
            selected12_eqb = eqb[9]

            preds12_before_opt = opt[13]
            preds12_before_eqb = eqb[13]

            preds12_after_opt = opt[14]
            preds12_after_eqb = eqb[14]
            #SINGLE Q4
            bi = getBi(aes,aes[5],5,failingRate)
            bi_n = failingRate/6
            selected4_opt,ep4_opt = one_q(counts[5],ts[5],bi,aes[5],aes,5,ls[5],epsilonMax,iter=1)
            selected4_eqb,ep4_eqb = one_q_eq(counts[5],ts[5],bi_n,aes[5],aes,5,ls[5],epsilonMax,iter=1)
            
            preds4_opt = [preds[5][k] for k in selected4_opt]
            preds4_eqb = [preds[5][k] for k in selected4_eqb]

            ep_total_opt += ep4_opt
            ep_total_eqb += ep4_eqb
            # Q123 U Q4
            selected_total_opt = [selected12_opt,selected4_opt]
            selected_final_opt = union_test(selected_total_opt)

            selected_total_eqb = [selected12_eqb,selected4_eqb]
            selected_final_eqb = union_test(selected_total_eqb)
            
            preds_after_shift_opt = union_test([preds12_after_opt,preds4_opt]) 
            preds_before_shift_opt = union_test([preds12_before_opt,preds[3]])

            preds_after_shift_eqb = union_test([preds12_after_eqb,preds3_eqb]) 
            preds_before_shift_eqb = union_test([preds12_before_eqb,preds[3]])
            
            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_opt,0,preds_before_shift_opt)
            positives = tp + fn
            negatives = tn + fp
            fps[5].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[5].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[5].append(ep_total_opt)
            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_eqb,0,preds_before_shift_eqb)
            positives = tp + fn
            negatives = tn + fp
            fpNA[5].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fnNA[5].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            epNA[5].append(ep_total_eqb)
        elif flag ==1: # N N N U U
            #first N
            opt,eqb,tn,tp = conj(counts[0:4],ts[0:4],failingRate,aes[0:4],ls[0:4],epsilonMax,iter=1,aes=aes,og_idx=[0,1,2,3],predicate_list=preds[0:4],combine=True)
            ep12_opt = opt[0]
            ep12_eqb = eqb[0]

            ep_total_opt += ep12_opt
            ep_total_eqb += ep12_eqb

            selected12_opt = opt[9]
            selected12_eqb = eqb[9]

            preds12_before_opt = opt[13]
            preds12_before_eqb = eqb[13]

            preds12_after_opt = opt[14]
            preds12_after_eqb = eqb[14]

            #last U 
            opt,eqb,tn,tp = disj(counts[4:6],ts[4:6],failingRate,aes[4:6],ls[4:6],epsilonMax,iter=1,aes=aes,og_idx=[4,5],predicate_list=preds[4:6],combine=True)
            ep34_opt = opt[0]
            ep34_eqb = eqb[0]

            ep_total_opt += ep34_opt
            ep_total_eqb += ep34_eqb

            selected34_opt = opt[9]
            selected34_eqb = eqb[9]

            preds34_before_opt = opt[13]
            preds34_before_eqb = eqb[13]

            preds34_after_opt = opt[14]
            preds34_after_eqb = eqb[14]

            #U in the middle
            # Q123 U Q45
            selected_total_opt = [selected12_opt,selected34_opt]
            selected_final_opt = union_test(selected_total_opt)

            selected_total_eqb = [selected12_eqb,selected34_eqb]
            selected_final_eqb = union_test(selected_total_eqb)
            
            preds_after_shift_opt = union_test([preds12_after_opt,preds34_after_opt]) 
            preds_before_shift_opt = union_test([preds12_before_opt,preds34_before_opt])

            preds_after_shift_eqb = union_test([preds12_after_eqb,preds34_after_eqb]) 
            preds_before_shift_eqb = union_test([preds12_before_eqb,preds34_before_eqb])
            
            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_opt,1,preds_before_shift_opt)
            positives = tp + fn
            negatives = tn + fp
            fps[5].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[5].append(len(fn)/len(positives) if len(positives)> 0 else 0)

            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_eqb,1,preds_before_shift_eqb)
            
            positives = tp + fn
            negatives = tn + fp
            fpNA[5].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fnNA[5].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[5].append(ep_total_opt)
            epNA[5].append(ep_total_eqb)
        elif flag == 2: # N N U U U
         #first N
            opt,eqb,tn,tp = conj(counts[0:3],ts[0:3],failingRate,aes[0:3],ls[0:3],epsilonMax,iter=1,aes=aes,og_idx=[0,1,2],predicate_list=preds[0:3],combine=True)
            ep12_opt = opt[0]
            ep12_eqb = eqb[0]

            ep_total_opt += ep12_opt
            ep_total_eqb += ep12_eqb

            selected12_opt = opt[9]
            selected12_eqb = eqb[9]

            preds12_before_opt = opt[13]
            preds12_before_eqb = eqb[13]

            preds12_after_opt = opt[14]
            preds12_after_eqb = eqb[14]

            #last U 
            opt,eqb,tn,tp = disj(counts[3:6],ts[3:6],failingRate,aes[3:6],ls[3:6],epsilonMax,iter=1,aes=aes,og_idx=[3,4,5],predicate_list=preds[3:6],combine=True)
            ep34_opt = opt[0]
            ep34_eqb = eqb[0]

            ep_total_opt += ep34_opt
            ep_total_eqb += ep34_eqb
            

            selected34_opt = opt[9]
            selected34_eqb = eqb[9]

            preds34_before_opt = opt[13]
            preds34_before_eqb = eqb[13]

            preds34_after_opt = opt[14]
            preds34_after_eqb = eqb[14]

            #U in the middle
            # Q123 U Q45
            selected_total_opt = [selected12_opt,selected34_opt]
            selected_final_opt = union_test(selected_total_opt)

            selected_total_eqb = [selected12_eqb,selected34_eqb]
            selected_final_eqb = union_test(selected_total_eqb)
            
            preds_after_shift_opt = union_test([preds12_after_opt,preds34_after_opt]) 
            preds_before_shift_opt = union_test([preds12_before_opt,preds34_before_opt])

            preds_after_shift_eqb = union_test([preds12_after_eqb,preds34_after_eqb]) 
            preds_before_shift_eqb = union_test([preds12_before_eqb,preds34_before_eqb])
            
            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_opt,2,preds_before_shift_opt)
            positives = tp + fn
            negatives = tn + fp
            fps[5].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[5].append(len(fn)/len(positives) if len(positives)> 0 else 0)

            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_eqb,2,preds_before_shift_eqb)
            positives = tp + fn
            negatives = tn + fp
            fpNA[5].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fnNA[5].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[5].append(ep_total_opt)
            epNA[5].append(ep_total_eqb)
        else: # N U U U U
           #first N
            opt,eqb,tn,tp = conj(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],epsilonMax,iter=1,aes=aes,og_idx=[0,1],predicate_list=preds[0:2],combine=True)
            ep12_opt = opt[0]
            ep12_eqb = eqb[0]

            ep_total_opt += ep12_opt
            ep_total_eqb += ep12_eqb

            selected12_opt = opt[9]
            selected12_eqb = eqb[9]

            preds12_before_opt = opt[13]
            preds12_before_eqb = eqb[13]

            preds12_after_opt = opt[14]
            preds12_after_eqb = eqb[14]

            #last U 
            opt,eqb,tn,tp = disj(counts[2:6],ts[2:6],failingRate,aes[2:6],ls[2:6],epsilonMax,iter=1,aes=aes,og_idx=[2,3,4,5],predicate_list=preds[2:6],combine=True)
            ep34_opt = opt[0]
            ep34_eqb = eqb[0]

            ep_total_opt += ep34_opt
            ep_total_eqb += ep34_eqb
            

            selected34_opt = opt[9]
            selected34_eqb = eqb[9]

            preds34_before_opt = opt[13]
            preds34_before_eqb = eqb[13]

            preds34_after_opt = opt[14]
            preds34_after_eqb = eqb[14]

            #U in the middle
            # Q123 U Q45
            selected_total_opt = [selected12_opt,selected34_opt]
            selected_final_opt = union_test(selected_total_opt)

            selected_total_eqb = [selected12_eqb,selected34_eqb]
            selected_final_eqb = union_test(selected_total_eqb)
            
            preds_after_shift_opt = union_test([preds12_after_opt,preds34_after_opt]) 
            preds_before_shift_opt = union_test([preds12_before_opt,preds34_before_opt])

            preds_after_shift_eqb = union_test([preds12_after_eqb,preds34_after_eqb]) 
            preds_before_shift_eqb = union_test([preds12_before_eqb,preds34_before_eqb])
            
            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_opt,3,preds_before_shift_opt)
            positives = tp + fn
            negatives = tn + fp
            fps[5].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[5].append(len(fn)/len(positives) if len(positives)> 0 else 0)

            tp,tn,fp,fn = classify_mix(counts,ts,selected_final_eqb,3,preds_before_shift_eqb)
            positives = tp + fn
            negatives = tn + fp
            fpNA[5].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fnNA[5].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[5].append(ep_total_opt)
            epNA[5].append(ep_total_eqb)
    fns = [mean(fnr) for fnr in fns]
    fps = [mean(fpr) for fpr in fps]
    eps = [mean(ep) for ep in eps]
    epNA= [mean(ep) for ep in epNA]
    fnNA= [mean(fnr) for fnr in fnNA]
    fpNA= [mean(fpr) for fpr in fpNA]
    return fns,fps,eps,fnNA,fpNA,epNA


def disj(counts_total, t_total, failingRate,a_total,l_sensitivity_total, epsilonMax, iter=100,aes=[],og_idx=[], alt=False,predicate_list=[],combine=False):
    b_total = []
    ep_list = []
    ep_i_lists = [[0]*iter for i in range(len(counts_total))]
    fn_i_lists = [[0]*iter for i in range(len(counts_total))]
    fp_i_lists = [[0]*iter for i in range(len(counts_total))]
    fp_list = []
    fn_list= []
    selected_returned = []
    positives,negatives = [],[]
    for i,a in enumerate(a_total) :
        if combine == True:
            prod = get_numerator(a_total[i],og_idx[i],aes)*l_sensitivity_total[i]
            bi = (prod/get_denominator(aes,alt,og_idx[i],s=l_sensitivity_total))*failingRate
            b_total.append(bi)
        else:
            prod = get_numerator(a,i,a_total)*l_sensitivity_total[i]
            bi = (prod/get_denominator(a_total,alt,i,s=l_sensitivity_total))*failingRate
            b_total.append(bi)

    b__total = [failingRate/len(aes)]*len(a_total) if combine==True else [failingRate/len(a_total)]*len(a_total) 
    preds_after_shift,preds_before_shift = [],[]
    for i in range(0,iter):
        ep_total = 0
        selected_total = []
        predicate_total = []
        for j,count in enumerate(counts_total):
            selected_j, ep_j = threshold_shift(count, [t_total[j]]*len(count),a_total[j],b_total[j],epsilonMax,l_sensitivity_total[j])
            ep_i_lists[j][i]=ep_j
            ep_total+= ep_j
            selected_total.append(selected_j)
            if len(predicate_list) > 0 :
                if len(predicate_list[j]) > 0:
                    predicate_total.append([predicate_list[j][p] for p in selected_j])    
        if(ep_total > epsilonMax):
            print('Query Denied')
            return [[],[],0,0]  
        ep_list.append(ep_total)
        selected = union_test(selected_total)
        preds_after_shift = union_test(predicate_total) 
        preds_before_shift = union_test(predicate_list)
        selected_returned = selected
        for j,count in enumerate(counts_total):
            tp_i,tn_i,fp_i,fn_i = classify(count,[t_total[j]]*len(count),selected_total[j])
            fn_i_lists[j][i] = len(fn_i)/len(count)
            fp_i_lists[j][i] = len(fp_i)/len(count)

        tp,tn,fp,fn= new_classify_or(counts_total, [[t_total[i]]*len(counts_total[i]) for i in range(len(t_total))],selected,preds_after_shift,preds_before_shift,predicate_list)
        positives = tp + fn
        negatives = tn + fp
        fp_list.append(len(fp)/len(negatives) if len(negatives) > 0 else 0)
        if len(get_union(counts_total,t_total)) > 0:
            fn_list.append(len(fn)/len(get_union(counts_total,t_total)))
        else: fn_list.append(0)
   
    param_optb = [mean(ep_list)]
    means = []
    for i in range(len(ep_i_lists)):
        means.append(mean(ep_i_lists[i]))
    param_optb.append(means)
    param_optb.append(mean(fn_list))
    means = []
    for i in range(len(fn_i_lists)):
        means.append(mean(fn_i_lists[i]))
    param_optb.append(means)
    
    param_optb.append(mean(fp_list))
    means = []
    for i in range(len(fp_i_lists)):
        means.append(mean(fp_i_lists[i]))
    param_optb.append(means)
    param_optb.append(ep_list)
    param_optb.append(fn_list)
    param_optb.append(fp_list)
    param_optb.append(selected_returned)
    param_optb.append(b_total)
    param_optb.append(positives)
    param_optb.append(negatives)
    param_optb.append(preds_before_shift)
    param_optb.append(preds_after_shift)
    ep_list = []
    ep_i_lists = [[0]*iter for i in range(len(counts_total))]
    fn_i_lists = [[0]*iter for i in range(len(counts_total))]
    fp_i_lists = [[0]*iter for i in range(len(counts_total))]
    fp_list = []
    fn_list= []
    preds_after_shift,preds_before_shift = [],[]
    for i in range(0,iter):
        ep_total = 0
        selected_total = []
        predicate_total = []
        for j,count in enumerate(counts_total):
            selected_j, ep_j = threshold_shift(count, [t_total[j]]*len(count),a_total[j],b__total[j],epsilonMax,l_sensitivity_total[j])
            ep_i_lists[j][i]=ep_j
            ep_total+= ep_j
            selected_total.append(selected_j)
            if len(predicate_list) > 0 :
                if len(predicate_list[j]) > 0:
                    predicate_total.append([predicate_list[j][p] for p in selected_j])   
        ep_list.append(ep_total)
        selected = union_test(selected_total)
        preds_after_shift = union_test(predicate_total) 
        preds_before_shift = union_test(predicate_list)
        selected_returned = selected

        for j,count in enumerate(counts_total):
            tp_i,tn_i,fp_i,fn_i = classify(count,[t_total[j]]*len(count),selected_total[j])
            fn_i_lists[j][i] = len(fn_i)/len(count)
            fp_i_lists[j][i] = len(fp_i)/len(count)
        
        tp,tn,fp,fn= new_classify_or(counts_total, [[t_total[i]]*len(counts_total[i]) for i in range(len(t_total))],selected,preds_after_shift,preds_before_shift,predicate_list)
        fp_list.append(len(fp)/len(negatives) if len(negatives) > 0 else 0)

        if len(get_union(counts_total,t_total)) > 0:
            fn_list.append(len(fn)/len(get_union(counts_total,t_total)))
        else: fn_list.append(0)
    
    param_eqb = [mean(ep_list)]
    means = []
    for i in range(len(ep_i_lists)):
        means.append(mean(ep_i_lists[i]))
    param_eqb.append(means)
  
    param_eqb.append(mean(fn_list))
    means = []
    for i in range(len(fn_i_lists)):
        means.append(mean(fn_i_lists[i]))
    param_eqb.append(means)
    
    param_eqb.append(mean(fp_list))
    means = []
    for i in range(len(fp_i_lists)):
        means.append(mean(fp_i_lists[i]))
    param_eqb.append(means)
    param_eqb.append(ep_list)
    param_eqb.append(fn_list)
    param_eqb.append(fp_list)
    param_eqb.append(selected_returned)
    param_eqb.append(b_total)
    param_eqb.append(positives)
    param_eqb.append(negatives)
    param_eqb.append(preds_before_shift)
    param_eqb.append(preds_after_shift)

    return param_optb,param_eqb,len(tn),len(tp)

def conj(counts_total, t_total, failingRate,a_total,l_sensitivity_total, epsilonMax, iter=100,aes=[],og_idx=[], alt=False,predicate_list=[],combine=False):
    b_total = []
    ep_list = []
    ep_i_lists = [[0]*iter for i in range(len(counts_total))]
    fn_i_lists = [[0]*iter for i in range(len(counts_total))]
    fp_i_lists = [[0]*iter for i in range(len(counts_total))]
    fp_list = []
    fn_list= []
    selected_returned = []
    positives, negatives = [],[]
    for i,a in enumerate(a_total) :
        if combine == True:
            prod = get_numerator(a_total[i],og_idx[i],aes)*l_sensitivity_total[i]
            bi = (prod/get_denominator(aes,alt,og_idx[i],s=l_sensitivity_total))*failingRate
            b_total.append(bi)
        else:
            prod = get_numerator(a,i,a_total)*l_sensitivity_total[i]
            bi = (prod/get_denominator(a_total,alt,i,s=l_sensitivity_total))*failingRate
            b_total.append(bi)

    b__total = [failingRate/len(aes)]*len(a_total) if combine==True else [failingRate/len(a_total)]*len(a_total) 
    preds_after_shift, preds_before_shift = [],[]  
    for i in range(0,iter):
        ep_total = 0
        selected_total = []
        early_stop = False
        predicate_total = []
        stop_j = 0
        for j,count in enumerate(counts_total):
            selected_j, ep_j = threshold_shift(count, [t_total[j]]*len(count),a_total[j],b_total[j],epsilonMax,l_sensitivity_total[j])
            ep_i_lists[j][i]=ep_j
            ep_total+= ep_j
            selected_total.append(selected_j)
            if len(predicate_list) > 0 :
                if len(predicate_list[j]) > 0:
                    predicate_total.append([predicate_list[j][p] for p in selected_j])

            if(len(selected_j) == 0):
                early_stop = True
                stop_j = j
                break
   
        if(ep_total > epsilonMax):
            print('Query Denied')
            return [[],[],0,0]  
            
        ep_list.append(ep_total)
        selected = intersection_test(selected_total)
        selected_returned = selected
        preds_after_shift = intersection_test(predicate_total)
        preds_before_shift = intersection_test(predicate_list)

        if(early_stop):
            tp,tn,fp,fn= new_classify_and(counts_total,[[t_total[i]]*len(counts_total[i]) for i in range(len(t_total))],selected,preds_after_shift,preds_before_shift,predicate_list)
            positives = tp + fn
            negatives = tn + fp
            fp_list.append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fn_list.append(len(fn)/len(positives) if len(positives)> 0 else 0)
            break
             
        for j,count in enumerate(counts_total):
            tp_i,tn_i,fp_i,fn_i = classify(count,[t_total[j]]*len(count),selected_total[j])
            fn_i_lists[j][i] = len(fn_i)/len(count)
            fp_i_lists[j][i] = len(fp_i)/len(count)

        tp,tn,fp,fn = new_classify_and(counts_total,[[t_total[i]]*len(counts_total[i]) for i in range(len(t_total))],selected,preds_after_shift,preds_before_shift,predicate_list)
        positives = tp + fn
        negatives = tn + fp
        fp_list.append(len(fp)/len(negatives) if len(negatives) > 0 else 0)
        fn_list.append(len(fn)/len(positives) if len(positives)>0 else 0)

   
    param_optb = [mean(ep_list)]
    means = []
    for i in range(len(ep_i_lists)):
        means.append(mean(ep_i_lists[i]))
    param_optb.append(means)
    param_optb.append(mean(fn_list))
    means = []
    for i in range(len(fn_i_lists)):
        means.append(mean(fn_i_lists[i]))
    param_optb.append(means)
    
    param_optb.append(mean(fp_list))
    means = []
    for i in range(len(fp_i_lists)):
        means.append(mean(fp_i_lists[i]))
    param_optb.append(means)
    param_optb.append(ep_list)
    param_optb.append(fn_list)
    param_optb.append(fp_list)
    param_optb.append(selected_returned)
    param_optb.append(b_total)
    param_optb.append(positives)
    param_optb.append(negatives)
    param_optb.append(preds_before_shift)
    param_optb.append(preds_after_shift)

    ep_list = []
    ep_i_lists = [[0]*iter for i in range(len(counts_total))]
    fn_i_lists = [[0]*iter for i in range(len(counts_total))]
    fp_i_lists = [[0]*iter for i in range(len(counts_total))]
    fp_list = []
    fn_list= []
    preds_after_shift, preds_before_shift = [],[]
    for i in range(0,iter):
        ep_total = 0
        selected_total = []
        predicate_total = []
        for j,count in enumerate(counts_total):
            selected_j, ep_j = threshold_shift(count, [t_total[j]]*len(count),a_total[j],b__total[j],epsilonMax,l_sensitivity_total[j])
            ep_i_lists[j][i]=ep_j
            ep_total+= ep_j
            selected_total.append(selected_j)
            if len(predicate_list) > 0 :
                if len(predicate_list[j]) > 0:
                    predicate_total.append([predicate_list[j][p] for p in selected_j])

        if(ep_total > epsilonMax):
            print('Query Denied')
            return [[],[],0,0]  
            
        ep_list.append(ep_total)
        selected = intersection_test(selected_total)
        selected_returned = selected
        preds_after_shift = intersection_test(predicate_total)
        preds_before_shift = intersection_test(predicate_list)

        for j,count in enumerate(counts_total):
            tp_i,tn_i,fp_i,fn_i = classify(count,[t_total[j]]*len(count),selected_total[j])
            fn_i_lists[j][i] = len(fn_i)/len(count)
            fp_i_lists[j][i] = len(fp_i)/len(count)

        
        tp,tn,fp,fn = new_classify_and(counts_total,[[t_total[i]]*len(counts_total[i]) for i in range(len(t_total))],selected,preds_after_shift,preds_before_shift,predicate_list)
        fp_list.append(len(fp)/len(get_union_conj(counts_total,t_total)))

        if len(get_intersection_conj(counts_total,t_total)) > 0:
            fn_list.append(len(fn)/len(get_intersection_conj(counts_total,t_total)))
        else: fn_list.append(0)
    
    param_eqb = [mean(ep_list)]
    means = []
    for i in range(len(ep_i_lists)):
        means.append(mean(ep_i_lists[i]))
    param_eqb.append(means)
  
    param_eqb.append(mean(fn_list))
    means = []
    for i in range(len(fn_i_lists)):
        means.append(mean(fn_i_lists[i]))
    param_eqb.append(means)
    
    param_eqb.append(mean(fp_list))
    means = []
    for i in range(len(fp_i_lists)):
        means.append(mean(fp_i_lists[i]))
    param_eqb.append(means)
    param_eqb.append(ep_list)
    param_eqb.append(fn_list)
    param_eqb.append(fp_list)
    param_eqb.append(selected_returned)
    param_eqb.append(b_total)
    param_eqb.append(positives)
    param_eqb.append(negatives)
    param_eqb.append(preds_before_shift)
    param_eqb.append(preds_after_shift)

    return param_optb,param_eqb,len(tn),len(tp)

def query_sales(start_time,end_time,beta,u,type):
    pred1,counts1,th1 = sales_volume(start_time,end_time)
    pred2,counts2,th2 = average_transaction_value(start_time,end_time)
    pred3,counts3,th3 = distinct_customers(start_time,end_time)
    pred4,counts4,th4 = category_count(start_time,end_time)
    pred5,counts5,th5 = sales_volume_marmara(start_time,end_time)
    pred6,counts6,th6 = sales_volume_women(start_time,end_time)
    pred = [pred1,pred2,pred3,pred4,pred5,pred6]
    counts = [(counts1,0),(counts2,1),(counts3,2),(counts4,3),(counts5,4),(counts6,5)]
    th=[th1,th2,th3,th4,th5,th6]

    a=0.12 if u==-1 else u/100
    failingRate = 0.05 if beta==-1 else beta
    
    epsilonMax= 20
    iter=100

    #For each query set the sensitivity, uncertain region
    a1= a*(max(counts1)-min(counts1)) 
    a2= a*(max(counts2)-min(counts2)) 
    a3 = a*(max(counts3)-min(counts3))
    a4= a*(max(counts4)-min(counts4))
    a5= a*(max(counts5)-min(counts5))
    a6= a*(max(counts6)-min(counts6))   
    a_total = [a1,a2,a3,a4,a5,a6]

    l = [1]*len(a_total)
    
    if type == 2:
        fns,fps,eps,fnNA,fpNA,epNA = combine(counts,th,failingRate,a_total,l,epsilonMax,pred,vary=True)
        entropyTSLM,entropyNaive = getEntropyArr(epNA,eps,max([len(pr) for pr in pred]))
        return eps,fns,fps,entropyTSLM,epNA,fnNA,fpNA,entropyNaive
    eps,fns,fps,epNA,fnNA,fpNA = [[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)]
    for i in range(iter):
        if type == 0: ##disj
            oneQ = random.choice(counts)
            opt,eqb,tn1,tp1 = disj([oneQ[0]],[th[oneQ[1]]],failingRate,[a_total[oneQ[1]]],[l[oneQ[1]]],epsilonMax,iter=5,predicate_list=[pred[oneQ[1]]])
            
            twoQ = counts[0:2]
            random.shuffle(twoQ)
            opt1,eqb1,tn2,tp2 = disj([twoQ[0][0],twoQ[1][0]],[th[twoQ[0][1]],th[twoQ[1][1]]],failingRate,[a_total[twoQ[0][1]],a_total[twoQ[1][1]]],[l[twoQ[0][1]],l[twoQ[1][1]]],epsilonMax,iter=5,predicate_list=[pred[twoQ[0][1]],pred[twoQ[1][1]]])
            
            threeQ = counts[0:3]
            random.shuffle(threeQ)
            opt2,eqb2,tn,tp = disj([threeQ[0][0],threeQ[1][0],threeQ[2][0]],[th[threeQ[0][1]],th[threeQ[1][1]], th[threeQ[2][1]]],failingRate,[a_total[threeQ[0][1]],a_total[threeQ[1][1]],a_total[threeQ[2][1]]],[l[threeQ[0][1]],l[threeQ[1][1]],l[threeQ[2][1]]],epsilonMax,iter=5,predicate_list=[pred[threeQ[0][1]],pred[threeQ[1][1]],pred[threeQ[2][1]]])
            
            fourQ = counts[0:4]
            random.shuffle(fourQ)
            opt3,eqb3,tn,tp = disj([fourQ[0][0],fourQ[1][0],fourQ[2][0],fourQ[3][0]],[th[fourQ[0][1]],th[fourQ[1][1]], th[fourQ[2][1]], th[fourQ[3][1]]],failingRate,[a_total[fourQ[0][1]],a_total[fourQ[1][1]],a_total[fourQ[2][1]],a_total[fourQ[3][1]]],[l[fourQ[0][1]],l[fourQ[1][1]],l[fourQ[2][1]],l[fourQ[3][1]]],epsilonMax,iter=5,predicate_list=[pred[fourQ[0][1]],pred[fourQ[1][1]],pred[fourQ[2][1]],pred[fourQ[3][1]]])
            
            fiveQ = counts[0:5]
            random.shuffle(fiveQ)
            opt4,eqb4,tn,tp = disj([fiveQ[0][0],fiveQ[1][0],fiveQ[2][0],fiveQ[3][0],fiveQ[4][0]],[th[fiveQ[0][1]],th[fiveQ[1][1]], th[fiveQ[2][1]], th[fiveQ[3][1]], th[fiveQ[4][1]]],failingRate,[a_total[fiveQ[0][1]],a_total[fiveQ[1][1]],a_total[fiveQ[2][1]],a_total[fiveQ[3][1]],a_total[fiveQ[4][1]]],[l[fiveQ[0][1]],l[fiveQ[1][1]],l[fiveQ[2][1]],l[fiveQ[3][1]],l[fiveQ[4][1]]],epsilonMax,iter=5,predicate_list=[pred[fiveQ[0][1]],pred[fiveQ[1][1]],pred[fiveQ[2][1]],pred[fiveQ[3][1]],pred[fiveQ[4][1]]])
            
            sixQ = counts
            random.shuffle(sixQ)
            opt5,eqb5,tn,tp = disj([sixQ[0][0],sixQ[1][0],sixQ[2][0],sixQ[3][0],sixQ[4][0],sixQ[5][0]],[th[sixQ[0][1]],th[sixQ[1][1]], th[sixQ[2][1]], th[sixQ[3][1]], th[sixQ[4][1]], th[sixQ[5][1]]],failingRate,[a_total[sixQ[0][1]],a_total[sixQ[1][1]],a_total[sixQ[2][1]],a_total[sixQ[3][1]],a_total[sixQ[4][1]],a_total[sixQ[5][1]]],[l[sixQ[0][1]],l[sixQ[1][1]],l[sixQ[2][1]],l[sixQ[3][1]],l[sixQ[4][1]],l[sixQ[5][1]]],epsilonMax,iter=5,predicate_list=[pred[sixQ[0][1]],pred[sixQ[1][1]],pred[sixQ[2][1]],pred[sixQ[3][1]],pred[sixQ[4][1]],pred[sixQ[5][1]]])
            
            eps[0].append(opt[0])
            epNA[0].append(eqb[0])
            fps[0].append(opt[4])
            fpNA[0].append(eqb[4])
            fns[0].append(opt[2])
            fnNA[0].append(eqb[2])

            eps[1].append(opt1[0])
            epNA[1].append(eqb1[0])
            fps[1].append(opt1[4])
            fpNA[1].append(eqb1[4])
            fns[1].append(opt1[2])
            fnNA[1].append(eqb1[2])

            eps[2].append(opt2[0])
            epNA[2].append(eqb2[0])
            fps[2].append(opt2[4])
            fpNA[2].append(eqb2[4])
            fns[2].append(opt2[2])
            fnNA[2].append(eqb2[2])

            eps[3].append(opt3[0])
            epNA[3].append(eqb3[0])
            fps[3].append(opt3[4])
            fpNA[3].append(eqb3[4])
            fns[3].append(opt3[2])
            fnNA[3].append(eqb3[2])

            eps[4].append(opt4[0])
            epNA[4].append(eqb4[0])
            fps[4].append(opt4[4])
            fpNA[4].append(eqb4[4])
            fns[4].append(opt4[2])
            fnNA[4].append(eqb4[2])

            eps[5].append(opt5[0])
            epNA[5].append(eqb5[0])
            fps[5].append(opt5[4])
            fpNA[5].append(eqb5[4])
            fns[5].append(opt5[2])
            fnNA[5].append(eqb5[2])
        else:
            oneQ = random.choice(counts)
            opt,eqb,tn1,tp1 = disj([oneQ[0]],[th[oneQ[1]]],failingRate,[a_total[oneQ[1]]],[l[oneQ[1]]],epsilonMax,iter=5,predicate_list=[pred[oneQ[1]]])
            
            twoQ = counts[0:2]
            random.shuffle(twoQ)
            opt1,eqb1,tn2,tp2 = disj([twoQ[0][0],twoQ[1][0]],[th[twoQ[0][1]],th[twoQ[1][1]]],failingRate,[a_total[twoQ[0][1]],a_total[twoQ[1][1]]],[l[twoQ[0][1]],l[twoQ[1][1]]],epsilonMax,iter=5,predicate_list=[pred[twoQ[0][1]],pred[twoQ[1][1]]])
            
            threeQ = counts[0:3]
            random.shuffle(threeQ)
            opt2,eqb2,tn,tp = disj([threeQ[0][0],threeQ[1][0],threeQ[2][0]],[th[threeQ[0][1]],th[threeQ[1][1]], th[threeQ[2][1]]],failingRate,[a_total[threeQ[0][1]],a_total[threeQ[1][1]],a_total[threeQ[2][1]]],[l[threeQ[0][1]],l[threeQ[1][1]],l[threeQ[2][1]]],epsilonMax,iter=5,predicate_list=[pred[threeQ[0][1]],pred[threeQ[1][1]],pred[threeQ[2][1]]])
            
            fourQ = counts[0:4]
            random.shuffle(fourQ)
            opt3,eqb3,tn,tp = disj([fourQ[0][0],fourQ[1][0],fourQ[2][0],fourQ[3][0]],[th[fourQ[0][1]],th[fourQ[1][1]], th[fourQ[2][1]], th[fourQ[3][1]]],failingRate,[a_total[fourQ[0][1]],a_total[fourQ[1][1]],a_total[fourQ[2][1]],a_total[fourQ[3][1]]],[l[fourQ[0][1]],l[fourQ[1][1]],l[fourQ[2][1]],l[fourQ[3][1]]],epsilonMax,iter=5,predicate_list=[pred[fourQ[0][1]],pred[fourQ[1][1]],pred[fourQ[2][1]],pred[fourQ[3][1]]])
            
            fiveQ = counts[0:5]
            random.shuffle(fiveQ)
            opt4,eqb4,tn,tp = disj([fiveQ[0][0],fiveQ[1][0],fiveQ[2][0],fiveQ[3][0],fiveQ[4][0]],[th[fiveQ[0][1]],th[fiveQ[1][1]], th[fiveQ[2][1]], th[fiveQ[3][1]], th[fiveQ[4][1]]],failingRate,[a_total[fiveQ[0][1]],a_total[fiveQ[1][1]],a_total[fiveQ[2][1]],a_total[fiveQ[3][1]],a_total[fiveQ[4][1]]],[l[fiveQ[0][1]],l[fiveQ[1][1]],l[fiveQ[2][1]],l[fiveQ[3][1]],l[fiveQ[4][1]]],epsilonMax,iter=5,predicate_list=[pred[fiveQ[0][1]],pred[fiveQ[1][1]],pred[fiveQ[2][1]],pred[fiveQ[3][1]],pred[fiveQ[4][1]]])
            
            sixQ = counts
            random.shuffle(sixQ)
            opt5,eqb5,tn,tp = disj([sixQ[0][0],sixQ[1][0],sixQ[2][0],sixQ[3][0],sixQ[4][0],sixQ[5][0]],[th[sixQ[0][1]],th[sixQ[1][1]], th[sixQ[2][1]], th[sixQ[3][1]], th[sixQ[4][1]], th[sixQ[5][1]]],failingRate,[a_total[sixQ[0][1]],a_total[sixQ[1][1]],a_total[sixQ[2][1]],a_total[sixQ[3][1]],a_total[sixQ[4][1]],a_total[sixQ[5][1]]],[l[sixQ[0][1]],l[sixQ[1][1]],l[sixQ[2][1]],l[sixQ[3][1]],l[sixQ[4][1]],l[sixQ[5][1]]],epsilonMax,iter=5,predicate_list=[pred[sixQ[0][1]],pred[sixQ[1][1]],pred[sixQ[2][1]],pred[sixQ[3][1]],pred[sixQ[4][1]],pred[sixQ[5][1]]])
            
            eps[0].append(opt[0])
            epNA[0].append(eqb[0])
            fps[0].append(opt[4])
            fpNA[0].append(eqb[4])
            fns[0].append(opt[2])
            fnNA[0].append(eqb[2])

            eps[1].append(opt1[0])
            epNA[1].append(eqb1[0])
            fps[1].append(opt1[4])
            fpNA[1].append(eqb1[4])
            fns[1].append(opt1[2])
            fnNA[1].append(eqb1[2])

            eps[2].append(opt2[0])
            epNA[2].append(eqb2[0])
            fps[2].append(opt2[4])
            fpNA[2].append(eqb2[4])
            fns[2].append(opt2[2])
            fnNA[2].append(eqb2[2])

            eps[3].append(opt3[0])
            epNA[3].append(eqb3[0])
            fps[3].append(opt3[4])
            fpNA[3].append(eqb3[4])
            fns[3].append(opt3[2])
            fnNA[3].append(eqb3[2])

            eps[4].append(opt4[0])
            epNA[4].append(eqb4[0])
            fps[4].append(opt4[4])
            fpNA[4].append(eqb4[4])
            fns[4].append(opt4[2])
            fnNA[4].append(eqb4[2])

            eps[5].append(opt5[0])
            epNA[5].append(eqb5[0])
            fps[5].append(opt5[4])
            fpNA[5].append(eqb5[4])
            fns[5].append(opt5[2])
            fnNA[5].append(eqb5[2])      
    fns = [mean(fnr) for fnr in fns]
    fps = [mean(fpr) for fpr in fps]
    eps = [mean(ep) for ep in eps]
    epNA= [mean(ep) for ep in epNA]
    fnNA= [mean(fnr) for fnr in fnNA]
    fpNA= [mean(fpr) for fpr in fpNA]
    entropyTSLM,entropyNaive = getEntropyArr(epNA,eps,max([len(pr) for pr in pred]))
    return eps,fns,fps,entropyTSLM,epNA,fnNA,fpNA,entropyNaive

def query_taxi(start_time,end_time,beta,u,type): #n qs where q is conj/disj of two queries
    pred1,counts1,tth1=query_taxi_count(start_time, end_time)
    pred2,counts2,tth2=query_taxi_fareamount(start_time, end_time)
    pred3,counts3,tth3=query_taxi_total_amount(start_time, end_time)
    pred4,counts4,tth4=query_taxi_count_flag(start_time, end_time)
    pred5,counts5,tth5=query_taxi_count_payment_type(start_time, end_time)
    pred6,counts6,tth6=query_taxi_count_location(start_time, end_time)
    pred = [pred1,pred2,pred3,pred4,pred5,pred6]
    counts = [(counts1,0),(counts2,1),(counts3,2),(counts4,3),(counts5,4),(counts6,5)]
    th=[tth1,tth2,tth3,tth4,tth5,tth6]
    a=0.12
    failingRate = 0.05 if beta==-1 else beta

    a1= a*(max(counts1)-min(counts1))   
    a2= a*(max(counts2)-min(counts2))
    a3= a*(max(counts3)-min(counts3))
    a4= a*(max(counts4)-min(counts4))
    a5= a*(max(counts5)-min(counts5))
    a6= a*(max(counts6)-min(counts6))
    a_total = [a1,a2,a3,a4,a5,a6]

    l = [1]*len(a_total)
    epsilonMax= 9
    iter=100

    if type == 2:
        fns,fps,eps,fnNA,fpNA,epNA = combine(counts,th,failingRate,a_total,l,epsilonMax,pred)
        entropyTSLM,entropyNaive = getEntropyArr(epNA,eps,max([len(pr) for pr in pred]))
        return eps,fns,fps,entropyTSLM,epNA,fnNA,fpNA,entropyNaive

    eps,fns,fps,epNA,fnNA,fpNA = [[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)]
    for i in range(iter):
        if type == 0: ##disj
            oneQ = random.choice(counts)
            opt,eqb,tn1,tp1 = disj([oneQ[0]],[th[oneQ[1]]],failingRate,[a_total[oneQ[1]]],[l[oneQ[1]]],epsilonMax,iter=5,predicate_list=[pred[oneQ[1]]])
            
            twoQ = counts[0:2]
            random.shuffle(twoQ)
            opt1,eqb1,tn2,tp2 = disj([twoQ[0][0],twoQ[1][0]],[th[twoQ[0][1]],th[twoQ[1][1]]],failingRate,[a_total[twoQ[0][1]],a_total[twoQ[1][1]]],[l[twoQ[0][1]],l[twoQ[1][1]]],epsilonMax,iter=5,predicate_list=[pred[twoQ[0][1]],pred[twoQ[1][1]]])
            
            threeQ = counts[0:3]
            random.shuffle(threeQ)
            opt2,eqb2,tn,tp = disj([threeQ[0][0],threeQ[1][0],threeQ[2][0]],[th[threeQ[0][1]],th[threeQ[1][1]], th[threeQ[2][1]]],failingRate,[a_total[threeQ[0][1]],a_total[threeQ[1][1]],a_total[threeQ[2][1]]],[l[threeQ[0][1]],l[threeQ[1][1]],l[threeQ[2][1]]],epsilonMax,iter=5,predicate_list=[pred[threeQ[0][1]],pred[threeQ[1][1]],pred[threeQ[2][1]]])
            
            fourQ = counts[0:4]
            random.shuffle(fourQ)
            opt3,eqb3,tn,tp = disj([fourQ[0][0],fourQ[1][0],fourQ[2][0],fourQ[3][0]],[th[fourQ[0][1]],th[fourQ[1][1]], th[fourQ[2][1]], th[fourQ[3][1]]],failingRate,[a_total[fourQ[0][1]],a_total[fourQ[1][1]],a_total[fourQ[2][1]],a_total[fourQ[3][1]]],[l[fourQ[0][1]],l[fourQ[1][1]],l[fourQ[2][1]],l[fourQ[3][1]]],epsilonMax,iter=5,predicate_list=[pred[fourQ[0][1]],pred[fourQ[1][1]],pred[fourQ[2][1]],pred[fourQ[3][1]]])
            
            fiveQ = counts[0:5]
            random.shuffle(fiveQ)
            opt4,eqb4,tn,tp = disj([fiveQ[0][0],fiveQ[1][0],fiveQ[2][0],fiveQ[3][0],fiveQ[4][0]],[th[fiveQ[0][1]],th[fiveQ[1][1]], th[fiveQ[2][1]], th[fiveQ[3][1]], th[fiveQ[4][1]]],failingRate,[a_total[fiveQ[0][1]],a_total[fiveQ[1][1]],a_total[fiveQ[2][1]],a_total[fiveQ[3][1]],a_total[fiveQ[4][1]]],[l[fiveQ[0][1]],l[fiveQ[1][1]],l[fiveQ[2][1]],l[fiveQ[3][1]],l[fiveQ[4][1]]],epsilonMax,iter=5,predicate_list=[pred[fiveQ[0][1]],pred[fiveQ[1][1]],pred[fiveQ[2][1]],pred[fiveQ[3][1]],pred[fiveQ[4][1]]])
            
            sixQ = counts
            random.shuffle(sixQ)
            opt5,eqb5,tn,tp = disj([sixQ[0][0],sixQ[1][0],sixQ[2][0],sixQ[3][0],sixQ[4][0],sixQ[5][0]],[th[sixQ[0][1]],th[sixQ[1][1]], th[sixQ[2][1]], th[sixQ[3][1]], th[sixQ[4][1]], th[sixQ[5][1]]],failingRate,[a_total[sixQ[0][1]],a_total[sixQ[1][1]],a_total[sixQ[2][1]],a_total[sixQ[3][1]],a_total[sixQ[4][1]],a_total[sixQ[5][1]]],[l[sixQ[0][1]],l[sixQ[1][1]],l[sixQ[2][1]],l[sixQ[3][1]],l[sixQ[4][1]],l[sixQ[5][1]]],epsilonMax,iter=5,predicate_list=[pred[sixQ[0][1]],pred[sixQ[1][1]],pred[sixQ[2][1]],pred[sixQ[3][1]],pred[sixQ[4][1]],pred[sixQ[5][1]]])
            
            eps[0].append(opt[0])
            epNA[0].append(eqb[0])
            fps[0].append(opt[4])
            fpNA[0].append(eqb[4])
            fns[0].append(opt[2])
            fnNA[0].append(eqb[2])

            eps[1].append(opt1[0])
            epNA[1].append(eqb1[0])
            fps[1].append(opt1[4])
            fpNA[1].append(eqb1[4])
            fns[1].append(opt1[2])
            fnNA[1].append(eqb1[2])

            eps[2].append(opt2[0])
            epNA[2].append(eqb2[0])
            fps[2].append(opt2[4])
            fpNA[2].append(eqb2[4])
            fns[2].append(opt2[2])
            fnNA[2].append(eqb2[2])

            eps[3].append(opt3[0])
            epNA[3].append(eqb3[0])
            fps[3].append(opt3[4])
            fpNA[3].append(eqb3[4])
            fns[3].append(opt3[2])
            fnNA[3].append(eqb3[2])

            eps[4].append(opt4[0])
            epNA[4].append(eqb4[0])
            fps[4].append(opt4[4])
            fpNA[4].append(eqb4[4])
            fns[4].append(opt4[2])
            fnNA[4].append(eqb4[2])

            eps[5].append(opt5[0])
            epNA[5].append(eqb5[0])
            fps[5].append(opt5[4])
            fpNA[5].append(eqb5[4])
            fns[5].append(opt5[2])
            fnNA[5].append(eqb5[2])
        else:
            oneQ = random.choice(counts)
            opt,eqb,tn1,tp1 = disj([oneQ[0]],[th[oneQ[1]]],failingRate,[a_total[oneQ[1]]],[l[oneQ[1]]],epsilonMax,iter=5,predicate_list=[pred[oneQ[1]]])
            
            twoQ = counts[0:2]
            random.shuffle(twoQ)
            opt1,eqb1,tn2,tp2 = disj([twoQ[0][0],twoQ[1][0]],[th[twoQ[0][1]],th[twoQ[1][1]]],failingRate,[a_total[twoQ[0][1]],a_total[twoQ[1][1]]],[l[twoQ[0][1]],l[twoQ[1][1]]],epsilonMax,iter=5,predicate_list=[pred[twoQ[0][1]],pred[twoQ[1][1]]])
            
            threeQ = counts[0:3]
            random.shuffle(threeQ)
            opt2,eqb2,tn,tp = disj([threeQ[0][0],threeQ[1][0],threeQ[2][0]],[th[threeQ[0][1]],th[threeQ[1][1]], th[threeQ[2][1]]],failingRate,[a_total[threeQ[0][1]],a_total[threeQ[1][1]],a_total[threeQ[2][1]]],[l[threeQ[0][1]],l[threeQ[1][1]],l[threeQ[2][1]]],epsilonMax,iter=5,predicate_list=[pred[threeQ[0][1]],pred[threeQ[1][1]],pred[threeQ[2][1]]])
            
            fourQ = counts[0:4]
            random.shuffle(fourQ)
            opt3,eqb3,tn,tp = disj([fourQ[0][0],fourQ[1][0],fourQ[2][0],fourQ[3][0]],[th[fourQ[0][1]],th[fourQ[1][1]], th[fourQ[2][1]], th[fourQ[3][1]]],failingRate,[a_total[fourQ[0][1]],a_total[fourQ[1][1]],a_total[fourQ[2][1]],a_total[fourQ[3][1]]],[l[fourQ[0][1]],l[fourQ[1][1]],l[fourQ[2][1]],l[fourQ[3][1]]],epsilonMax,iter=5,predicate_list=[pred[fourQ[0][1]],pred[fourQ[1][1]],pred[fourQ[2][1]],pred[fourQ[3][1]]])
            
            fiveQ = counts[0:5]
            random.shuffle(fiveQ)
            opt4,eqb4,tn,tp = disj([fiveQ[0][0],fiveQ[1][0],fiveQ[2][0],fiveQ[3][0],fiveQ[4][0]],[th[fiveQ[0][1]],th[fiveQ[1][1]], th[fiveQ[2][1]], th[fiveQ[3][1]], th[fiveQ[4][1]]],failingRate,[a_total[fiveQ[0][1]],a_total[fiveQ[1][1]],a_total[fiveQ[2][1]],a_total[fiveQ[3][1]],a_total[fiveQ[4][1]]],[l[fiveQ[0][1]],l[fiveQ[1][1]],l[fiveQ[2][1]],l[fiveQ[3][1]],l[fiveQ[4][1]]],epsilonMax,iter=5,predicate_list=[pred[fiveQ[0][1]],pred[fiveQ[1][1]],pred[fiveQ[2][1]],pred[fiveQ[3][1]],pred[fiveQ[4][1]]])
            
            sixQ = counts
            random.shuffle(sixQ)
            opt5,eqb5,tn,tp = disj([sixQ[0][0],sixQ[1][0],sixQ[2][0],sixQ[3][0],sixQ[4][0],sixQ[5][0]],[th[sixQ[0][1]],th[sixQ[1][1]], th[sixQ[2][1]], th[sixQ[3][1]], th[sixQ[4][1]], th[sixQ[5][1]]],failingRate,[a_total[sixQ[0][1]],a_total[sixQ[1][1]],a_total[sixQ[2][1]],a_total[sixQ[3][1]],a_total[sixQ[4][1]],a_total[sixQ[5][1]]],[l[sixQ[0][1]],l[sixQ[1][1]],l[sixQ[2][1]],l[sixQ[3][1]],l[sixQ[4][1]],l[sixQ[5][1]]],epsilonMax,iter=5,predicate_list=[pred[sixQ[0][1]],pred[sixQ[1][1]],pred[sixQ[2][1]],pred[sixQ[3][1]],pred[sixQ[4][1]],pred[sixQ[5][1]]])
            
            eps[0].append(opt[0])
            epNA[0].append(eqb[0])
            fps[0].append(opt[4])
            fpNA[0].append(eqb[4])
            fns[0].append(opt[2])
            fnNA[0].append(eqb[2])

            eps[1].append(opt1[0])
            epNA[1].append(eqb1[0])
            fps[1].append(opt1[4])
            fpNA[1].append(eqb1[4])
            fns[1].append(opt1[2])
            fnNA[1].append(eqb1[2])

            eps[2].append(opt2[0])
            epNA[2].append(eqb2[0])
            fps[2].append(opt2[4])
            fpNA[2].append(eqb2[4])
            fns[2].append(opt2[2])
            fnNA[2].append(eqb2[2])

            eps[3].append(opt3[0])
            epNA[3].append(eqb3[0])
            fps[3].append(opt3[4])
            fpNA[3].append(eqb3[4])
            fns[3].append(opt3[2])
            fnNA[3].append(eqb3[2])

            eps[4].append(opt4[0])
            epNA[4].append(eqb4[0])
            fps[4].append(opt4[4])
            fpNA[4].append(eqb4[4])
            fns[4].append(opt4[2])
            fnNA[4].append(eqb4[2])

            eps[5].append(opt5[0])
            epNA[5].append(eqb5[0])
            fps[5].append(opt5[4])
            fpNA[5].append(eqb5[4])
            fns[5].append(opt5[2])
            fnNA[5].append(eqb5[2])

    fns = [mean(fnr) for fnr in fns]
    fps = [mean(fpr) for fpr in fps]
    eps = [mean(ep) for ep in eps]
    epNA= [mean(ep) for ep in epNA]
    fnNA= [mean(fnr) for fnr in fnNA]
    fpNA= [mean(fpr) for fpr in fpNA]
    return eps,fns,fps,epNA,fnNA,fpNA

