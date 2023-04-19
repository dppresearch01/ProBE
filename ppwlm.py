import numpy as np
import math
from scipy.stats import entropy,norm
from decimal import Decimal, Context
from scipy import integrate
import pickle
import os
import query
import datetime
import time
import random
from collections import OrderedDict, Counter
import pandas as pd
from statistics import mean,stdev,mode,median
import mysql.connector
from queries.queries_sales import sales_volume,average_transaction_value,distinct_customers,transaction_per_category,avg_total_per_category,avg_itemprice_per_category,category_count,average_item_number,sales_volume_women
from queries.queries_sales import sales_volume_marmara
from itertools import combinations
from queries.taxi_queries import query_taxi_count_location,query_taxi_count, query_congestion_amount, query_taxi_fareamount, query_taxi_tip_amount,query_taxi_tolls_amount,query_taxi_total_amount,query_taxi_count_payment_type,query_taxi_count_flag
from entropy import computeEntropy,computeMinEntropy
import matplotlib.patches as mpatches
from utils import intersection_test,union_test,classify_mix,get_denominator,get_numerator,getBi,new_classify_and,new_classify_or

def noise_down(lap_noise, eps_old, eps_new):
    #assert eps_new > eps_old

    pdf = [eps_old / eps_new * np.exp((eps_old - eps_new) * abs(lap_noise)),
           (eps_new - eps_old) / (2.0 * eps_new),
           (eps_old + eps_new) / (2.0 * eps_new) * (1.0 - np.exp((eps_old - eps_new) * abs(lap_noise)))]

    p = np.random.random_sample()
    if p <= pdf[0]:
        z = lap_noise

    elif p <= pdf[0] + pdf[1]:
        z = np.log(p) / (eps_old + eps_new)

    elif p <= pdf[0] + pdf[1] + pdf[2]:
        z = np.log(p * (np.exp(abs(lap_noise) * (eps_old - eps_new)) - 1.0) + 1.0) / (eps_old - eps_new)

    else:
        z = abs(lap_noise) - np.log(1.0 - p) / (eps_new + eps_old)

    return z


def distribute(counts_total,t_total,failingRate,a_total,l_sensitivity_total,epsilonMax,predicate_list=[]):
    eps,fns,fps,pred_eps,entropies = [],[],[],[],[]
    startingEp = 0.01
    m=4
    for i in range(100):
        ep_total = 0
        q1 = counts_total[0]
        q2 = counts_total[1]
        q3 = counts_total[2]
        q_i = [q1,q2,q3]
        idx = [q[1] for q in q_i]
        counts = [counts_total[j][0] for j in idx]
        ts = [t_total[j] for j in idx]
        aes = [a_total[j] for j in idx]
        ls = [l_sensitivity_total[j] for j in idx]
        preds = [predicate_list[j] for j in idx]
        #Q1 U Q2 
        fnr12,fpr12,ep12,pred_ep12,pred_ep_dict12,pred_a12,pred_b12,bi_total = ppwlmPROBE(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],[startingEp]*2, epsilonMax, m,0,aes=aes,og_idx=[0,1],predicate_list = preds[0:2],combine=True,alt=True)
        ep_total+= ep12

        #Q1 U Q3 
        fnr13,fpr13,ep13,pred_ep13,pred_ep_dict13,pred_a13,pred_b13,bi_total = ppwlmPROBE([counts[0],counts[2]],[ts[0],ts[2]],failingRate,[aes[0],aes[2]],[ls[0],ls[2]],[startingEp]*2,epsilonMax,m,0,aes=aes,og_idx=[0,2],predicate_list =[preds[0],preds[2]],combine=True,alt=True)
        ep_total+= ep13
        ep_list = dict(Counter(pred_ep_dict12)+Counter(pred_ep_dict13))

        #classify
        preds_after_shift = intersection_test([pred_a12,pred_a13])
        preds_before_shift = intersection_test([pred_b12,pred_b13])
        tp,tn,fp,fn = classify_mix(counts,ts,preds_after_shift,0,preds_before_shift,distrib=True)
        positives= tp+fn
        negatives= tn+fp
        fps.append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
        fns.append(len(fn)/len(positives) if len(positives)> 0 else 0)
        eps.append(ep_total)
        entropies.append(computeMinEntropy(list(ep_list.values()))[0])
    return mean(eps),mean(fns),mean(fps), mean(entropies)


def progCombine(counts_total,t_total,failingRate,a_total,l_sensitivity_total,epsilonMax,predicate_list=[],vary=False):
    eps,fns,fps,ep_preds,entropies= [[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)]
    startingEp = 0.1
    m=4
    if vary == True:
        for i in range(100):
            #for n=3 we do N U
            ep_total = 0
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

            # conj first q1 N q2
            fnr12,fpr12,ep12,pred_ep12,pred_ep_dict12,pred_a12,pred_b12,bi_total = ppwlmPROBE(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],[startingEp]*2, epsilonMax, m,1,aes=aes,og_idx=[0,1],predicate_list = preds[0:2],combine=True)
            ep_total+= ep12
            #single Q3
            bi =failingRate-bi_total
            ep3,ep_preds3,ep_list_dict3,pred3 = ppwlm(counts[2],ts[2],bi,aes[2],2,aes,ls[2],startingEp,epsilonMax,m,predicate_list=preds[2])
            ep_total+=ep3
            
            preds_after_shift = union_test([pred_a12,pred3])
            preds_before_shift= union_test([pred_b12,preds[2]])
            ep_list = dict(Counter(pred_ep_dict12)+Counter(ep_list_dict3))

            tp,tn,fp,fn = classify_mix(counts,ts,preds_after_shift,0,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[2].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[2].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[2].append(ep_total)
            ep_preds[2] = list(ep_list.values())
            entropies[2].append(computeMinEntropy(list(ep_list.values()))[0])
        return mean(eps[2]),mean(fns[2]),mean(fps[2]), mean(entropies[2])

    ##FOR PPWLM 1 IS CONJ AND 0 IS DISJ
    for i in range(150):

        #first for n-1 we do one_q
        q_i = random.choice(counts_total)      
        idx = q_i[1]
        counts = [q_i[0]]
        
        ts = [t_total[idx]]
        aes = [a_total[idx]]
        ls = [l_sensitivity_total[idx]]
        preds = [predicate_list[idx]]
        ep,ep_list_preds,ep_list_dict,preds_selected = ppwlm(counts[0],ts[0],failingRate,aes[0],0,aes,ls[0],startingEp,epsilonMax,m,predicate_list=preds[0])
        tp,tn,fp,fn = new_classify_or(counts,[[ts[k]]*len(counts[k]) for k in range(len(ts))],[],preds_selected,preds[0],preds)
        positives = tp + fn
        negatives = tn + fp
        fps[0].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
        fns[0].append(len(fn)/len(positives) if len(positives)> 0 else 0)
        eps[0].append(ep)
        entropies[0].append(computeMinEntropy(ep_list_preds)[0])
        ep_preds[0] = ep_list_preds

        
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
        funcs = [0,1]
        if i < 75:
            fnr,fpr,ep,pred_ep,pred_ep_dict,pred_a,pred_b,bi_total= ppwlmPROBE(counts,ts,failingRate,aes,ls,[startingEp]*2,epsilonMax,m,0, predicate_list=preds)
            eps[1].append(ep)
            fps[1].append(fpr)
            fns[1].append(fnr)
            entropies[1].append(computeMinEntropy(pred_ep)[0])
            ep_preds[1] = pred_ep
        else:
            fnr,fpr,ep,pred_ep,pred_ep_dict,pred_a,pred_b,bi_total= ppwlmPROBE(counts,ts,failingRate,aes,ls,[startingEp]*2,epsilonMax,m,1, predicate_list=preds)
            eps[1].append(ep)
            fps[1].append(fpr)
            fns[1].append(fnr)
            entropies[1].append(computeMinEntropy(pred_ep)[0])
            ep_preds[1] = pred_ep
        #for n=3 we do N U
        ep_total = 0
        combos = [list(c) for c in combinations(counts_total,3)]
        q_i = random.choice(combos)
        idx = [q[1] for q in q_i]
        counts = [counts_total[j][0] for j in idx]
        ts = [t_total[j] for j in idx]
        aes = [a_total[j] for j in idx]
        ls = [l_sensitivity_total[j] for j in idx]
        preds = [predicate_list[j] for j in idx]

        # conj first q1 N q2
        fnr12,fpr12,ep12,pred_ep12,pred_ep_dict12,pred_a12,pred_b12,bi_total = ppwlmPROBE(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],[startingEp]*2, epsilonMax, m,1,aes=aes,og_idx=[0,1],predicate_list = preds[0:2],combine=True)
       
        ep_total+= ep12
        #single Q3
        ep3,ep_preds3,ep_list_dict3,pred3 = ppwlm(counts[2],ts[2],failingRate,aes[2],2,aes,ls[2],startingEp,epsilonMax,m,predicate_list=preds[2], one=True)
        ep_total+=ep3
        preds_after_shift = union_test([pred_a12,pred3])
        preds_before_shift= union_test([pred_b12,preds[2]])
        ep_list = dict(Counter(pred_ep_dict12)+Counter(ep_list_dict3))

        tp,tn,fp,fn = classify_mix(counts,ts,preds_after_shift,0,preds_before_shift)
        positives= tp+fn
        negatives= tn+fp
        fps[2].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
        fns[2].append(len(fn)/len(positives) if len(positives)> 0 else 0)
        eps[2].append(ep_total)
        ep_preds[2] = list(ep_list.values())
        entropies[2].append(computeMinEntropy(list(ep_list.values()))[0])
 
        # for n=4 we do N N U or N U U
        ep_total = 0
        combos = [list(c) for c in combinations(counts_total,4)]
        q_i = random.choice(combos)
        idx = [q[1] for q in q_i]
        counts = [counts_total[j][0] for j in idx]
        ts = [t_total[j] for j in idx]
        aes = [a_total[j] for j in idx]
        ls = [l_sensitivity_total[j] for j in idx]
        preds = [predicate_list[j] for j in idx]
        #randomly choose N N U or N U U
        flag = random.choice([0,1])
        flag = 0 if i < 75 else 1
        if flag == 1: # N N U          
            fnr123,fpr123,ep123,pred_ep123,pred_ep_dict123,pred_a123,pred_b123,bi_total = ppwlmPROBE(counts[0:3],ts[0:3],failingRate,aes[0:3],ls[0:3],[startingEp]*3, epsilonMax, m,1,aes=aes,og_idx=[0,1,2],predicate_list = preds[0:3],combine=True)
            ep_total+= ep123
            #single Q3
            bi= getBi(aes,aes[3],3,failingRate)
            ep4,ep_preds4,ep_list_dict4,pred4 = ppwlm(counts[3],ts[3],failingRate,aes[3],3,aes,ls[3],startingEp,epsilonMax,m,predicate_list=preds[3], one=True)
            ep_total+=ep4
            preds_after_shift = union_test([pred_a12,pred3])
            preds_before_shift= union_test([pred_b12,preds[2]])
            ep_list = dict(Counter(pred_ep_dict123)+Counter(ep_list_dict4))
            tp,tn,fp,fn = classify_mix(counts,ts,preds_after_shift,0,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[3].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[3].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[3].append(ep_total)
            ep_preds[3] = list(ep_list.values())
            entropies[3].append(computeMinEntropy(list(ep_list.values()))[0])
        else: # N U U         
            #FIRST CONJ
            fnr12,fpr12,ep12,pred_ep12,pred_ep_dict12,pred_a12,pred_b12,bi_total = ppwlmPROBE(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],[startingEp]*2, epsilonMax, m,1,aes=aes,og_idx=[0,1],predicate_list = preds[0:2],combine=True)
            ep_total+= ep12
            # SECOND DISJ
            fnr34,fpr34,ep34,pred_ep34,pred_ep_dict34,pred_a34,pred_b34,bi_total = ppwlmPROBE(counts[2:4],ts[2:4],failingRate,aes[2:4],ls[2:4],[startingEp]*2, epsilonMax, m,0,aes=aes,og_idx=[2,3],predicate_list = preds[2:4],combine=True)
            ep_total+=ep34
            preds_after_shift = union_test([pred_a12,pred_a34])
            preds_before_shift = union_test([pred_b12,pred_b34])
            ep_list = dict(Counter(pred_ep_dict12)+Counter(pred_ep_dict34))
            tp,tn,fp,fn = classify_mix(counts,ts,preds_after_shift,1,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[3].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[3].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[3].append(ep_total)
            ep_preds[3] = list(ep_list.values())
            entropies[3].append(computeMinEntropy(list(ep_list.values()))[0])
        # for n=5 we do N N U U / N N N U / N U U U
        ep_total = 0
        combos = [list(c) for c in combinations(counts_total,5)]
        q_i = random.choice(combos)
        idx = [q[1] for q in q_i]
        counts = [counts_total[j][0] for j in idx]
        ts = [t_total[j] for j in idx]
        aes = [a_total[j] for j in idx]
        ls = [l_sensitivity_total[j] for j in idx]
        preds = [predicate_list[j] for j in idx]
        #randomly choose N N U or N U U
        flag = random.choice([0,1,2])
        if i < 50:
            flag = 0
        elif i < 100:
            flag = 1
        else:
            flag =2
        if flag == 0: # N N N U
            fnr123,fpr123,ep123,pred_ep123,pred_ep_dict123,pred_a123,pred_b123,bi_total = ppwlmPROBE(counts[0:4],ts[0:4],failingRate,aes[0:4],ls[0:4],[startingEp]*4, epsilonMax, m,1,aes=aes,og_idx=[0,1,2,3],predicate_list = preds[0:4],combine=True)
            ep_total+= ep123
            #single Q4         
            ep3,ep_preds3,ep_list_dict3,pred3 = ppwlm(counts[4],ts[4],failingRate,aes[4],4,aes,ls[4],startingEp,epsilonMax,m,predicate_list=preds[4], one=True)
            ep_total+=ep3
            preds_after_shift = union_test([pred_a12,pred3])
            preds_before_shift= union_test([pred_b12,preds[4]])
            ep_list = dict(Counter(pred_ep_dict123)+Counter(ep_list_dict3))
            
            tp,tn,fp,fn = classify_mix(counts,ts,preds_after_shift,0,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[4].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[4].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[4].append(ep_total)
            ep_preds[4] = list(ep_list.values())
            entropies[4].append(computeMinEntropy(list(ep_list.values()))[0])
        elif flag == 1: # N N U U 
            #FIRST CONJ
            fnr12,fpr12,ep12,pred_ep12,pred_ep_dict12,pred_a12,pred_b12,bi_total = ppwlmPROBE(counts[0:3],ts[0:3],failingRate,aes[0:3],ls[0:3],[startingEp]*3, epsilonMax, m,1,aes=aes,og_idx=[0,1,2],predicate_list = preds[0:3],combine=True)
            
            ep_total+= ep12
            # SECOND DISJ
            fnr34,fpr34,ep34,pred_ep34,pred_ep_dict34,pred_a34,pred_b34,bi_total = ppwlmPROBE(counts[3:5],ts[3:5],failingRate,aes[3:5],ls[3:5],[startingEp]*2, epsilonMax, m,0,aes=aes,og_idx=[3,4],predicate_list = preds[3:5],combine=True)
            ep_total+=ep34
            preds_after_shift = union_test([pred_a12,pred_a34])
            preds_before_shift = union_test([pred_b12,pred_b34])
            ep_list = dict(Counter(pred_ep_dict12)+Counter(pred_ep_dict34))

            tp,tn,fp,fn= classify_mix(counts,ts,preds_after_shift,1,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[4].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[4].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[4].append(ep_total)
            ep_preds[4] = list(ep_list.values())
            entropies[4].append(computeMinEntropy(list(ep_list.values()))[0])
        else: # N U U U
            #FIRST CONJ
            fnr12,fpr12,ep12,pred_ep12,pred_ep_dict12,pred_a12,pred_b12,bi_total = ppwlmPROBE(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],[startingEp]*2, epsilonMax, m,1,aes=aes,og_idx=[0,1],predicate_list = preds[0:2],combine=True)
            ep_total+= ep12
            # SECOND DISJ
            fnr34,fpr34,ep34,pred_ep34,pred_ep_dict34,pred_a34,pred_b34,bi_total = ppwlmPROBE(counts[2:5],ts[2:5],failingRate,aes[2:5],ls[2:5],[startingEp]*3, epsilonMax, m,0,aes=aes,og_idx=[2,3,4],predicate_list = preds[2:5],combine=True)
            ep_total+=ep34
            preds_after_shift = union_test([pred_a12,pred_a34])
            preds_before_shift = union_test([pred_b12,pred_b34])
            ep_list = dict(Counter(pred_ep_dict12)+Counter(pred_ep_dict34))
            
            tp,tn,fp,fn= classify_mix(counts,ts,preds_after_shift,2,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[4].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[4].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[4].append(ep_total)
            ep_preds[4] = list(ep_list.values())
            entropies[4].append(computeMinEntropy(list(ep_list.values()))[0])
        # for n=6 we do N N N N U / N N N U U / N N U U U / N U U U U
        ep_total = 0
        combos = [list(c) for c in combinations(counts_total,6)]
        q_i = random.choice(combos)
        idx = [q[1] for q in q_i]
        counts = [counts_total[j][0] for j in idx]
        ts = [t_total[j] for j in idx]
        aes = [a_total[j] for j in idx]
        ls = [l_sensitivity_total[j] for j in idx]
        preds = [predicate_list[j] for j in idx]
        #randomly choose N N U or N U U
        flag = random.choice([0,1,2,3])
        if i < 37:
            flag = 0
        elif i < 74:
            flag = 1
        elif i <111:
            flag =2
        else:
            flag=3
        if flag == 0:  # N N N N U
            fnr123,fpr123,ep123,pred_ep123,pred_ep_dict123,pred_a123,pred_b123,bi_total = ppwlmPROBE(counts[0:5],ts[0:5],failingRate,aes[0:5],ls[0:5],[startingEp]*5, epsilonMax, m,1,aes=aes,og_idx=[0,1,2,3,4],predicate_list = preds[0:5],combine=True)
            ep_total+= ep123
            #single Q4
            ep3,ep_preds3,ep_list_dict3,pred3 = ppwlm(counts[5],ts[5],failingRate,aes[5],5,aes,ls[5],startingEp,epsilonMax,m,predicate_list=preds[5], one=True)
            ep_total+=ep3
            preds_after_shift = union_test([pred_a12,pred3])
            preds_before_shift= union_test([pred_b12,preds[5]])
            tp,tn,fp,fn= classify_mix(counts,ts,preds_after_shift,0,preds_before_shift)
            ep_list = dict(Counter(pred_ep_dict123)+Counter(ep_list_dict3))
            
            positives= tp+fn
            negatives= tn+fp
            fps[5].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[5].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[5].append(ep_total)
            ep_preds[5] = list(ep_list.values())
            entropies[5].append(computeMinEntropy(list(ep_list.values()))[0])

        elif flag == 1:  # N N N U U
            #FIRST CONJ
            fnr12,fpr12,ep12,pred_ep12,pred_ep_dict12,pred_a12,pred_b12,bi_total = ppwlmPROBE(counts[0:4],ts[0:4],failingRate,aes[0:4],ls[0:4],[startingEp]*4, epsilonMax, m,1,aes=aes,og_idx=[0,1,2,3],predicate_list = preds[0:4],combine=True)
            ep_total+= ep12
            # SECOND DISJ
            fnr34,fpr34,ep34,pred_ep34,pred_ep_dict34,pred_a34,pred_b34,bi_total = ppwlmPROBE(counts[4:6],ts[4:6],failingRate,aes[4:6],ls[4:6],[startingEp]*2, epsilonMax, m,0,aes=aes,og_idx=[4,5],predicate_list = preds[4:6],combine=True)
            ep_total+=ep34
            preds_after_shift = union_test([pred_a12,pred_a34])
            preds_before_shift = union_test([pred_b12,pred_b34])
            ep_list = dict(Counter(pred_ep_dict12)+Counter(pred_ep_dict34))
            
            tp,tn,fp,fn= classify_mix(counts,ts,preds_after_shift,1,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[5].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[5].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[5].append(ep_total)
            ep_preds[5] = list(ep_list.values())
            entropies[5].append(computeMinEntropy(list(ep_list.values()))[0])
        elif flag == 2:  # N N U U U
            #FIRST CONJ
            fnr12,fpr12,ep12,pred_ep12,pred_ep_dict12,pred_a12,pred_b12,bi_total = ppwlmPROBE(counts[0:3],ts[0:3],failingRate,aes[0:3],ls[0:3],[startingEp]*3, epsilonMax, m,1,aes=aes,og_idx=[0,1,2],predicate_list = preds[0:3],combine=True)
            ep_total+= ep12
            # SECOND DISJ
            fnr34,fpr34,ep34,pred_ep34,pred_ep_dict34,pred_a34,pred_b34,bi_total = ppwlmPROBE(counts[3:6],ts[3:6],failingRate,aes[3:6],ls[3:6],[startingEp]*3, epsilonMax, m,0,aes=aes,og_idx=[3,4,5],predicate_list = preds[3:6],combine=True)
            ep_total+=ep34
            preds_after_shift = union_test([pred_a12,pred_a34])
            preds_before_shift = union_test([pred_b12,pred_b34])
            ep_list = dict(Counter(pred_ep_dict12)+Counter(pred_ep_dict34))
            
            tp,tn,fp,fn= classify_mix(counts,ts,preds_after_shift,2,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[5].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[5].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[5].append(ep_total)
            ep_preds[5] = list(ep_list.values())
            entropies[5].append(computeMinEntropy(list(ep_list.values()))[0])
        else:  # N U U U U
            #FIRST CONJ
            fnr12,fpr12,ep12,pred_ep12,pred_ep_dict12,pred_a12,pred_b12,bi_total = ppwlmPROBE(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],[startingEp]*2, epsilonMax, m,1,aes=aes,og_idx=[0,1],predicate_list = preds[0:2],combine=True)
            ep_total+= ep12
            # SECOND DISJ
            fnr34,fpr34,ep34,pred_ep34,pred_ep_dict34,pred_a34,pred_b34,bi_total = ppwlmPROBE(counts[2:6],ts[2:6],failingRate,aes[2:6],ls[2:6],[startingEp]*4, epsilonMax, m,0,aes=aes,og_idx=[2,3,4,5],predicate_list = preds[2:6],combine=True)
            ep_total+=ep34
            preds_after_shift = union_test([pred_a12,pred_a34])
            preds_before_shift = union_test([pred_b12,pred_b34])
            ep_list = dict(Counter(pred_ep_dict12)+Counter(pred_ep_dict34))

            tp,tn,fp,fn= classify_mix(counts,ts,preds_after_shift,3,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[5].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[5].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[5].append(ep_total)
            ep_preds[5] = list(ep_list.values())
            entropies[5].append(computeMinEntropy(list(ep_list.values()))[0])
    fns = [mean(fnr) for fnr in fns]
    fps = [mean(fpr) for fpr in fps]
    eps = [mean(ep) for ep in eps]
    entropies = [mean(ent) for ent in entropies]
    return eps,fns,fps, entropies

def ppwlm(count,thr,failingRate,a,idx,aes,l,ep_start,epsilonMax,m,predicate_list=[],one=False):
    if one == True:
        prod = get_numerator(a,idx,aes)*l
        bi = (prod/get_denominator(aes,False,idx,s=[l]))*failingRate
        failingRate = bi
    preds_after_shift=[]
    preds_before_shift=[]
    predicate_total=[]
    predicate_eps = []
    early_stop = False
    ep_total = 0
    ep_m = l*np.log(0.5*m/failingRate)/a*1.0
    if ep_m < epsilonMax:
        ep = ep_start
        base=pow(1.0*ep_m/ep,(1.0/(m-1)))
        lap_noises = np.random.laplace(0,1.0*l/ep,len(count))
        a_start = np.log(0.5*m/failingRate)/ep*1.0
        eliminated_idx = [k for k in range(len(count)) if (count[k] +lap_noises[k] < thr-a_start)]
        decided_idx = [k for k in range(len(count)) if (count[k] +lap_noises[k] > thr+a_start)]
        decided_preds = [predicate_list[k] for k in decided_idx]
        undecided_idx = [k for k in range(len(count)) if ((count[k] +lap_noises[k] < thr+a_start) and (count[k] +lap_noises[k] > thr-a_start))]
        undecided_preds = [predicate_list[k] for k in undecided_idx]
        if(type==1 and len(decided_idx) == 0 and len([k for k in range(len(count)) if (count[k]+lap_noises[k] > thr)]) == 0):
            predicate_total=decided_preds
            early_stop = True
            stop_j = i
            ep_total += ep
            predicate_eps += list(zip(iter(decided_idx),[ep]*len(count)))
            predicate_eps += list(zip(iter(eliminated_idx),[ep]*len(count)))
        else:
            for j in range(1,m):
                if len(undecided_idx) == 0:
                    predicate_eps += list(zip(iter(decided_idx),[ep]*len(count)))
                    predicate_eps += list(zip( iter(eliminated_idx),[ep]*len(count)))
                    selected = decided_idx
                    predicate_total= decided_preds
                    ep_total +=  ep
                    break
                prev_ep = ep
                ep = ep_start*pow(base,j-1)
                
                lap_noises=[noise_down(lap_noise, prev_ep, ep) for lap_noise in lap_noises]
                shifted_a = np.log(0.5*m/failingRate)/ep*1.0
                eliminated_idx = [k for k in range(len(count)) if (count[k] +lap_noises[k] < thr-shifted_a)]
                
                predicate_eps += list(zip(iter(eliminated_idx),[ep]*len(count)))
                decided_idx=union_test([decided_idx,[k for k in undecided_idx if count[k]+lap_noises[k]>thr+shifted_a]])
                decided_preds = [predicate_list[k] for k in decided_idx]
                undecided_idx=[ k for k in undecided_idx if ((count[k]+lap_noises[k] < thr+shifted_a) and (count[k] +lap_noises[k] > thr-shifted_a)) ]
                undecided_preds = [predicate_list[k] for k in undecided_idx]
                if j == m-1:
                    predicate_total=union_test([undecided_preds,decided_preds])
                    predicate_eps += list(zip(iter(union_test([undecided_idx,decided_idx])),[ep_m]*len(count)))
                    selected=union_test([undecided_idx,decided_idx])
                    ep_total += ep_m 
    else:
        print('QUERY DENIED')
        return -1,-1,-1

    pred_eps_unique = {}
    for tup in predicate_eps:
        if tup[0] in pred_eps_unique:
            pred_eps_unique[tup[0]] += tup[1]
        else:
            pred_eps_unique[tup[0]] = tup[1]
    ep_list_preds = list(pred_eps_unique.values()) 
    ep_list_preds = [ep for ep in ep_list_preds if ep != 0]
    
    return ep_total, ep_list_preds,pred_eps_unique, predicate_total
def ppwlmPROBE(counts,thresholds,failingRate,a_s,l_sens,eps_start,epsilonMax,m,type,alt=False,aes=[],og_idx=[],predicate_list=[],combine=False):
   ## type = 0 if disj 1 if conj
    epsilon_f = 0
    ep_total = 0
    ep_list = []
    predicate_total=[]
    predicate_eps = []
    eps_max = []
    stop_j = -1
    leftover_idx = -1
    new_as=[]
    preds_after_shift=[]
    preds_before_shift=[]
    bi_total =0
    for i,count in enumerate(counts):
        early_stop = False 
        selected = []
        decided, undecided, eliminated = [],[],[]
        thr = thresholds[i]
        a_start = a_s[i]
        prod = 0
        bi = 0
        ep_m = 0

        if combine==True:
            prod =  get_numerator(a_s[i],og_idx[i],aes)*l_sens[i]
            bi = (prod/get_denominator(aes,alt,og_idx[i],s=l_sens))*failingRate
            ep_m = l_sens[i]*np.log(0.5*m/bi)/a_s[i]*1.0
            bi_total+=bi
        else:
            prod = get_numerator(a_s[i],i,a_s)*l_sens[i]
            bi = (prod/get_denominator(a_s,alt,i,s=l_sens))*failingRate
            ep_m = l_sens[i]*np.log(0.5*m/bi)/a_s[i]*1.0
        # else:
        if (leftover_idx != -1 ):
            new_idx = i-leftover_idx
            prod = get_numerator(a_s[i],new_idx,new_as)*l_sens[i]
            bi = (prod/get_denominator(new_as,alt,new_idx,s=l_sens))*failingRate
            ep_m = l_sens[i]*np.log(0.5*m/bi)/new_as[new_idx]*1.0
        if ep_m < epsilonMax:
            ep=eps_start[i]
            base=pow(1.0*ep_m/ep,(1.0/(m-1)))
            lap_noises = np.random.laplace(0,1.0*l_sens[i]/ep,len(count))
            a_start = np.log(0.5*m/bi)/ep*1.0
            eliminated_idx = [k for k in range(len(count)) if (count[k] +lap_noises[k] < thr-a_start)]
            decided_idx = [k for k in range(len(count)) if (count[k] +lap_noises[k] > thr+a_start)]
            decided_preds = [predicate_list[i][k] for k in decided_idx]
            undecided_idx = [k for k in range(len(count)) if ((count[k] +lap_noises[k] < thr+a_start) and (count[k] +lap_noises[k] > thr-a_start))]
            undecided_preds = [predicate_list[i][k] for k in undecided_idx]
            if(type==1 and len(decided_idx) == 0 and len([k for k in range(len(count)) if (count[k]+lap_noises[k] > thr)]) == 0):
                predicate_total.append(decided_preds)
                early_stop = True
                stop_j = i
                ep_total += ep
                predicate_eps += list(zip(iter(decided_idx),[ep]*len(count)))
                predicate_eps += list(zip(iter(eliminated_idx),[ep]*len(count)))
                predicate_eps += list(zip(iter(undecided_idx),[ep]*len(count)))
                break
            for j in range(1,m):
                if len(undecided_idx) == 0:
                    predicate_eps += list(zip(iter(decided_idx),[ep]*len(count)))
                    predicate_eps += list(zip( iter(eliminated_idx),[ep]*len(count)))
                    selected = decided_idx
                    predicate_total.append(decided_preds)
                    ep_total +=  ep
                    
                    failingRate = failingRate- bi* (j/m)
                    leftover_idx = i + 1
                    new_as = a_s[i+1:]
                    break
                prev_ep = ep
                ep = eps_start[i]*pow(base,j-1)
                lap_noises=[noise_down(lap_noise, prev_ep, ep) for lap_noise in lap_noises]
                shifted_a = np.log(0.5*m/bi)/ep*1.0
                
                eliminated_idx = [k for k in range(len(count)) if (count[k] +lap_noises[k] < thr-shifted_a)]
                
                predicate_eps += list(zip(iter(eliminated_idx),[ep]*len(count)))
                decided_idx=union_test([decided_idx,[k for k in undecided_idx if count[k]+lap_noises[k]>thr+shifted_a]])
                decided_preds = [predicate_list[i][k] for k in decided_idx]
                undecided_idx=[ k for k in undecided_idx if ((count[k]+lap_noises[k] < thr+shifted_a) and (count[k] +lap_noises[k] > thr-shifted_a)) ]
                undecided_preds = [predicate_list[i][k] for k in undecided_idx]
                if j == m-1:
                    predicate_total.append(union_test([undecided_preds,decided_preds]))
                    predicate_eps += list(zip(iter(union_test([undecided_idx,decided_idx])),[ep_m]*len(count)))
                    selected.append(union_test([undecided_idx,decided_idx]))
                    ep_total += ep_m 
        else:
            print('QUERY DENIED')
            return -1,-1,-1,-1,-1,-1
    
    pred_eps_unique = {}
    for tup in predicate_eps:
        if tup[0] in pred_eps_unique:
            pred_eps_unique[tup[0]] += tup[1]
        else:
            pred_eps_unique[tup[0]] = tup[1]
    ep_list_preds = list(pred_eps_unique.values())
    
    ep_list_preds = [ep for ep in ep_list_preds if ep != 0]
    if(early_stop):
        preds_after_shift = union_test(predicate_total) if type == 0 else intersection_test(predicate_total)
        preds_before_shift= union_test(predicate_list) if type == 0 else intersection_test(predicate_list)
        tp,tn,fp,fn = [],[],[],[]
        if type == 1:
            tp,tn,fp,fn= new_classify_and(counts,[[thresholds[i]]*len(counts[i]) for i in range(len(thresholds))],selected,preds_after_shift,preds_before_shift,predicate_list)
        else:
            tp,tn,fp,fn= new_classify_or(counts,[[thresholds[i]]*len(counts[i]) for i in range(len(thresholds))],selected,preds_after_shift,preds_before_shift,predicate_list)
        positives = tp + fn
        negatives = tn + fp
        fnr = len(fn)/len(positives) if len(positives) > 0 else 0
        fpr = len(fp)/len(negatives) if len(negatives) > 0 else 0
        return fnr,fpr,ep_total,ep_list_preds,pred_eps_unique, preds_after_shift,preds_before_shift,bi_total
    preds_after_shift = union_test(predicate_total) if type == 0 else intersection_test(predicate_total)
    preds_before_shift= union_test(predicate_list) if type == 0 else intersection_test(predicate_list)
    
    if type == 0:
        tp,tn,fp,fn= new_classify_or(counts, [[thresholds[i]]*len(counts[i]) for i in range(len(thresholds))],selected,preds_after_shift,preds_before_shift,predicate_list)
    else:
        tp,tn,fp,fn= new_classify_and(counts, [[thresholds[i]]*len(counts[i]) for i in range(len(thresholds))],selected,preds_after_shift,preds_before_shift,predicate_list)
  
    positives = fn + tp
    negatives = fp + tn

    fnr = len(fn)/len(positives) if len(positives) > 0 else 0
    fpr = len(fp)/len(negatives) if len(negatives) > 0 else 0

    return fnr,fpr, ep_total,ep_list_preds,pred_eps_unique, preds_after_shift,preds_before_shift,bi_total

def sales(start_time,end_time,beta,u,m_i,e,type):
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
    failingRate = 0.005 if beta==-1 else beta
    startingEp=0.01 if e==-1 else e
    m=4 if m_i ==-1 else m_i

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
        eps,fns,fps,entropies = progCombine(counts,th,failingRate,a_total,l,epsilonMax,predicate_list=pred,vary=True) 
        return eps,fns,fps,entropies
    num_tests = 6    
    fnrs,fprs,eps,pred_eps,entropies= [[] for i in range(num_tests)],[[] for i in range(num_tests)],[[] for i in range(num_tests)],[[] for i in range(num_tests)],[[] for i in range(num_tests)]
    for i in range(iter):
        oneQ = random.choice(counts)
        fnr,fpr,ep,pred_ep , pred_ep_dict,preds_a,preds_b , bi_total= ppwlmPROBE([oneQ[0]],[th[oneQ[1]]],failingRate,[a_total[oneQ[1]]],[l[oneQ[1]]],[startingEp],epsilonMax,m,type,predicate_list=[pred[oneQ[1]]])
        fnrs[0].append(fnr)
        fprs[0].append(fpr)
        eps[0].append(ep)
        entropies[0].append(computeMinEntropy(pred_ep)[0])
        
        twoQ = counts[0:2]

        random.shuffle(twoQ)
        fnr,fpr,ep,pred_ep , pred_ep_dict,preds_a,preds_b , bi_total= ppwlmPROBE([twoQ[0][0],twoQ[1][0]],[th[twoQ[0][1]],th[twoQ[1][1]]],failingRate,[a_total[twoQ[0][1]],a_total[twoQ[1][1]]],[l[twoQ[0][1]],l[twoQ[1][1]]],[startingEp,startingEp],epsilonMax,m,type,predicate_list=[pred[twoQ[0][1]],pred[twoQ[1][1]]])
        fnrs[1].append(fnr)
        fprs[1].append(fpr)
        eps[1].append(ep)
        entropies[1].append(computeMinEntropy(pred_ep)[0])

        threeQ = counts[0:3]
        random.shuffle(threeQ)
        fnr,fpr,ep,pred_ep , pred_ep_dict,preds_a,preds_b , bi_total= ppwlmPROBE([threeQ[0][0],threeQ[1][0],threeQ[2][0]],[th[threeQ[0][1]],th[threeQ[1][1]], th[threeQ[2][1]]],failingRate,[a_total[threeQ[0][1]],a_total[threeQ[1][1]],a_total[threeQ[2][1]]],[l[threeQ[0][1]],l[threeQ[1][1]],l[threeQ[2][1]]],[startingEp,startingEp,startingEp],epsilonMax,m,type,predicate_list=[pred[threeQ[0][1]],pred[threeQ[1][1]],pred[threeQ[2][1]]])
        fnrs[2].append(fnr)
        fprs[2].append(fpr)   
        eps[2].append(ep)
        entropies[2].append(computeMinEntropy(pred_ep)[0])

        fourQ = counts[0:4]
        random.shuffle(fourQ)
        fnr,fpr,ep,pred_ep , pred_ep_dict,preds_a,preds_b , bi_total= ppwlmPROBE([fourQ[0][0],fourQ[1][0],fourQ[2][0],fourQ[3][0]],[th[fourQ[0][1]],th[fourQ[1][1]], th[fourQ[2][1]], th[fourQ[3][1]]],failingRate,[a_total[fourQ[0][1]],a_total[fourQ[1][1]],a_total[fourQ[2][1]],a_total[fourQ[3][1]]],[l[fourQ[0][1]],l[fourQ[1][1]],l[fourQ[2][1]],l[fourQ[3][1]]],[startingEp,startingEp,startingEp,startingEp],epsilonMax,m,type,predicate_list=[pred[fourQ[0][1]],pred[fourQ[1][1]],pred[fourQ[2][1]],pred[fourQ[3][1]]])
        fnrs[3].append(fnr)
        fprs[3].append(fpr)
        eps[3].append(ep)
        entropies[3].append(computeMinEntropy(pred_ep)[0])

        fiveQ = counts[0:5]
        random.shuffle(fiveQ)
        fnr,fpr,ep,pred_ep , pred_ep_dict,preds_a,preds_b , bi_total= ppwlmPROBE([fiveQ[0][0],fiveQ[1][0],fiveQ[2][0],fiveQ[3][0],fiveQ[4][0]],[th[fiveQ[0][1]],th[fiveQ[1][1]], th[fiveQ[2][1]], th[fiveQ[3][1]], th[fiveQ[4][1]]],failingRate,[a_total[fiveQ[0][1]],a_total[fiveQ[1][1]],a_total[fiveQ[2][1]],a_total[fiveQ[3][1]],a_total[fiveQ[4][1]]],[l[fiveQ[0][1]],l[fiveQ[1][1]],l[fiveQ[2][1]],l[fiveQ[3][1]],l[fiveQ[4][1]]],[startingEp,startingEp,startingEp,startingEp,startingEp],epsilonMax,m,type,predicate_list=[pred[fiveQ[0][1]],pred[fiveQ[1][1]],pred[fiveQ[2][1]],pred[fiveQ[3][1]],pred[fiveQ[4][1]]])
        fnrs[4].append(fnr)
        fprs[4].append(fpr) 
        eps[4].append(ep)
        entropies[4].append(computeMinEntropy(pred_ep)[0])

        sixQ = counts
        random.shuffle(sixQ)
        fnr,fpr,ep,pred_ep , pred_ep_dict,preds_a,preds_b , bi_total= ppwlmPROBE([sixQ[0][0],sixQ[1][0],sixQ[2][0],sixQ[3][0],sixQ[4][0],sixQ[5][0]],[th[sixQ[0][1]],th[sixQ[1][1]], th[sixQ[2][1]], th[sixQ[3][1]], th[sixQ[4][1]], th[sixQ[5][1]]],failingRate,[a_total[sixQ[0][1]],a_total[sixQ[1][1]],a_total[sixQ[2][1]],a_total[sixQ[3][1]],a_total[sixQ[4][1]],a_total[sixQ[5][1]]],[l[sixQ[0][1]],l[sixQ[1][1]],l[sixQ[2][1]],l[sixQ[3][1]],l[sixQ[4][1]],l[sixQ[5][1]]],[startingEp,startingEp,startingEp,startingEp,startingEp,startingEp],epsilonMax,m,type,predicate_list=[pred[sixQ[0][1]],pred[sixQ[1][1]],pred[sixQ[2][1]],pred[sixQ[3][1]],pred[sixQ[4][1]],pred[sixQ[5][1]]])
        fnrs[5].append(fnr)
        fprs[5].append(fpr)
        eps[5].append(ep)
        entropies[5].append(computeMinEntropy(pred_ep)[0])
    fnrs = [mean(fnr) for fnr in fnrs]
    fprs = [mean(fpr) for fpr in fprs]
    eps = [mean(ep) for ep in eps]
    entropies = [mean(ent) for ent in entropies]
    return eps,fnrs,fprs,entropies

def taxi(start_time,end_time,beta,u,m_i,e,type):
    pred1,counts1,tth1=query_taxi_count(start_time, end_time)
    pred2,counts2,tth2=query_taxi_fareamount(start_time, end_time)
    pred3,counts3,tth3=query_taxi_total_amount(start_time, end_time)
    pred4,counts4,tth4=query_taxi_count_flag(start_time, end_time)
    pred5,counts5,tth5=query_taxi_count_payment_type(start_time, end_time)
    pred6,counts6,tth6=query_taxi_count_location(start_time, end_time)
    
    pred = [pred1,pred2,pred3,pred4,pred5,pred6]
    counts = [(counts1,0),(counts2,1),(counts3,2),(counts4,3),(counts5,4),(counts6,5)]
    th=[tth1,tth2,tth3,tth4,tth5,tth6]

    a=0.12 if u==-1 else u/100
    failingRate = 0.005 if beta==-1 else beta
    startingEp=0.01 if e==-1 else e
    m=4 if m_i ==-1 else m_i

    a1= a*(max(counts1)-min(counts1))   
    a2= a*(max(counts2)-min(counts2))
    a3= a*(max(counts3)-min(counts3))
    a4= a*(max(counts4)-min(counts4))
    a5= a*(max(counts5)-min(counts5))
    a6= a*(max(counts6)-min(counts6))
    a_total = [a1,a2,a3,a4,a5,a6]
    
    l=[1]*len(a_total)
    epsilonMax= 9
    iter=100
    num_tests = 6

    if type == 2:
       eps,fns,fps,entropies = progCombine(counts,th,failingRate,a_total,l,epsilonMax,predicate_list=pred) 
    fnrs,fprs,eps,entropies= [[] for i in range(num_tests)],[[] for i in range(num_tests)],[[] for i in range(num_tests)],[[] for i in range(num_tests)]
    for i in range(iter):
        oneQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(2):
            fnr,fpr,ep,pred_ep, pred_ep_dict,preds_a,preds_b , bi_total= ppwlmPROBE([oneQ[0]],[th[oneQ[1]]],failingRate,[a_total[oneQ[1]]],[l[oneQ[1]]],[startingEp],epsilonMax,m,type,predicate_list=[pred[oneQ[1]]])
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[0].append(mean(fnrL))
        fprs[0].append(mean(fprL))
        eps[0].append(mean(epL))
        entropies[0].append(computeMinEntropy(pred_epL)[0])
        counts = counts[:oneQ[1]] + counts[oneQ[1]+1:]

        secondQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(2):
            fnr,fpr,ep,pred_ep, pred_ep_dict,preds_a,preds_b , bi_total= ppwlmPROBE([oneQ[0],secondQ[0]],[th[oneQ[1]],th[secondQ[1]]],failingRate,[a_total[oneQ[1]],a_total[secondQ[1]]],[l[oneQ[1]],l[secondQ[1]]],[startingEp]*2,epsilonMax,m,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]]])
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[1].append(mean(fnrL))
        fprs[1].append(mean(fprL))
        eps[1].append(mean(epL))
        entropies[1].append(computeMinEntropy(pred_epL)[0])
        counts = counts[:secondQ[1]] + counts[secondQ[1]+1:]

        thirdQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(2):
            fnr,fpr,ep,pred_ep, pred_ep_dict,preds_a,preds_b , bi_total= ppwlmPROBE([oneQ[0],secondQ[0],thirdQ[0]],[th[oneQ[1]],th[secondQ[1]],th[thirdQ[1]]],failingRate,[a_total[oneQ[1]],a_total[secondQ[1]],a_total[thirdQ[1]]],[l[oneQ[1]],l[secondQ[1]],l[thirdQ[1]]],[startingEp]*3,epsilonMax,m,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]],pred[thirdQ[1]]])  
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[2].append(mean(fnrL))
        fprs[2].append(mean(fprL))   
        eps[2].append(mean(epL))
        entropies[2].append(computeMinEntropy(pred_epL)[0])
        counts = counts[:thirdQ[1]] + counts[thirdQ[1]+1:]

        fourQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(2):
            fnr,fpr,ep,pred_ep, pred_ep_dict,preds_a,preds_b , bi_total= ppwlmPROBE([oneQ[0],secondQ[0],thirdQ[0],fourQ[0]],[th[oneQ[1]],th[secondQ[1]],th[thirdQ[1]],th[fourQ[1]]],failingRate,[a_total[oneQ[1]],a_total[secondQ[1]],a_total[thirdQ[1]],a_total[fourQ[1]]],[l[oneQ[1]],l[secondQ[1]],l[thirdQ[1]],l[fourQ[1]]],[startingEp]*4,epsilonMax,m,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]],pred[thirdQ[1]],pred[fourQ[1]]])  
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[3].append(mean(fnrL))
        fprs[3].append(mean(fprL))
        eps[3].append(mean(epL))
        entropies[3].append(computeMinEntropy(pred_epL)[0])
        counts = counts[:fourQ[1]] + counts[fourQ[1]+1:]

        fiveQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(2):
            fnr,fpr,ep,pred_ep, pred_ep_dict,preds_a,preds_b , bi_total= ppwlmPROBE([oneQ[0],secondQ[0],thirdQ[0],fourQ[0],fiveQ[0]],[th[oneQ[1]],th[secondQ[1]],th[thirdQ[1]],th[fourQ[1]],th[fiveQ[1]]],failingRate,[a_total[oneQ[1]],a_total[secondQ[1]],a_total[thirdQ[1]],a_total[fourQ[1]],a_total[fiveQ[1]]],[l[oneQ[1]],l[secondQ[1]],l[thirdQ[1]],l[fourQ[1]],l[fiveQ[1]]],[startingEp]*5,epsilonMax,m,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]],pred[thirdQ[1]],pred[fourQ[1]],pred[fiveQ[1]]])  
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[4].append(mean(fnrL))
        fprs[4].append(mean(fprL)) 
        eps[4].append(mean(epL))
        entropies[4].append(computeMinEntropy(pred_epL)[0])
        counts = counts[:fiveQ[1]] + counts[fiveQ[1]+1:]
      
        sixQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(2):
            fnr,fpr,ep,pred_ep, pred_ep_dict,preds_a,preds_b , bi_total= ppwlmPROBE([oneQ[0],secondQ[0],thirdQ[0],fourQ[0],fiveQ[0],sixQ[0]],[th[oneQ[1]],th[secondQ[1]],th[thirdQ[1]],th[fourQ[1]],th[fiveQ[1]],th[sixQ[1]]],failingRate,[a_total[oneQ[1]],a_total[secondQ[1]],a_total[thirdQ[1]],a_total[fourQ[1]],a_total[fiveQ[1]],a_total[sixQ[1]]],[l[oneQ[1]],l[secondQ[1]],l[thirdQ[1]],l[fourQ[1]],l[fiveQ[1]],l[sixQ[1]]],[startingEp]*6,epsilonMax,m,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]],pred[thirdQ[1]],pred[fourQ[1]],pred[fiveQ[1]],pred[sixQ[1]]])  
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[5].append(mean(fnrL))
        fprs[5].append(mean(fprL))
        eps[5].append(mean(epL))
        entropies[5].append(computeMinEntropy(pred_epL)[0])
       
    fnrs = [mean(fnr) for fnr in fnrs]
    fprs = [mean(fpr) for fpr in fprs]
    eps = [mean(ep) for ep in eps]
    entropies = [mean(ent) for ent in entropies]
    return eps,fnrs,fprs,entropies
