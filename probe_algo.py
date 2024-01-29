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

from queries.queries_sales import sales_volume,average_transaction_value,distinct_customers,transaction_per_category,avg_total_per_category,avg_itemprice_per_category,category_count,average_item_number,sales_volume_women,sales_volume_marmara

from itertools import combinations
from queries.taxi_queries import query_taxi_count_location,query_taxi_count,query_taxi_fareamount,query_congestion_amount,query_taxi_tip_amount,query_taxi_tolls_amount,query_taxi_total_amount,query_taxi_count_payment_type,query_taxi_count_flag
from utils import new_classify_and, new_classify_or,classify, intersection_test,union_test,get_denominator,get_numerator, classify_mix,getBi


## early elimination of predicates in the uncertain region
## depends on which operator is ran
def earlyElimination(first_step_classes, type):
    idx_to_elim = []
    for i in range(len(first_step_classes[0])):
        class_per_idx = []
        for c in first_step_classes:
            class_per_idx.append(c[i])
        ## if conjunction then check if ONE is DECIDED FALSE -> 0 and ONE is undecided -> 2
        if type == 1 and 0 in class_per_idx and 2 in class_per_idx:
            idx_to_elim.append(i)

        ## if disjunction then check if ONE is TRUE
        elif type == 0 and 1 in class_per_idx and 2 in class_per_idx:
            idx_to_elim.append(i)
    return idx_to_elim

def estimateFPs(noisy_vals,idx_to_exclude,threshold,u,beta):
    potential_fps=[]
    fps_idx = []
    op_idx =[]
    o_p, o_pp, o_u, o_n = [],[],[],[]
    for i,n in enumerate(noisy_vals):
        if n > threshold:
            o_pp.append(n)
            
        if n > threshold - u:
            o_p.append(n)
            op_idx.append(i)
        if (n <= threshold and n>= threshold-u):
            potential_fps.append(n)
            fps_idx.append(i)
        if ( n < threshold - u):
            o_n.append(n)
    estimated_negatives = (len(o_n) - beta*len(noisy_vals))/(1-beta)
    o_u = list(set(o_p)-set(o_pp))
    fps = len(o_u)+len(o_pp)*beta
    fp_pc = fps /estimated_negatives 
    
    return potential_fps,fps_idx,op_idx,fp_pc,estimated_negatives
        
def secondStep(noisy_vals,vals_classified,idx_to_exclude, threshold,u,beta_i,alpha,l,idx,alt, n,counts,epsilon_max):
    potential_fps,fps_idx,opp_idx,fp_pc,estimated_negatives = estimateFPs(noisy_vals,idx_to_exclude,threshold,u,beta_i)
    limit = alpha
    if (fp_pc > limit):
        # 1. find how many fps to remove to be under limit
        # it's the percentage - limit * # elements
        fp_to_remove = int(math.ceil((fp_pc-limit)*len(noisy_vals)))
        # 2. choose u such that fp_to_remove are outside [t-2u,t]
        # thought: order noisy fps. get to fp_to_remove index and choose value between the two elements at that index
        sorted_fps = sorted(potential_fps)
        #cutoff is actually = t-u so we can get u=(t-cutoff)
        cutoff = sorted_fps[fp_to_remove-1]
        new_u = (threshold-cutoff)
        
        ##get updated beta_i using new uncertain region u
        selected, ep, noisy_vals_2,undecided,vals_classified = threshold_shift([counts[i] for i in fps_idx], [threshold]*len(potential_fps),new_u,beta_i,10,l)
        #replace the old noisy values of potential fps in noisy_vals by new noisy values
        for i, fp in enumerate(noisy_vals_2):
            noisy_vals[fps_idx[i]]=fp
        
        #get new positives
        selected =[]
        for i,n in enumerate(noisy_vals):
            if n > threshold - new_u:
                selected.append(i)
        p,i,o,update_fp,en = estimateFPs(noisy_vals,idx_to_exclude,threshold,u,beta_i)
        if update_fp > limit:
            return -1,[],-1
        return new_u, selected,ep  
    return 0,[],0

def probe(counts_total, t_total, failingRate,alpha, a_total,l_sensitivity_total, epsilonMax, iter=100,aes=[],og_idx=[], alt=False,predicate_list=[],combine=False,type=1, o =[]):
    b_total = []
    ep_list = []
    fp_list = []
    fn_list= []
    selected_returned = []
    positives, negatives = [],[]
    for i,a in enumerate(a_total) :
        if combine == True:
            prod = get_numerator(a_total[i],og_idx[i],aes)
            bi = 0
            if alt == True:
                bi = (prod/get_denominator(aes,alt,og_idx[i], o=o))*failingRate/2
            else:
                bi = (prod/get_denominator(aes,alt,og_idx[i],o=o))*failingRate/2
            b_total.append(bi)
        else:
            prod = get_numerator(a,i,a_total)
            bi = (prod/get_denominator(a_total,alt,i,o=o))*(failingRate/2)
            b_total.append(bi)

    b__total = [failingRate/len(aes)]*len(a_total) if combine==True else [failingRate/len(a_total)]*len(a_total) 
    preds_after_shift, preds_before_shift = [],[]
    ep_total = 0
    ep_total_2 = 0
    selected_total = []
    selected_total_2 = []
    early_stop = False
    predicate_total = []
    predicate_total_2 = []
    first_step_classes = []
    first_step_noisy_vals =[]
    stop_j = 0 
    second_step = False
    ##FIRST STEP/ITERATION OF THE ALGORITHM WITH DEFAULT U
    for j,count in enumerate(counts_total):
        ep_j_total = 0
        selected_j, ep_j, noisy_vals_j,undecided,vals_classified= threshold_shift(count, [t_total[j]]*len(count),a_total[j],b_total[j],epsilonMax,l_sensitivity_total[j])
        first_step_noisy_vals.append(noisy_vals_j)
        first_step_classes.append(vals_classified)
        ep_j_total+=ep_j
        ep_total+= ep_j_total
        selected_total.append(selected_j)

        if len(predicate_list) > 0 :
            if len(predicate_list[j]) > 0:
                predicate_total.append([predicate_list[j][p] for p in selected_j])

    if(ep_total > epsilonMax):
        print('Query Denied')
        return [[],[],0,0] 
    
    #pre-process
    # FIRST make sure noisy values/preds match for each sub-query
    max_len = max([len(i) for i in first_step_classes])
    for i,c in enumerate(first_step_classes):
        if len(c) < max_len:
            new_class_list = []
            for j,p in enumerate(predicate_list[0]):
                if p in predicate_list[i]:
                    #index where this p is within the smaller list
                    idx = predicate_list[i].index(p)
                    ele = c[idx]

                    new_class_list.append(ele)
                else:
                    new_class_list.append(3)
            first_step_classes[i] = new_class_list

    ##early elimination algorithm
    idx_to_elim = earlyElimination(first_step_classes,type)

    ##SECOND STEP/ITERATION OF THE ALGORITHM
    second_selecteds = []
    second_eps = []
    second_predicate_total = []
    for j,count in enumerate(counts_total):
        new_u, second_selected,second_ep = secondStep(first_step_noisy_vals[j],first_step_classes[j],idx_to_elim,t_total[j],a_total[j],b_total[j],alpha,l_sensitivity_total[j],j,alt, len(counts_total),counts_total[j],epsilonMax)
        ## alpha limit is hit, so query is denied
        if second_ep == -1:
            print("Query Denied.")
            return 0,0,0,[],[],[] 

        ## union the selected from this and the selected from previous phase
        second_selected = union_test([second_selected, selected_total[j]])
        second_selecteds.append(second_selected)
        second_predicate_total.append([predicate_list[j][p] for p in second_selected])
        second_eps.append(second_ep)
    
    

    ##second step was not run
    if (sum(second_eps) == 0):
        preds_after_shift = union_test(predicate_total) if type == 0 else intersection_test(predicate_total)
        selected_total = union_test(selected_total) if type == 0 else intersection_test(selected_total)
        preds_before_shift= union_test(predicate_list) if type == 0 else intersection_test(predicate_list)

        tp,tn,fp,fn = new_classify_and(counts_total,[[t_total[i]]*len(counts_total[i]) for i in range(len(t_total))],selected_total,preds_after_shift,preds_before_shift,predicate_list) if type == 1 else new_classify_or(counts_total,[[t_total[i]]*len(counts_total[i]) for i in range(len(t_total))],selected_total,preds_after_shift,preds_before_shift,predicate_list)
        positives = tp + fn

        negatives = tn + fp
        fpr =len(fp)/len(negatives) if len(negatives) > 0 else 0
        fnr =len(fn)/len(positives) if len(positives)>0 else 0
        fp_list.append(len(fp)/len(negatives) if len(negatives) > 0 else 0)
        fn_list.append(len(fn)/len(positives) if len(positives)>0 else 0)

        negatives = tn + fp
        fpr =len(fp)/len(negatives) if len(negatives) > 0 else 0
        fnr =len(fn)/len(positives) if len(positives)>0 else 0
        fp_list.append(len(fp)/len(negatives) if len(negatives) > 0 else 0)
        fn_list.append(len(fn)/len(positives) if len(positives)>0 else 0)

        return ep_total,fnr,fpr, selected_total, preds_after_shift,preds_before_shift
    
    ep_list.append(ep_total+sum(second_eps))
    selected = intersection_test(second_selecteds) if type == 1 else union_test(second_selecteds)
    selected_returned = selected

    preds_after_shift = intersection_test(second_predicate_total) if type == 1 else union_test(second_predicate_total)
    preds_before_shift = intersection_test(predicate_list) if type == 1 else union_test(predicate_list)
    
    tp,tn,fp,fn = new_classify_and(counts_total,[[t_total[i]]*len(counts_total[i]) for i in range(len(t_total))],selected,preds_after_shift,preds_before_shift,predicate_list) if type == 1 else new_classify_or(counts_total,[[t_total[i]]*len(counts_total[i]) for i in range(len(t_total))],selected,preds_after_shift,preds_before_shift,predicate_list)
    positives = tp + fn

    negatives = tn + fp
    fpr =len(fp)/len(negatives) if len(negatives) > 0 else 0
    fnr =len(fn)/len(positives) if len(positives)>0 else 0
    fp_list.append(len(fp)/len(negatives) if len(negatives) > 0 else 0)
    fn_list.append(len(fn)/len(positives) if len(positives)>0 else 0)

    return ep_total,fnr,fpr, selected, preds_after_shift,preds_before_shift

def threshold_shift(countsByPartition, thresholds, uncertainRegion, failingRate, epsilonMax, l_sensitivity):
    ep = l_sensitivity*np.log( 0.5 / failingRate) /uncertainRegion*1.0
    selected = []
    epList = []
    uncertainRegionList = []
    noisy_vals = []
    undecided =[]
    vals_classified =[] # 0 -> negative 1 -> positive 2 -> undecided
    fp_max_count = 0
    if(ep<epsilonMax):

        lap_noises = np.random.laplace(0, 1.0*l_sensitivity/ep, len(countsByPartition))
        for i in range(len(countsByPartition)):
            noisy_val = countsByPartition[i]+lap_noises[i]
            epList.append(ep)
            uncertainRegionList.append(uncertainRegion)
            noisy_vals.append(noisy_val)
            if(noisy_val > thresholds[i]):
                vals_classified.append(1)
            elif(noisy_val < thresholds[i]-2*uncertainRegion):
                vals_classified.append(0)
            else:
                if(noisy_val >= thresholds[i]-2*uncertainRegion and noisy_val <= thresholds[i]):
                    undecided.append(i)
                    vals_classified.append(2)

            if(noisy_val>thresholds[i]-uncertainRegion):
                    selected.append(i)
    return selected, ep,noisy_vals,undecided, vals_classified

def one_q(counts,t,failingRate,a,all_a,idx,l,epsilonMax,iter=100,alt=False,predicate_list=[],one=False):
    if(one):
        selected,ep,noisy_vals,undecided,vals_classified =threshold_shift(counts,[t]*len(counts), a,failingRate,epsilonMax,l)
        return selected,ep
    prod = get_numerator(a,idx,all_a)
    b1 = (prod/get_denominator(all_a,False,idx))*failingRate
    selected,ep,noisy_vals,undecided,vals_classified = threshold_shift(counts,[t]*len(counts), a,b1,epsilonMax,l)
    return selected,ep

def apportionment(beta, u_arr,l_arr,idx,alt):
    prod = get_numerator(u_arr[idx],idx,u_arr)
    bi = (prod/get_denominator(u_arr,alt,idx))*beta
    return bi

def query_sales(start_time,end_time,beta,alpha,type):
    pred1,counts1,th1 = sales_volume(start_time,end_time)
    pred2,counts2,th2 = average_transaction_value(start_time,end_time)
    pred3,counts3,th3 = distinct_customers(start_time,end_time)
    pred4,counts4,th4 = category_count(start_time,end_time)
    pred5,counts5,th5 = sales_volume_marmara(start_time,end_time)
    pred6,counts6,th6 = sales_volume_women(start_time,end_time)
    a=0.3

    pred = [pred1,pred2,pred3,pred4,pred5,pred6]
    counts = [(counts1,0),(counts2,1),(counts3,2),(counts4,3),(counts5,4),(counts6,5)]
    th=[th1,th2,th3,th4,th5,th6]

    failingRate = 0.05 if beta==-1 else beta
    epsilonMax= 20
    iter= 100

    a1= a*(max(counts1)-min(counts1)) 
    a2= a*(max(counts2)-min(counts2)) 
    a3 = a*(max(counts3)-min(counts3))
    a4= a*(max(counts4)-min(counts4))
    a5= a*(max(counts5)-min(counts5))
    a6= a*(max(counts6)-min(counts6))   
    a_total = [a1,a2,a3,a4,a5,a6]
    
    l_sensitivity1=1
    l_sensitivity2=1
    l_sensitivity3=1
    l_sensitivity4=1
    l_sensitivity5=1
    l_sensitivity6=1
    l = [l_sensitivity1,l_sensitivity2,l_sensitivity3,l_sensitivity4,l_sensitivity5,l_sensitivity6]
    num_tests = 6
    
    eps,fnrs,fprs = run_queries(counts,pred,th,a_total,l,failingRate,alpha,epsilonMax)
    return eps,fnrs,fprs

def query_taxi(start_time,end_time,beta,alpha,type):
    pred1,counts1,th1=query_taxi_count(start_time, end_time)
    pred2,counts2,th2=query_taxi_fareamount(start_time, end_time)
    pred3,counts3,th3=query_taxi_total_amount(start_time, end_time)
    pred4,counts4,th4=query_taxi_count_flag(start_time, end_time)
    pred5,counts5,th5=query_taxi_count_payment_type(start_time, end_time)
    pred6,counts6,th6=query_taxi_count_location(start_time, end_time)
    pred = [pred1,pred2,pred3,pred4,pred5,pred6]
    counts = [(counts1,0),(counts2,1),(counts3,2),(counts4,3),(counts5,4),(counts6,5)]
    th=[th1,th2,th3,th4,th5,th6]
    a = 0.3
    ranges = np.array([max(counts1)-min(counts1),max(counts2)-min(counts2)])

    a1= a*(max(counts1)-min(counts1))   
    a2= a*(max(counts2)-min(counts2))
    a3= a*(max(counts3)-min(counts3))
    a4= a*(max(counts4)-min(counts4))
    a5= a*(max(counts5)-min(counts5))
    a6= a*(max(counts6)-min(counts6))
    a_total = [a1,a2,a3,a4,a5,a6]
   
    l_sensitivity1=1

    l_sensitivity2=1
    l_sensitivity3=1
    l_sensitivity4=1
    l_sensitivity5=1
    l_sensitivity6 =1
    l = [l_sensitivity1,l_sensitivity2,l_sensitivity3,l_sensitivity4,l_sensitivity5,l_sensitivity6]


    failingRate = 0.05 if beta==-1 else beta
    epsilonMax= 9
    eps,fnrs,fprs = run_queries(counts,pred,th,a_total,l,failingRate,alpha,epsilonMax)
    return eps,fnrs,fprs

def run_queries(counts,pred,th,a_total,l,failingRate,alpha,epsilonMax):
    num_tests =6
    fnrs,fprs,eps= [[] for i in range(num_tests)],[[] for i in range(num_tests)],[[] for i in range(num_tests)]
    for i in range(100):
        oneQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(50):
            ep,fnr,fpr,selected, preds_a,preds_b = probe([oneQ[0]],[th[oneQ[1]]],failingRate,alpha,[a_total[oneQ[1]]],[l[oneQ[1]]],epsilonMax,type,predicate_list=[pred[oneQ[1]]])
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            
        fnrs[0].append(mean(fnrL))
        fprs[0].append(mean(fprL))
        eps[0].append(mean(epL))
        counts = counts[:oneQ[1]] + counts[oneQ[1]+1:]

        
        secondQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(50):
            ep,fnr,fpr,selected, preds_a,preds_b= probe([oneQ[0],secondQ[0]],[th[oneQ[1]],th[secondQ[1]]],failingRate,alpha,[a_total[oneQ[1]],a_total[secondQ[1]]],[l[oneQ[1]],l[secondQ[1]]],epsilonMax,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]]])
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
        fnrs[1].append(mean(fnrL))
        fprs[1].append(mean(fprL))
        eps[1].append(mean(epL))
        counts = counts[:secondQ[1]] + counts[secondQ[1]+1:]

        
        thirdQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(50):
            ep,fnr,fpr,selected, preds_a,preds_b= probe([oneQ[0],secondQ[0],thirdQ[0]],[th[oneQ[1]],th[secondQ[1]],th[thirdQ[1]]],failingRate,alpha,[a_total[oneQ[1]],a_total[secondQ[1]],a_total[thirdQ[1]]],[l[oneQ[1]],l[secondQ[1]],l[thirdQ[1]]],epsilonMax,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]],pred[thirdQ[1]]])  
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
           
        fnrs[2].append(mean(fnrL))
        fprs[2].append(mean(fprL))   
        eps[2].append(mean(epL))
        counts = counts[:thirdQ[1]] + counts[thirdQ[1]+1:]

       
        fourQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(50):
            ep,fnr,fpr,selected, preds_a,preds_b= probe([oneQ[0],secondQ[0],thirdQ[0],fourQ[0]],[th[oneQ[1]],th[secondQ[1]],th[thirdQ[1]],th[fourQ[1]]],failingRate,alpha,[a_total[oneQ[1]],a_total[secondQ[1]],a_total[thirdQ[1]],a_total[fourQ[1]]],[l[oneQ[1]],l[secondQ[1]],l[thirdQ[1]],l[fourQ[1]]],epsilonMax,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]],pred[thirdQ[1]],pred[fourQ[1]]])  
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            
        fnrs[3].append(mean(fnrL))
        fprs[3].append(mean(fprL))
        eps[3].append(mean(epL))
        counts = counts[:fourQ[1]] + counts[fourQ[1]+1:]

        fiveQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(50):
            ep,fnr,fpr, selected,preds_a,preds_b= probe([oneQ[0],secondQ[0],thirdQ[0],fourQ[0],fiveQ[0]],[th[oneQ[1]],th[secondQ[1]],th[thirdQ[1]],th[fourQ[1]],th[fiveQ[1]]],failingRate,alpha,[a_total[oneQ[1]],a_total[secondQ[1]],a_total[thirdQ[1]],a_total[fourQ[1]],a_total[fiveQ[1]]],[l[oneQ[1]],l[secondQ[1]],l[thirdQ[1]],l[fourQ[1]],l[fiveQ[1]]],epsilonMax,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]],pred[thirdQ[1]],pred[fourQ[1]],pred[fiveQ[1]]])  
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            
        fnrs[4].append(mean(fnrL))
        fprs[4].append(mean(fprL)) 
        eps[4].append(mean(epL))
        counts = counts[:fiveQ[1]] + counts[fiveQ[1]+1:]
      
        sixQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(50):
            ep,fnr,fpr,selected, preds_a,preds_b= probe([oneQ[0],secondQ[0],thirdQ[0],fourQ[0],fiveQ[0],sixQ[0]],[th[oneQ[1]],th[secondQ[1]],th[thirdQ[1]],th[fourQ[1]],th[fiveQ[1]],th[sixQ[1]]],failingRate,alpha,[a_total[oneQ[1]],a_total[secondQ[1]],a_total[thirdQ[1]],a_total[fourQ[1]],a_total[fiveQ[1]],a_total[sixQ[1]]],[l[oneQ[1]],l[secondQ[1]],l[thirdQ[1]],l[fourQ[1]],l[fiveQ[1]],l[sixQ[1]]],epsilonMax,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]],pred[thirdQ[1]],pred[fourQ[1]],pred[fiveQ[1]],pred[sixQ[1]]])  
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            
        fnrs[5].append(mean(fnrL))
        fprs[5].append(mean(fprL))
        eps[5].append(mean(epL))

        return eps,fnrs,fprs 
