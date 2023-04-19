
import numpy as np
import math 
from decimal import Decimal, Context
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib as mpl
from statistics import mean,stdev,mode,median
import pickle
import os
import query
import datetime
import time
import random
from collections import OrderedDict
import entropy
from itertools import combinations
from queries.queries_sales import sales_volume,average_transaction_value,distinct_customers,transaction_per_category,avg_total_per_category,avg_itemprice_per_category,category_count,average_item_number,sales_volume_women,sales_volume_marmara
from queries.taxi_queries import query_taxi_count_location,query_taxi_count, query_congestion_amount, query_taxi_fareamount, query_taxi_tip_amount,query_taxi_tolls_amount,query_taxi_total_amount,query_taxi_count_payment_type,query_taxi_count_flag
from utils import intersection_test,union_test,classify_mix,get_denominator,get_numerator,getBi,new_classify_and,new_classify_or

dict_ent = entropy.LRUCache(1000)
matches={}
matches[0]=0


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


def integrandF1(o,ep_1,ep_2):
    F1 = lambda z:(ep_1*ep_2/4.0) * (math.exp(ep_2*(o-z))/(ep_1+ep_2) +
                                     (math.exp(ep_2*(o-z))-math.exp(ep_1*(o-z)))/(ep_1-ep_2) +
                                     math.exp(ep_1*(o-z))/(ep_1+ep_2))
        
    return F1
def integrandF2(o,ep_1,ep_2):
    
    F2 = lambda z:(ep_1*ep_2/4.0) * (math.exp(ep_2*(z-o))/(ep_1+ep_2) +
                                     (math.exp(ep_2*(z-o))-math.exp(ep_1*(z-o)))/(ep_1-ep_2)*1.0 +
                                     math.exp(ep_1*(z-o))/(ep_1+ep_2))
    return F2
def probx_gt_t(o,t,ep_1,ep_2):   # o is observered noisy count with ep_1, calculates prob that observed noisy count with ep_2 noise will be > t
    F1=integrandF1(o,ep_1,ep_2)
    F2=integrandF2(o,ep_1,ep_2)
    if(t>=o):
        ans,er=integrate.quad(F1, t, np.inf)
    else:
        ans1,er1=integrate.quad(F2, t,o)
        ans2,er2=integrate.quad(F1, o,np.inf)
        ans=ans1+ans2
    
    return ans


def chooseEpsfailingRateEstimatedEntropy(j,ep_start,maxSteps,maxFineSteps,failingRate,remainingfailingRate,selected,epList,ep_prev,ep_est,lap_noises,countsByPartition,thresholds,l):
    beta_counter=1
    failingRate_j=failingRate/(maxSteps*1.0)
    base=pow(1.0*ep_est/ep_start,(1.0/(maxSteps-1)))
    ep_j=ep_start*pow(base,j-1)
    ep_jminus1=ep_prev
    ep=ep_prev
    opt_j=j
    opt_ent=0
    ep_opt=ep_est
    optfailingRate = remainingfailingRate 
       
    while(ep<ep_est and remainingfailingRate>0 and remainingfailingRate>((failingRate/maxSteps))): 
        for fs in range(maxFineSteps):
            ep_range=ep_j-ep_jminus1
            if(ep_range<=0):
                break
            ep=ep+1.0*ep_range/maxFineSteps
            exp_pos=0
            for i in range(len(lap_noises)):
                if(i in selected):
                    shiftedUncertainRegion = l*np.log( 0.5 / failingRate_j) / ep*1.0
                    prob_pos_eliminated=probx_gt_t(countsByPartition[i]+lap_noises[i],thresholds[i]+shiftedUncertainRegion,ep_prev,ep)
                    prob_pos_selected=probx_gt_t(countsByPartition[i]+lap_noises[i],thresholds[i]-shiftedUncertainRegion,ep_prev,ep)
                    exp_pos+=(prob_pos_selected-prob_pos_eliminated)
            cost=list(epList)
            cost=sorted(cost)
            for i in selected:
                cost.remove(ep_prev)
                cost.append(ep)
            if math.isnan(exp_pos):
                exp_pos=0
            for i in range(0,int(round(exp_pos))):
                cost.remove(ep)
                cost.append(ep_est)
            found=dict_ent.get(str(cost))
            if   found==-1:  
                new_ent=entropy.computeMinEntropy(cost)[0]
                dict_ent.put(str(cost),new_ent)
            else:
                matches[0]+=1
                new_ent=found
            if(opt_ent<new_ent and int(round(exp_pos))!=len(selected)):
                opt_ent=new_ent
                ep_opt=ep
                optfailingRate=1.0*(failingRate/maxSteps)*beta_counter
                opt_j=j
        beta_counter+=1
        j+=1
        ep_jminus1=ep_j
        ep_j=ep_start*pow(base,j-1)  
    return ep_opt, optfailingRate,opt_j

def aggregatePredList(ep_lists):
    length = max([len(ep_list) for ep_list in ep_lists])
    final_ep_list = [0]*length
    for i in range(length):
        for ep_list in ep_lists:
            if len(ep_list) <= i:

                final_ep_list[i] += 0
            else:
                final_ep_list[i] += ep_list[i]
    return final_ep_list


def multiDDPWLM(counts,thresholds,failingRate,a_s,l_sens,eps_start,epsilonMax,m,mf,type,aes=[],og_idx=[],alt=False,predicate_list=[],combine=False):
    epsilon_f = 0
    ep_total = 0
    ep_list = []
    predicate_total=[]
    selected_total = []
    predicate_eps = []
    eps_max = []
    stop_j = -1
    leftover_idx = -1
    leftover_idx_og = -1
    new_as=[]
    preds_after_shift=[]
    preds_before_shift=[]
    predicate_total = []
    ep_pred_lists = []
    bi_total=0
    for i,count in enumerate(counts):
        bi = 0
        if leftover_idx != -1 :
            if combine==True:
                new_as = aes[:leftover_idx_og-1]+aes[leftover_idx_og:]
                new_idx = i-leftover_idx
                prod = get_numerator(a_s[i],new_idx,new_as)*l_sens[i]
                bi = (prod/get_denominator(new_as,alt,new_idx,s=l_sens))*failingRate
            else:
                new_idx = i-leftover_idx
                prod = get_numerator(a_s[i],new_idx,new_as)*l_sens[i]
                bi = (prod/get_denominator(new_as,alt,new_idx,s=l_sens))*failingRate
        if combine == True:
            prod = get_numerator(a_s[i],og_idx[i],aes)*l_sens[i]
            bi = (prod/get_denominator(aes,alt,og_idx[i],s=l_sens))*failingRate
            bi_total+=bi
        else:
            prod = get_numerator(a_s[i],i,a_s)*l_sens[i]
            bi = (prod/get_denominator(a_s,alt,i,s=l_sens))*failingRate
            bi_total+=bi
   
        selected, ep,ep_list,usedBeta = DDPWLM(m,mf,eps_start[i],count,[thresholds[i]]*len(count),a_s[i],bi,epsilonMax,l_sens[i])
        ep_pred_lists.append(ep_list)
        if (usedBeta != 0):
            failingRate = failingRate - usedBeta
            leftover_idx = i+1
            if combine == True:
                leftover_idx_og = og_idx[i]+1
            new_as = a_s[leftover_idx:]
        preds = [predicate_list[i][k] for k in selected]
        predicate_total.append(preds)
        selected_total.append(selected)
        ep_total += ep
        # if conj and no selected just stop
        if len(selected) == 0 and type== 1:
            # early stop 
            break
    
    ep_list = aggregatePredList(ep_pred_lists)
    # now we can classify
    
    preds_after_shift = union_test(predicate_total) if type == 0 else intersection_test(predicate_total)
    selected_total = union_test(selected_total) if type == 0 else intersection_test(selected_total)
    preds_before_shift= union_test(predicate_list) if type == 0 else intersection_test(predicate_list)
    tp,tn,fp,fn,selected = [],[],[],[],[]

    if type == 1:
        tp,tn,fp,fn = new_classify_and(counts,[[thresholds[i]]*len(counts[i]) for i in range(len(thresholds))],selected,preds_after_shift,preds_before_shift,predicate_list)
    else:
        tp,tn,fp,fn= new_classify_or(counts,[[thresholds[i]]*len(counts[i]) for i in range(len(thresholds))],selected,preds_after_shift,preds_before_shift,predicate_list)

    positives = tp + fn
    negatives = tn + fp
    fnr = len(fn)/len(positives) if len(positives) > 0 else 0
    fpr = len(fp)/len(negatives) if len(negatives) > 0 else 0
   
    return fnr,fpr,ep_total,ep_list,preds_after_shift,preds_before_shift,bi_total,selected_total

###Data Dependent Predicate-wise Laplace Mechanism
def DDPWLM(maxSteps, maxFineSteps,ep_start, countsByPartition, thresholds, uncertainRegion, failingRate, epsilonMax, l): 
    ep_est = l* np.log( 0.5 / (failingRate/maxSteps)) / uncertainRegion*1.0
    remainingfailingRate=failingRate
    returnBeta = 0
    selected=[]
    positives=[]
    epList = np.zeros(len(countsByPartition))  #eps used by each count
    ep_total= 0
    left_over_beta = 0
    if(ep_est<epsilonMax):
        ep = ep_start
        lap_noises = np.random.laplace(0, 1.0/ep, len(countsByPartition))
        for i in range(len(countsByPartition)):
            shiftedUncertainRegion = l*np.log( 0.5*maxSteps / failingRate) / ep*1.0
            if(countsByPartition[i]+lap_noises[i]>thresholds[i]+shiftedUncertainRegion):
                positives.append(i)
            elif(countsByPartition[i]+lap_noises[i]>thresholds[i]-shiftedUncertainRegion):
                selected.append(i)
            epList[i] = ep  
        
        j=1
        remainingfailingRate-=failingRate/maxSteps*1.0
        if len(selected) == 0:
            returnBeta = failingRate-remainingfailingRate
        while (ep<=ep_est and remainingfailingRate>0 and len(selected)!=0):
            j=j+1
            prev_ep=ep
            ep,failingRate_j,j_traversed=chooseEpsfailingRateEstimatedEntropy(j,ep_start,maxSteps,maxFineSteps,failingRate,remainingfailingRate,selected,epList,prev_ep,ep_est,lap_noises,countsByPartition,thresholds,l)
            lap_noises=[noise_down(lap_noise, prev_ep, ep) for lap_noise in lap_noises]  
            shiftedUncertainRegion = l*np.log( 0.5 / failingRate_j) / ep*1.0
            for i in range(len(countsByPartition)):           
                if(i in selected):
                    if( countsByPartition[i]+lap_noises[i]>thresholds[i]+shiftedUncertainRegion):
                        epList[i]=ep 
                        positives.append(i)
                        selected.remove(i)
                    elif( countsByPartition[i]+lap_noises[i]>thresholds[i]-shiftedUncertainRegion):
                        epList[i]=ep 
                    else:
                        selected.remove(i)
                        epList[i]=ep
            remainingfailingRate-= failingRate_j
        ep_total= ep
    returnBeta = failingRate-remainingfailingRate      
    return positives+selected, ep_total,epList,returnBeta


def v2distribute(counts_total,t_total,failingRate,a_total,l_sensitivity_total,epsilonMax,predicate_list=[]):
    eps,fns,fps,entropies = [],[],[],[]
    epsD,fnsD,fpsD,entropiesD = [],[],[],[]
    startingEp = 0.01
    m=4
    mf=3
    for i in range(100):
        #NOT DISTRIBUTED
        ep_total = 0
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

        bi = getBi(aes,aes[2],2,failingRate)
        pred3, ep3,ep_list3,remainingBeta = DDPWLM(m,mf,startingEp,counts[2],[ts[2]]*len(counts[2]),aes[2],bi,epsilonMax,ls[2])
        ep_total+=ep3

        if len(pred3) == 0:
            #early stop
            tp,tn,fp,fn = classify_mix([counts[2]],[ts[2]],selected3,1,preds[2])
            positives= tp+fn
            negatives= tn+fp
            fps.append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns.append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps.append(ep_total)
            entropies.append(entropy.computeMinEntropy(ep_preds3)[0])
        else:
            fnr12,fpr12,ep12,pred_ep12,pred_a12,pred_b12 , bi_total ,selected_total= multiDDPWLM(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],[startingEp]*2,epsilonMax,m,mf,0,aes,og_idx=[0,1],predicate_list=preds[0:2],combine=True)
            ep_total+= ep12
            preds_after_shift = intersection_test([pred_a12,pred3])
            preds_before_shift= intersection_test([pred_b12,preds[2]])
            ep_list = aggregatePredList([pred_ep12,ep_list3])
            
            tp,tn,fp,fn = classify_mix(counts,ts,preds_after_shift,1,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps.append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns.append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps.append(ep_total)
            entropies.append(entropy.computeMinEntropy(ep_list)[0])

        #DISTRIBUTED
        ep_total = 0
        #Q1 N Q2 
        fnr12,fpr12,ep12,pred_ep12,pred_a12,pred_b12 , bi_total ,selected_total= multiDDPWLM(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],[startingEp]*2,epsilonMax,m,mf,1,aes,og_idx=[0,1],predicate_list=preds[0:2],combine=True)
        ep_total+= ep12
        #Q1 N Q3 
        fnr13,fpr13,ep13,pred_ep13,pred_a13,pred_b13 , bi_total ,selected13= multiDDPWLM([counts[0],counts[2]],[ts[0],ts[2]],failingRate,[aes[0],aes[2]],[ls[0],ls[2]],[startingEp]*2,epsilonMax,m,mf,1,aes,og_idx=[0,2],predicate_list=[preds[0],preds[2]],combine=True)
        ep_total+= ep13
        ep_list = aggregatePredList([pred_ep12,pred_ep13])
        #classify
        preds_after_shift = union_test([pred_a12,pred_a13])
        preds_before_shift = union_test([pred_b12,pred_b13])
        tp,tn,fp,fn = classify_mix(counts,ts,preds_after_shift,1,preds_before_shift,distrib=True)
        positives= tp+fn
        negatives= tn+fp
        fpsD.append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
        fnsD.append(len(fn)/len(positives) if len(positives)> 0 else 0)
        epsD.append(ep_total)
        entropiesD.append(entropy.computeMinEntropy(ep_list)[0])

    return mean(eps),mean(fns),mean(fps), mean(entropies),mean(epsD),mean(fnsD),mean(fpsD), mean(entropiesD)

def distribute(counts_total,t_total,failingRate,a_total,l_sensitivity_total,epsilonMax,predicate_list=[]):
    #(Q1 U Q2) N (Q1 U Q3)
    eps,fns,fps,pred_eps,entropies = [],[],[],[],[]
    startingEp = 0.01
    m=4
    mf =3
    for i in range(20):
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
        fnr12,fpr12,ep12,pred_ep12,pred_a12,pred_b12 , bi_total ,selected12= multiDDPWLM(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],[startingEp]*2,epsilonMax,m,mf,0,aes,og_idx=[0,1],predicate_list=preds[0:2],combine=True,alt=True)
        ep_total+= ep12

        #Q1 U Q3
        fnr13,fpr13,ep13,pred_ep13,pred_a13,pred_b13 , bi_total ,selected13= multiDDPWLM([counts[0],counts[2]],[ts[0],ts[2]],failingRate,[aes[0],aes[2]],[ls[0],ls[2]],[startingEp]*2,epsilonMax,m,mf,0,aes,og_idx=[0,2],predicate_list=[preds[0],preds[2]],combine=True,alt=True)
        ep_total+= ep13
        ep_list = aggregatePredList([pred_ep12,pred_ep13])

        #classify  
        selected_total = union_test([selected13,selected12])
        preds_before_shift= union_test([pred_b12,pred_b13])
        tp,tn,fp,fn = classify_mix(counts,ts,selected_total,0,preds_before_shift,distrib=True)
        positives= tp+fn
        negatives= tn+fp
        fps.append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
        fns.append(len(fn)/len(positives) if len(positives)> 0 else 0)
        eps.append(ep_total)
        entropies.append(entropy.computeMinEntropy(list(ep_list))[0])
    return mean(eps),mean(fns),mean(fps), mean(entropies)
def combineV2(counts_og,t_total,failingRate,a_total,l_sensitivity_total,epsilonMax,predicate_list=[],vary=False):
    eps,fns,fps,pred_eps,entropies= [[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)],[[] for i in range(6)]
    startingEp = 0.1
    m=4
    mf =3
    if vary == True:
        for i in range(20):
            counts_total = list(counts_og)
            q1 = counts_total[0]
            q2 = counts_total[1]
            #for n=3 we do N U
            q3 = counts_total[2]
            ep_total = 0
            q_i = [q1,q2,q3]
            idx = [q[1] for q in q_i]
            counts = [counts_og[j][0] for j in idx]
            ts = [t_total[j] for j in idx]
            aes = [a_total[j] for j in idx]
            ls = [l_sensitivity_total[j] for j in idx]
            preds = [predicate_list[j] for j in idx]
            # conj first q1 N q2
            fnr12,fpr12,ep12,pred_ep12,pred_a12,pred_b12 , bi_total ,selected_total= multiDDPWLM(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],[startingEp]*2,epsilonMax,m,mf,1,aes,og_idx=[0,1],predicate_list=preds[0:2],combine=True)
            ep_total+= ep12
            
            #single Q3
            bi = failingRate-bi_total
            selected3, ep3,ep_list3,remainingBeta = DDPWLM(m,mf,startingEp,counts[2],[ts[2]]*len(counts[2]),aes[2],bi,epsilonMax,ls[2])
            ep_total += ep3
            pred3 = [preds[2][k] for k  in selected3]
    
            preds_after_shift = union_test([pred_a12,pred3])
            selected_total = union_test([selected3,selected_total])
            preds_before_shift= union_test([pred_b12,preds[2]])
            ep_list = aggregatePredList([pred_ep12,ep_list3])
            tp,tn,fp,fn = classify_mix(counts,ts,selected_total,0,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[2].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[2].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[2].append(ep_total)
            entropies[2].append(entropy.computeMinEntropy(list(ep_list))[0]) 
        return mean(eps[2]),mean(fns[2]),mean(fps[2]), mean(entropies[2])
    for i in range(100):
        #first for n-1 we do one_q
        counts_total = list(counts_og)
        q1 = random.choice(counts_total)
        q_i = q1      
        idx = q_i[1]
        counts = [q_i[0]]
        ts = [t_total[idx]]
        aes = [a_total[idx]]
        ls = [l_sensitivity_total[idx]]
        preds = [predicate_list[idx]]
        fnr,fpr,ep,pred_ep,pred_a,pred_b,bi_total ,selected_total= multiDDPWLM(counts,ts,failingRate,aes,ls,[startingEp],epsilonMax,m,mf,0,aes, predicate_list=preds)
        fns[0].append(fnr)
        fps[0].append(fpr)
        eps[0].append(ep)
        entropies[0].append(entropy.computeMinEntropy(pred_ep)[0])
        counts_total.remove(q1)

        # for n=2 we do either U or N
        q2 = random.choice(counts_total)
        combos = [list(c) for c in combinations(counts_total,2)]
        q_i = [q1,q2]
        idx = [q[1] for q in q_i]
        counts = [counts_og[j][0] for j in idx]
        ts = [t_total[j] for j in idx]
        aes = [a_total[j] for j in idx]
        ls = [l_sensitivity_total[j] for j in idx]
        preds = [predicate_list[j] for j in idx]
        funcs = [0,1]
        fnr,fpr,ep,pred_ep,pred_a,pred_b , bi_total ,selected_total= multiDDPWLM(counts,ts,failingRate,aes,ls,[startingEp]*2,epsilonMax,m,mf,random.choice(funcs), predicate_list=preds)
        eps[1].append(ep)
        fps[1].append(fpr)
        fns[1].append(fnr)
        entropies[1].append(entropy.computeMinEntropy(pred_ep)[0])
        counts_total.remove(q2)

        #for n=3 we do N U
        q3 = random.choice(counts_total)
        ep_total = 0
        combos = [list(c) for c in combinations(counts_total,3)]
        q_i = [q1,q2,q3]
        idx = [q[1] for q in q_i]
        counts = [counts_og[j][0] for j in idx]
        ts = [t_total[j] for j in idx]
        aes = [a_total[j] for j in idx]
        ls = [l_sensitivity_total[j] for j in idx]
        preds = [predicate_list[j] for j in idx]
        # conj first q1 N q2
        fnr12,fpr12,ep12,pred_ep12,pred_a12,pred_b12 , bi_total ,selected_total= multiDDPWLM(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],[startingEp]*2,epsilonMax,m,mf,1,aes,og_idx=[0,1],predicate_list=preds[0:2],combine=True)
        ep_total+= ep12
        #single Q3
        bi = getBi(aes,aes[2],2,failingRate)
        selected3, ep3,ep_list3,remainingBeta = DDPWLM(m,mf,startingEp,counts[2],[ts[2]]*len(counts[2]),aes[2],bi,epsilonMax,ls[2])
        ep_total += ep3
        pred3 = [preds[2][k] for k  in selected3]
        preds_after_shift = union_test([pred_a12,pred3])
        selected_total = union_test([selected3,selected_total])
        preds_before_shift= union_test([pred_b12,preds[2]])
        ep_list = aggregatePredList([pred_ep12,ep_list3])
        tp,tn,fp,fn = classify_mix(counts,ts,selected_total,0,preds_before_shift)
        positives= tp+fn
        negatives= tn+fp
        fps[2].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
        fns[2].append(len(fn)/len(positives) if len(positives)> 0 else 0)
        eps[2].append(ep_total)
        entropies[2].append(entropy.computeMinEntropy(list(ep_list))[0])
        counts_total.remove(q3)

        # for n=4 we do N N U or N U U
        ep_total = 0
        q4 = random.choice(counts_total)
        combos = [list(c) for c in combinations(counts_total,4)]
        q_i = [q1,q2,q3,q4]
        idx = [q[1] for q in q_i]
        counts = [counts_og[j][0] for j in idx]
        ts = [t_total[j] for j in idx]
        aes = [a_total[j] for j in idx]
        ls = [l_sensitivity_total[j] for j in idx]
        preds = [predicate_list[j] for j in idx]
        flag = random.choice([0,1])
        if flag == 0: #  N N U
            fnr123,fpr123,ep123,pred_ep123,pred_a123,pred_b123 , bi_total ,selected_total= multiDDPWLM(counts[0:3],ts[0:3],failingRate,aes[0:3],ls[0:3],[startingEp]*3,epsilonMax,m,mf,1,aes,og_idx=[0,1,2],predicate_list=preds[0:3],combine=True)
            ep_total+= ep123
            bi = getBi(aes,aes[3],3,failingRate)
            selected3, ep3,ep_list3,remainingBeta = DDPWLM(m,mf,startingEp,counts[3],[ts[3]]*len(counts[3]),aes[3],bi,epsilonMax,ls[3])
            ep_total += ep3
            pred3 = [preds[3][k] for k  in selected3]
            preds_after_shift = union_test([pred_a12,pred3])
            preds_before_shift= union_test([pred_b12,preds[3]])
            ep_list = aggregatePredList([pred_ep12,ep_list3])
            selected_total = union_test([selected3,selected_total])
            tp,tn,fp,fn = classify_mix(counts,ts,selected_total,0,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[3].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[3].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[3].append(ep_total)
            entropies[3].append(entropy.computeMinEntropy(list(ep_list))[0])     
        else: # N U U
            # conj first q1 N q2
            fnr12,fpr12,ep12,pred_ep12,pred_a12,pred_b12 , bi_total ,selected_total12= multiDDPWLM(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],[startingEp]*2,epsilonMax,m,mf,1,aes,og_idx=[0,1],predicate_list=preds[0:2],combine=True)
            ep_total+= ep12
            fnr34,fpr34,ep34,pred_ep34,pred_a34,pred_b34 , bi_total ,selected_total34= multiDDPWLM(counts[2:4],ts[2:4],failingRate,aes[2:4],ls[2:4],[startingEp]*2,epsilonMax,m,mf,0,aes,og_idx=[2,3], predicate_list=preds[2:4],combine=True)
            ep_total+= ep34
            preds_after_shift = union_test([pred_a12,pred_a34])
            preds_before_shift = union_test([pred_b12,pred_b34])
            ep_list = aggregatePredList([pred_ep12,pred_ep34])
            selected_total = union_test([selected_total12,selected_total34])
            tp,tn,fp,fn = classify_mix(counts,ts,selected_total,1,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[3].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[3].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[3].append(ep_total)
            entropies[3].append(entropy.computeMinEntropy(list(ep_list))[0])
        counts_total.remove(q4)

        # for n=5 we do N N U U / N N N U / N U U U
        ep_total = 0
        q5 = random.choice(counts_total)
        combos = [list(c) for c in combinations(counts_total,5)]
        q_i = [q1,q2,q3,q4,q5]
        idx = [q[1] for q in q_i]
        counts = [counts_og[j][0] for j in idx]
        ts = [t_total[j] for j in idx]
        aes = [a_total[j] for j in idx]
        ls = [l_sensitivity_total[j] for j in idx]
        preds = [predicate_list[j] for j in idx]
        flag = random.choice([0,1,2])
        if flag == 0: # N N N U
            fnr123,fpr123,ep123,pred_ep123,pred_a123,pred_b123 , bi_total ,selected_total= multiDDPWLM(counts[0:4],ts[0:4],failingRate,aes[0:4],ls[0:4],[startingEp]*4,epsilonMax,m,mf,1,aes,og_idx=[0,1,2,3],predicate_list=preds[0:4],combine=True)
            ep_total+= ep123
            bi = getBi(aes,aes[4],4,failingRate)
            selected3, ep3,ep_list3,remainingBeta = DDPWLM(m,mf,startingEp,counts[4],[ts[4]]*len(counts[4]),aes[4],bi,epsilonMax,ls[4])
            ep_total += ep3
            pred3 = [preds[4][k] for k  in selected3]
            preds_after_shift = union_test([pred_a12,pred3])
            preds_before_shift= union_test([pred_b12,preds[4]])
            ep_list = aggregatePredList([pred_ep12,ep_list3])

            selected_total = union_test([selected3,selected_total])
            tp,tn,fp,fn = classify_mix(counts,ts,selected_total,0,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[4].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[4].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[4].append(ep_total)
            entropies[4].append(entropy.computeMinEntropy(list(ep_list))[0])
        
        elif flag == 1: # N N U U 
             # conj first q1 N q2
            fnr12,fpr12,ep12,pred_ep12,pred_a12,pred_b12 , bi_total ,selected_total12= multiDDPWLM(counts[0:3],ts[0:3],failingRate,aes[0:3],ls[0:3],[startingEp]*3,epsilonMax,m,mf,1,aes,og_idx=[0,1,2],predicate_list=preds[0:3],combine=True)
            ep_total+= ep12
            fnr34,fpr34,ep34,pred_ep34,pred_a34,pred_b34 , bi_total ,selected_total34= multiDDPWLM(counts[3:5],ts[3:5],failingRate,aes[3:5],ls[3:5],[startingEp]*2,epsilonMax,m,mf,0,aes,og_idx=[3,4], predicate_list=preds[3:5],combine=True)
            ep_total+= ep34
            preds_after_shift = union_test([pred_a12,pred_a34])
            preds_before_shift = union_test([pred_b12,pred_b34])
            ep_list = aggregatePredList([pred_ep12,pred_ep34])
            selected_total = union_test([selected_total12,selected_total34])
            tp,tn,fp,fn = classify_mix(counts,ts,selected_total,1,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[4].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[4].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[4].append(ep_total)
            entropies[4].append(entropy.computeMinEntropy(list(ep_list))[0])
       
        else: # N U U U 
            # conj first q1 N q2
            fnr12,fpr12,ep12,pred_ep12,pred_a12,pred_b12 , bi_total ,selected_total12= multiDDPWLM(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],[startingEp]*2,epsilonMax,m,mf,1,aes,og_idx=[0,1],predicate_list=preds[0:2],combine=True)
            ep_total+= ep12
            fnr34,fpr34,ep34,pred_ep34,pred_a34,pred_b34 , bi_total ,selected_total34= multiDDPWLM(counts[2:5],ts[2:5],failingRate,aes[2:5],ls[2:5],[startingEp]*3,epsilonMax,m,mf,0,aes,og_idx=[2,3,4], predicate_list=preds[2:5],combine=True)
            ep_total+= ep34
            preds_after_shift = union_test([pred_a12,pred_a34])
            preds_before_shift = union_test([pred_b12,pred_b34])
            ep_list = aggregatePredList([pred_ep12,pred_ep34])
            selected_total = union_test([selected_total12,selected_total34])
            tp,tn,fp,fn = classify_mix(counts,ts,selected_total,2,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[4].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[4].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[4].append(ep_total)
            entropies[4].append(entropy.computeMinEntropy(list(ep_list))[0])
        counts_total.remove(q5)

        # for n=6 we do N N N N U / N N N U U / N N U U U / N U U U U
        ep_total = 0
        q6 = random.choice(counts_total)
        combos = [list(c) for c in combinations(counts_total,6)]
        q_i = q_i = [q1,q2,q3,q4,q5,q6]
        idx = [q[1] for q in q_i]
        counts = [counts_og[j][0] for j in idx]
        ts = [t_total[j] for j in idx]
        aes = [a_total[j] for j in idx]
        ls = [l_sensitivity_total[j] for j in idx]
        preds = [predicate_list[j] for j in idx]
        flag = random.choice([0,1,2,3])
        if flag == 0: # N N N N U
            fnr123,fpr123,ep123,pred_ep123,pred_a123,pred_b123 , bi_total ,selected_total= multiDDPWLM(counts[0:5],ts[0:5],failingRate,aes[0:5],ls[0:5],[startingEp]*5,epsilonMax,m,mf,1,aes,og_idx=[0,1,2,3,4],predicate_list=preds[0:5],combine=True)
            ep_total+= ep123
            bi = getBi(aes,aes[5],5,failingRate)
            selected3, ep3,ep_list3,remainingBeta = DDPWLM(m,mf,startingEp,counts[5],[ts[5]]*len(counts[5]),aes[5],bi,epsilonMax,ls[5])
            ep_total += ep3
            pred3 = [preds[5][k] for k  in selected3]
            preds_after_shift = union_test([pred_a12,pred3])
            preds_before_shift= union_test([pred_b12,preds[4]])
            ep_list = aggregatePredList([pred_ep12,ep_list3])

            selected_total = union_test([selected3,selected_total])
            tp,tn,fp,fn = classify_mix(counts,ts,selected_total,0,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[5].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[5].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[5].append(ep_total)
            entropies[5].append(entropy.computeMinEntropy(list(ep_list))[0])
        
        elif flag == 1: # N N N U U 
            # conj first q1 N q2
            fnr12,fpr12,ep12,pred_ep12,pred_a12,pred_b12 , bi_total ,selected_total12= multiDDPWLM(counts[0:4],ts[0:4],failingRate,aes[0:4],ls[0:4],[startingEp]*4,epsilonMax,m,mf,1,aes,og_idx=[0,1,2,3],predicate_list=preds[0:4],combine=True)
            ep_total+= ep12
            fnr34,fpr34,ep34,pred_ep34,pred_a34,pred_b34 , bi_total ,selected_total34= multiDDPWLM(counts[4:6],ts[4:6],failingRate,aes[4:6],ls[4:6],[startingEp]*2,epsilonMax,m,mf,0,aes,og_idx=[4,5], predicate_list=preds[4:6],combine=True)
            ep_total+= ep34
            preds_after_shift = union_test([pred_a12,pred_a34])
            preds_before_shift = union_test([pred_b12,pred_b34])
            ep_list = aggregatePredList([pred_ep12,pred_ep34])
            selected_total = union_test([selected_total12,selected_total34])
            tp,tn,fp,fn = classify_mix(counts,ts,selected_total,1,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[5].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[5].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[5].append(ep_total)
            entropies[5].append(entropy.computeMinEntropy(list(ep_list))[0])
       
        elif flag == 2: # N N U U U
             # conj first q1 N q2
            fnr12,fpr12,ep12,pred_ep12,pred_a12,pred_b12 , bi_total ,selected_total12= multiDDPWLM(counts[0:3],ts[0:3],failingRate,aes[0:3],ls[0:3],[startingEp]*3,epsilonMax,m,mf,1,aes,og_idx=[0,1,2],predicate_list=preds[0:3],combine=True)
            ep_total+= ep12
            fnr34,fpr34,ep34,pred_ep34,pred_a34,pred_b34 , bi_total ,selected_total34= multiDDPWLM(counts[3:6],ts[3:6],failingRate,aes[3:6],ls[3:6],[startingEp]*3,epsilonMax,m,mf,0,aes,og_idx=[3,4,5], predicate_list=preds[3:6],combine=True)
            ep_total+= ep34
            preds_after_shift = union_test([pred_a12,pred_a34])
            preds_before_shift = union_test([pred_b12,pred_b34])
            ep_list = aggregatePredList([pred_ep12,pred_ep34])
            selected_total = union_test([selected_total12,selected_total34])
            tp,tn,fp,fn = classify_mix(counts,ts,selected_total,2,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[5].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[5].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[5].append(ep_total)
            entropies[5].append(entropy.computeMinEntropy(list(ep_list))[0])
       
        else: # N U U U U
              # conj first q1 N q2
            fnr12,fpr12,ep12,pred_ep12,pred_a12,pred_b12 , bi_total ,selected_total12= multiDDPWLM(counts[0:2],ts[0:2],failingRate,aes[0:2],ls[0:2],[startingEp]*2,epsilonMax,m,mf,1,aes,og_idx=[0,1],predicate_list=preds[0:2],combine=True)
            ep_total+= ep12
            fnr34,fpr34,ep34,pred_ep34,pred_a34,pred_b34 , bi_total ,selected_total34= multiDDPWLM(counts[2:6],ts[2:6],failingRate,aes[2:6],ls[2:6],[startingEp]*4,epsilonMax,m,mf,0,aes,og_idx=[2,3,4,5], predicate_list=preds[2:6],combine=True)
            ep_total+= ep34
            preds_after_shift = union_test([pred_a12,pred_a34])
            preds_before_shift = union_test([pred_b12,pred_b34])
            ep_list = aggregatePredList([pred_ep12,pred_ep34])
            selected_total = union_test([selected_total12,selected_total34])
            tp,tn,fp,fn = classify_mix(counts,ts,selected_total,3,preds_before_shift)
            positives= tp+fn
            negatives= tn+fp
            fps[5].append(len(fp)/len(negatives) if len(negatives)> 0 else 0)
            fns[5].append(len(fn)/len(positives) if len(positives)> 0 else 0)
            eps[5].append(ep_total)
            entropies[5].append(entropy.computeMinEntropy(list(ep_list))[0])
         
    fns = [mean(fnr) for fnr in fns]
    fps = [mean(fpr) for fpr in fps]
    eps = [mean(ep) for ep in eps]
    entropies = [mean(ent) for ent in entropies]
    return eps,fns,fps, entropies

def sales(start_time,end_time,beta,u,m_i,f,e,type):
    pred1,counts1,th1 = sales_volume(start_time,end_time)
    pred2,counts2,th2 = average_transaction_value(start_time,end_time)
    pred3,counts3,th3 = distinct_customers(start_time,end_time)
    pred4,counts4,th4 = category_count(start_time,end_time)
    pred5,counts5,th5 = sales_volume_marmara(start_time,end_time)
    pred6,counts6,th6 = sales_volume_women(start_time,end_time)
    pred = [pred1,pred2,pred3,pred4,pred5,pred6]
    counts = [(counts1,0),(counts2,1),(counts3,2),(counts4,3),(counts5,4),(counts6,5)]
    th=[tth1,tth2,tth3,tth4,tth5,tth6]
    
    a=0.12 if u==-1 else u/100
    failingRate = 0.005 if beta==-1 else beta
    startingEp=0.01 if e==-1 else e
    m=4 if m_i ==-1 else m_i
    mf =3  if f==-1 else f

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
    num_tests = 6
    
    if type == 2:
        eps,fnrs,fprs, entropies = combineV2(counts,th,failingRate,a_total,l,epsilonMax,predicate_list=pred)
        return eps,fnrs,fprs,entropies
    fnrs,fprs,eps,entropies= [[] for i in range(num_tests)],[[] for i in range(num_tests)],[[] for i in range(num_tests)],[[] for i in range(num_tests)]
    for i in range(1):
        counts_i = list(counts) 
        oneQ = random.choice(counts_i)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(5):
            fnr,fpr,ep,pred_ep, preds_a,preds_b , bi_total ,selected_total= multiDDPWLM([oneQ[0]],[th[oneQ[1]]],failingRate,[a_total[oneQ[1]]],[l[oneQ[1]]],[startingEp],epsilonMax,m,mf,type,predicate_list=[pred[oneQ[1]]])
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[0].append(mean(fnrL))
        fprs[0].append(mean(fprL))
        eps[0].append(mean(epL))
        entropies[0].append(entropy.computeMinEntropy(pred_epL)[0])
        counts_i.remove(oneQ)

        
        secondQ = random.choice(counts_i)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(5):
            fnr,fpr,ep,pred_ep, preds_a,preds_b , bi_total ,selected_total= multiDDPWLM([oneQ[0],secondQ[0]],[th[oneQ[1]],th[secondQ[1]]],failingRate,[a_total[oneQ[1]],a_total[secondQ[1]]],[l[oneQ[1]],l[secondQ[1]]],[startingEp]*2,epsilonMax,m,mf,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]]])
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[1].append(mean(fnrL))
        fprs[1].append(mean(fprL))
        eps[1].append(mean(epL))
        entropies[1].append(entropy.computeMinEntropy(pred_epL)[0])
        counts_i.remove(secondQ)

        thirdQ = random.choice(counts_i)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(5):
            fnr,fpr,ep,pred_ep, preds_a,preds_b , bi_total ,selected_total= multiDDPWLM([oneQ[0],secondQ[0],thirdQ[0]],[th[oneQ[1]],th[secondQ[1]],th[thirdQ[1]]],failingRate,[a_total[oneQ[1]],a_total[secondQ[1]],a_total[thirdQ[1]]],[l[oneQ[1]],l[secondQ[1]],l[thirdQ[1]]],[startingEp]*3,epsilonMax,m,mf,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]],pred[thirdQ[1]]])  
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[2].append(mean(fnrL))
        fprs[2].append(mean(fprL))   
        eps[2].append(mean(epL))
        entropies[2].append(entropy.computeMinEntropy(pred_epL)[0])
        counts_i.remove(thirdQ)
 
        fourQ = random.choice(counts_i)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
      
        for i in range(5):
            fnr,fpr,ep,pred_ep, preds_a,preds_b , bi_total ,selected_total= multiDDPWLM([oneQ[0],secondQ[0],thirdQ[0],fourQ[0]],[th[oneQ[1]],th[secondQ[1]],th[thirdQ[1]],th[fourQ[1]]],failingRate,[a_total[oneQ[1]],a_total[secondQ[1]],a_total[thirdQ[1]],a_total[fourQ[1]]],[l[oneQ[1]],l[secondQ[1]],l[thirdQ[1]],l[fourQ[1]]],[startingEp]*4,epsilonMax,m,mf,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]],pred[thirdQ[1]],pred[fourQ[1]]])  
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[3].append(mean(fnrL))
        fprs[3].append(mean(fprL))
        eps[3].append(mean(epL))
        entropies[3].append(entropy.computeMinEntropy(pred_epL)[0])
        counts_i.remove(fourQ)
      
        fiveQ = random.choice(counts_i)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(5):
            fnr,fpr,ep,pred_ep, preds_a,preds_b , bi_total ,selected_total= multiDDPWLM([oneQ[0],secondQ[0],thirdQ[0],fourQ[0],fiveQ[0]],[th[oneQ[1]],th[secondQ[1]],th[thirdQ[1]],th[fourQ[1]],th[fiveQ[1]]],failingRate,[a_total[oneQ[1]],a_total[secondQ[1]],a_total[thirdQ[1]],a_total[fourQ[1]],a_total[fiveQ[1]]],[l[oneQ[1]],l[secondQ[1]],l[thirdQ[1]],l[fourQ[1]],l[fiveQ[1]]],[startingEp]*5,epsilonMax,m,mf,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]],pred[thirdQ[1]],pred[fourQ[1]],pred[fiveQ[1]]])  
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[4].append(mean(fnrL))
        fprs[4].append(mean(fprL)) 
        eps[4].append(mean(epL))
        entropies[4].append(entropy.computeMinEntropy(pred_epL)[0])
        counts_i.remove(fiveQ)
    
        sixQ = random.choice(counts_i)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(5):
            fnr,fpr,ep,pred_ep, preds_a,preds_b , bi_total ,selected_total= multiDDPWLM([oneQ[0],secondQ[0],thirdQ[0],fourQ[0],fiveQ[0],sixQ[0]],[th[oneQ[1]],th[secondQ[1]],th[thirdQ[1]],th[fourQ[1]],th[fiveQ[1]],th[sixQ[1]]],failingRate,[a_total[oneQ[1]],a_total[secondQ[1]],a_total[thirdQ[1]],a_total[fourQ[1]],a_total[fiveQ[1]],a_total[sixQ[1]]],[l[oneQ[1]],l[secondQ[1]],l[thirdQ[1]],l[fourQ[1]],l[fiveQ[1]],l[sixQ[1]]],[startingEp]*6,epsilonMax,m,mf,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]],pred[thirdQ[1]],pred[fourQ[1]],pred[fiveQ[1]],pred[sixQ[1]]])  
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[5].append(mean(fnrL))
        fprs[5].append(mean(fprL))
        eps[5].append(mean(epL))
        entropies[5].append(entropy.computeMinEntropy(pred_epL)[0])
    
    fnrs = [mean(fnr) for fnr in fnrs]
    fprs = [mean(fpr) for fpr in fprs]
    eps = [mean(ep) for ep in eps]
    entropies = [mean(ent) for ent in entropies]
   
    return eps,fnrs,fprs,entropies

def taxi(start_time,end_time,beta,u,m_i,f,e,type):
    pred1,counts1,tth1=query_taxi_count(start_time, end_time)
    pred2,counts2,tth2=query_taxi_fareamount(start_time, end_time)
    pred3,counts3,tth3=query_taxi_total_amount(start_time, end_time)
    pred4,counts4,tth4=query_taxi_count_flag(start_time, end_time)
    pred5,counts5,tth5=query_taxi_count_payment_type(start_time, end_time)
    pred6,counts6,tth6=query_taxi_count_location(start_time, end_time)
    
    counts_list = [counts1,counts2,counts3,counts4,counts5,counts6]
    pred = [pred1,pred2,pred3,pred4,pred5,pred6]
    counts = [(counts1,0),(counts2,1),(counts3,2),(counts4,3),(counts5,4),(counts6,5)]
    th=[tth1,tth2,tth3,tth4,tth5,tth6]
    
    a=0.12 if u==-1 else u/100
    failingRate = 0.005 if beta==-1 else beta
    startingEp=0.01 if e==-1 else e
    m=4 if m_i ==-1 else m_i
    mf=3  if f==-1 else f 

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
    num_tests = 6

    if type == 2:
        eps,fnrs,fprs, entropies = combineV2(counts,th,failingRate,a_total,l,epsilonMax,predicate_list=pred)
        return eps,fnrs,fprs,entropies
    fnrs,fprs,eps,entropies= [[] for i in range(num_tests)],[[] for i in range(num_tests)],[[] for i in range(num_tests)],[[] for i in range(num_tests)]
    for i in range(100):
        oneQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(5):
            fnr,fpr,ep,pred_ep, preds_a,preds_b , bi_total ,selected_total= multiDDPWLM([oneQ[0]],[th[oneQ[1]]],failingRate,[a_total[oneQ[1]]],[l[oneQ[1]]],[startingEp],epsilonMax,m,mf,type,predicate_list=[pred[oneQ[1]]])
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[0].append(mean(fnrL))
        fprs[0].append(mean(fprL))
        eps[0].append(mean(epL))
        entropies[0].append(entropy.computeMinEntropy(pred_epL)[0])
        counts = counts[:oneQ[1]] + counts[oneQ[1]+1:]

        
        secondQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(5):
            fnr,fpr,ep,pred_ep, preds_a,preds_b , bi_total ,selected_total= multiDDPWLM([oneQ[0],secondQ[0]],[th[oneQ[1]],th[secondQ[1]]],failingRate,[a_total[oneQ[1]],a_total[secondQ[1]]],[l[oneQ[1]],l[secondQ[1]]],[startingEp]*2,epsilonMax,m,mf,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]]])
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[1].append(mean(fnrL))
        fprs[1].append(mean(fprL))
        eps[1].append(mean(epL))
        entropies[1].append(entropy.computeMinEntropy(pred_epL)[0])
        counts = counts[:secondQ[1]] + counts[secondQ[1]+1:]

        
        thirdQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(5):
            fnr,fpr,ep,pred_ep, preds_a,preds_b , bi_total ,selected_total= multiDDPWLM([oneQ[0],secondQ[0],thirdQ[0]],[th[oneQ[1]],th[secondQ[1]],th[thirdQ[1]]],failingRate,[a_total[oneQ[1]],a_total[secondQ[1]],a_total[thirdQ[1]]],[l[oneQ[1]],l[secondQ[1]],l[thirdQ[1]]],[startingEp]*3,epsilonMax,m,mf,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]],pred[thirdQ[1]]])  
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[2].append(mean(fnrL))
        fprs[2].append(mean(fprL))   
        eps[2].append(mean(epL))
        entropies[2].append(entropy.computeMinEntropy(pred_epL)[0])
        counts = counts[:thirdQ[1]] + counts[thirdQ[1]+1:]

       
        fourQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(5):
            fnr,fpr,ep,pred_ep, preds_a,preds_b , bi_total ,selected_total= multiDDPWLM([oneQ[0],secondQ[0],thirdQ[0],fourQ[0]],[th[oneQ[1]],th[secondQ[1]],th[thirdQ[1]],th[fourQ[1]]],failingRate,[a_total[oneQ[1]],a_total[secondQ[1]],a_total[thirdQ[1]],a_total[fourQ[1]]],[l[oneQ[1]],l[secondQ[1]],l[thirdQ[1]],l[fourQ[1]]],[startingEp]*4,epsilonMax,m,mf,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]],pred[thirdQ[1]],pred[fourQ[1]]])  
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[3].append(mean(fnrL))
        fprs[3].append(mean(fprL))
        eps[3].append(mean(epL))
        entropies[3].append(entropy.computeMinEntropy(pred_epL)[0])
        counts = counts[:fourQ[1]] + counts[fourQ[1]+1:]

        fiveQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(5):
            fnr,fpr,ep,pred_ep, preds_a,preds_b , bi_total ,selected_total= multiDDPWLM([oneQ[0],secondQ[0],thirdQ[0],fourQ[0],fiveQ[0]],[th[oneQ[1]],th[secondQ[1]],th[thirdQ[1]],th[fourQ[1]],th[fiveQ[1]]],failingRate,[a_total[oneQ[1]],a_total[secondQ[1]],a_total[thirdQ[1]],a_total[fourQ[1]],a_total[fiveQ[1]]],[l[oneQ[1]],l[secondQ[1]],l[thirdQ[1]],l[fourQ[1]],l[fiveQ[1]]],[startingEp]*5,epsilonMax,m,mf,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]],pred[thirdQ[1]],pred[fourQ[1]],pred[fiveQ[1]]])  
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[4].append(mean(fnrL))
        fprs[4].append(mean(fprL)) 
        eps[4].append(mean(epL))
        entropies[4].append(entropy.computeMinEntropy(pred_epL)[0])
        counts = counts[:fiveQ[1]] + counts[fiveQ[1]+1:]
      
        sixQ = random.choice(counts)
        fnrL,fprL,epL,pred_epL = [],[],[],[]
        for i in range(5):
            fnr,fpr,ep,pred_ep, preds_a,preds_b , bi_total ,selected_total= multiDDPWLM([oneQ[0],secondQ[0],thirdQ[0],fourQ[0],fiveQ[0],sixQ[0]],[th[oneQ[1]],th[secondQ[1]],th[thirdQ[1]],th[fourQ[1]],th[fiveQ[1]],th[sixQ[1]]],failingRate,[a_total[oneQ[1]],a_total[secondQ[1]],a_total[thirdQ[1]],a_total[fourQ[1]],a_total[fiveQ[1]],a_total[sixQ[1]]],[l[oneQ[1]],l[secondQ[1]],l[thirdQ[1]],l[fourQ[1]],l[fiveQ[1]],l[sixQ[1]]],[startingEp]*6,epsilonMax,m,mf,type,predicate_list=[pred[oneQ[1]],pred[secondQ[1]],pred[thirdQ[1]],pred[fourQ[1]],pred[fiveQ[1]],pred[sixQ[1]]])  
            fnrL.append(fnr)
            fprL.append(fpr)
            epL.append(ep)
            pred_epL = pred_ep
        fnrs[5].append(mean(fnrL))
        fprs[5].append(mean(fprL))
        eps[5].append(mean(epL))
        entropies[5].append(entropy.computeMinEntropy(pred_epL)[0])
    fnrs = [mean(fnr) for fnr in fnrs]
    fprs = [mean(fpr) for fpr in fprs]
    eps = [mean(ep) for ep in eps]
    entropies = [mean(ent) for ent in entropies]
    return eps,fnrs,fprs,entropies