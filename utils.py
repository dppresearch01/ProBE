from itertools import combinations
import math
from entropy import computeMinEntropy
import numpy as np

def get_numerator(a,i,a_total):
    prod = 1
    for j in range(len(a_total)):
        if j != i:
            prod*= a_total[j]
    return prod   
 
def get_denominator(a_total,alt,idx, s=[],o=[]):
    a_total2=[]
    idx_set = set(list(np.arange(len(a_total))))
    for i in range(len(a_total)):
        a_total2.append((i,a_total[i]))
    res = 0
    if len(a_total) == 1:
        res = 1
    elif len(a_total) == 2:
        res = sum(a_total)
    else:
        combi_len = len(a_total)-1
        combos = [list(c) for c in combinations(a_total,combi_len)]
        combos2 = [[combi[1] for combi in c] for c in combinations(a_total2,combi_len)]
        idx = [[combi[0] for combi in c] for c in combinations(a_total2,combi_len)]
        for i, c in enumerate(combos):
            missing_idx = list(idx_set - set(idx[i]))[0]
            if o != []:
                prod_o = [o[i] for i in idx[i]]
                res+= math.prod(prod_o)*math.prod(c)*s[missing_idx] if len(s) > 0 else math.prod(prod_o)*math.prod(c)
            else:
                res+= math.prod(c)*s[missing_idx] if len(s) > 0 else math.prod(c)
    return res

def intersection_test(lists):
    return list(set.intersection(*map(set,lists)))
    
def union_test(lists):
    return list(set.union(*map(set,lists)))

def getBi(all_a,a,idx,failingRate,o=[]):
    prod = get_numerator(a,idx,all_a)
    bi = (prod/get_denominator(all_a,False,idx,o=o))*failingRate
    return bi

def new_classify_and(c,t,selected,preds,pred_total,pred_list):
    ##preds ->preds_after_shift (noisy predicates that are POSITIVE) (INTERSECTed)
    ##pred_total ->preds_before_shift (same as predicate_list but INTERSECTed)
    ##pred_list -> predicate_list (list of all predicates) (list of lists)
    tp=[]
    tn=[]
    fp=[]
    fn=[]
    flags = []*len(pred_total)
    c_t_tuples=[]*len(pred_total)
    
    for i,p in enumerate(pred_total):
        c_t_tuples.append([])
        for j,pred in enumerate(pred_list):
            if p in pred:               
                idx = pred.index(p)
                c_t_tuples[i].append((c[j][idx], t[j][idx]))    
    for i,tup_list in enumerate(c_t_tuples):
        flags.append([])
        for tup in tup_list:
            flags[i].append(tup[0] > tup[1])
    for i in range(len(flags)):
        if all(flags[i]) and (pred_total[i] in preds):
            tp.append(i)
        elif all(flags[i]) and (pred_total[i] not in preds):
            fn.append(i)
        elif (all(flags[i]))== False and (pred_total[i] in preds):
            fp.append(i)
        elif (all(flags[i])) == False and (pred_total[i] not in preds):
            tn.append(i)
    return tp,tn,fp,fn

def new_classify_or(c,t,selected,preds,pred_total,pred_list):
    ##preds ->preds_after_shift (noisy predicates that are POSITIVE) (UNIONed)
    ## pred_total ->preds_before_shift (same as predicate_list but UNIONed)
    ##pred_list -> predicate_list (list of all predicates) (list of lists)
    tp=[]
    tn=[]
    fp=[]
    fn=[]
    flags = []*len(pred_total)
    c_t_tuples=[]*len(pred_total) 
    for i,p in enumerate(pred_total):
        c_t_tuples.append([])
        for j,pred in enumerate(pred_list):
            if p in pred:               
                idx = pred.index(p)
                c_t_tuples[i].append((c[j][idx], t[j][idx]))
   
    for i,tup_list in enumerate(c_t_tuples):
        flags.append([])
        for tup in tup_list:
            flags[i].append(tup[0] > tup[1])
    for i in range(len(flags)):
        if any(flags[i]) and (pred_total[i] in preds):
            tp.append(i)
        elif any(flags[i]) and (pred_total[i] not in preds):
            fn.append(i)
        elif (any(flags[i]))== False and (pred_total[i] in preds):
            fp.append(i)
        elif (any(flags[i])) == False and (pred_total[i] not in preds):
            tn.append(i)
    return tp,tn,fp,fn

def getEntropyArr(epNaive,epTS,len):
    entropyTSLM,entropyNaive = [],[]
    for ept in epTS:
        entropyTSLM.append(computeMinEntropy([ept]*len)[0])
    for epn in epNaive:
        entropyNaive.append(computeMinEntropy([epn]*len)[0])
    return entropyTSLM,entropyNaive
    

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
