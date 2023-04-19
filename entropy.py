import math 
from scipy.stats import entropy
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
import numpy as np
 
class LRUCache:
 
    # initialising capacity
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity
 
    # we return the value of the key
    # that is queried in O(1) and return -1 if we
    # don't find the key in out dict / cache.
    # And also move the key to the end
    # to show that it was recently used.
    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        else:
            self.cache.move_to_end(key)
            return self.cache[key]
 
    # first, we add / update the key by conventional methods.
    # And also move the key to the end to show that it was recently used.
    # But here we will also check whether the length of our
    # ordered dictionary has exceeded our capacity,
    # If so we remove the first key (least recently used)
    def put(self, key: int, value: int) -> None:
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)
 
###Privacy Measurements
def computeEntropy(pList):
    e=entropy(pList)
    return round(float(e),12)

def ranges(epsList):
    lsum = sum([np.exp(eps) for eps in epsList])
    usum = sum([np.exp(-1.0 * eps) for eps in epsList])
   #print(lsum,usum)
    
    lList = []
    uList = []
    dList = []
    for eps in epsList:
        l = np.exp(-1.0 * eps)/lsum
       # print(np.exp(-1.0 * eps)/lsum,"vs", l)
        u = round(np.exp(eps)/usum,3)
        d = u-l
        if(u-l==0):
            l = round(np.exp(-1.0 * eps)/lsum,4)
            u = round(np.exp(eps)/usum,4)
        lList.append(l)
        uList.append(u)
        dList.append(d)
    return lList, uList, dList


num_reduced={0:0}
cacheMimEnt= LRUCache(100000)

def minEntRNonImrovedApprox(lList, uList, dList, s):
    CacheKey=str(lList)+str( uList)+str(s)
    if(cacheMimEnt.get(CacheKey) != -1):
        return cacheMimEnt.get(CacheKey)
    k = len(dList)
    if k == 1:
        sol = lList
        sol[0] = sol[0]+s
        return sol
    preSum = sum(dList[:k-1])
    lk = lList[k-1]
    uk = uList[k-1]
    dk = dList[k-1]
    sol1 = lList
    sol2 = lList
    if(round(s-dk,2)==0):
        s=dk
    if(round(s-preSum,2)==0):
        s=preSum
    if s >= max(preSum, dk):
        #print("A,s,preSum,dk",s,preSum,dk)
        sol1 = uList[:k-1]
        ak = s - preSum + lk
        sol1.append(ak)
        sol2 = minEntRNonImrovedApprox(lList[:k-1],uList[:k-1],dList[:k-1],s-dk)
        sol2.append(uk)
    elif s <= min(preSum,dk):
        #print("B,s,preSum,dk",s,preSum,dk,k)
        sol1 = lList[:k-1]
        sol1.append(lk+s)
        sol2 = minEntRNonImrovedApprox(lList[:k-1],uList[:k-1],dList[:k-1],s)
        sol2.append(lk)
    elif s <= dk and s >= preSum:
        #print("C,s,preSum,dk",s,preSum,dk)
        sol1 = lList[:k-1]
        sol1.append(s+lk)
        sol2 = uList[:k-1]
        sol2.append(lk+s-preSum)
    elif s > dk and s < preSum:
        #print("D,s,preSum,dk",s,preSum,dk,k)
        sol1 = minEntRNonImrovedApprox(lList[:k-1],uList[:k-1],dList[:k-1],s-dk)
        sol1.append(uk)
        sol2 = minEntRNonImrovedApprox(lList[:k-1],uList[:k-1],dList[:k-1],s)
        sol2.append(lk)
    else:
        print("uncovered cases:", uList, lList, s)
    ent1 = computeEntropy(sol1)
    ent2 = computeEntropy(sol2)
    if (ent1 > ent2):
        cacheMimEnt.put(CacheKey, sol2)
        return sol2
    else:
        cacheMimEnt.put(CacheKey,sol1)
        return sol1


def computeMinEntropy(ep_list):
    lList,uList,dList = ranges(sorted(ep_list))
    s = 1.0 - sum(lList)
    sol = minEntRNonImrovedApprox(lList, uList, dList, s)
    ent=float(computeEntropy(sol))/(computeEntropy([1.0/len(ep_list)]*len(ep_list)))
    return ent,sol

