import numpy as np
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
import queries.creds as creds

##per city
def sales_volume(start_time,end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    crnt_query = 'SELECT CITY, COUNT(*) as c from TurkishMarketSales where DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) >= "' + start_time + '" AND DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) <  "' + end_time + '" AND  (region NOT LIKE "Akdeniz") AND (region NOT LIKE "Ege") AND (region NOT LIKE "Karadeniz")  group by DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")), CITY'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()

    counts,predicate_list=[],[]
    test = []
    for t, count_trip in result_set:
        predicate_list.append(t)
        counts.append(int(count_trip))

    meann = mean(counts)
    std = stdev(counts)
    threshold =  meann+3*std
    cursor.close()
    cnx.close()
    counts = [x for _, x in sorted(zip(predicate_list, counts))]
    predicate_list = sorted(predicate_list)
    return predicate_list, counts, threshold

def sales_volume_marmara(start_time,end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    crnt_query = 'SELECT CITY, COUNT(*) as c from TurkishMarketSales where DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) >= "' + start_time + '" AND DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) <  "' + end_time + '" AND REGION="Marmara" AND  (region NOT LIKE "Akdeniz") AND (region NOT LIKE "Ege") AND (region NOT LIKE "Karadeniz") group by DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")), CITY'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()

    counts,predicate_list=[],[]
    for t, count_trip in result_set:
        predicate_list.append(t)
        counts.append(int(count_trip))
    meann = mean(counts)
    std = stdev(counts)
    threshold =  meann+3*std
    cursor.close()
    cnx.close()
    counts = [x for _, x in sorted(zip(predicate_list, counts))]
    predicate_list = sorted(predicate_list)
    return predicate_list, counts, threshold

def average_transaction_value(start_time,end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    
    crnt_query = 'SELECT CITY, COUNT(*) as c from TurkishMarketSales where DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) >= "' + start_time + '" AND DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) <  "' + end_time + '" AND  (BRANCH != "Adana Subesi") AND (BRANCH != "Ardahan Subesi") AND (BRANCH != "Manisa Subesi") AND  (region NOT LIKE "Akdeniz") AND (region NOT LIKE "Ege") AND (region NOT LIKE "Karadeniz") group by DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")), CITY'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()

    counts,predicate_list=[],[]
    for t, count_trip in result_set:
        predicate_list.append(t)
        predicate_list.append(t)
        counts.append(int(count_trip))

    meann = mean(counts)
    std = stdev(counts)
    threshold =  meann+3*std

    cursor.close()
    cnx.close()
    counts = [x for _, x in sorted(zip(predicate_list, counts))]
    predicate_list = sorted(predicate_list)
    return predicate_list, counts, threshold

def avg_transaction_value(start_time,end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    
    crnt_query = 'SELECT CITY, AVG(LINENETTOTAL*10) as avg from TurkishMarketSales where DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) >= "' + start_time + '" AND DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) <  "' + end_time + '" AND  (region NOT LIKE "Akdeniz") AND (region NOT LIKE "Ege") AND (region NOT LIKE "Karadeniz")  group by DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")), CITY'
    threshold_query ='select max(y.avg)-min(y.avg), avg(y.avg), std(y.avg) from (SELECT CITY, AVG(LINENETTOTAL*10) as avg from TurkishMarketSales where DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) >= "' + start_time + '" AND DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) <  "' + end_time + '"AND  (region NOT LIKE "Akdeniz") AND (region NOT LIKE "Ege") AND (region NOT LIKE "Karadeniz")  group by DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")), CITY) y'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()

    counts,predicate_list=[],[]
    for t, count_trip in result_set:
        predicate_list.append(t)
        counts.append(int(count_trip))
    meann = mean(counts)
    std = stdev(counts)
    threshold = meann+3*std

    cursor.close()
    cnx.close()
    counts = [x for _, x in sorted(zip(predicate_list, counts))]
    predicate_list = sorted(predicate_list)
    return predicate_list, counts, threshold

def distinct_customers(start_time,end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    
    crnt_query = 'SELECT CITY, COUNT(DISTINCT(CLIENTCODE)) as c from TurkishMarketSales where DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) >= "' + start_time + '" AND DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) <  "' + end_time + '" AND  (region NOT LIKE "Akdeniz") AND (region NOT LIKE "Ege") AND (region NOT LIKE "Karadeniz") group by DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")), CITY'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()

    counts,predicate_list=[],[]
    for t, count_trip in result_set:
        predicate_list.append(t)
        counts.append(int(count_trip))
    meann = mean(counts)
    std = stdev(counts)
    threshold =  meann+3*std
    cursor.close()
    cnx.close()
    counts = [x for _, x in sorted(zip(predicate_list, counts))]
    predicate_list = sorted(predicate_list)
    return predicate_list, counts, threshold

##count of product categories sold by city/branch per day
def category_count(start_time,end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB) 
    crnt_query = 'SELECT CITY, COUNT(DISTINCT(CATEGORY_NAME2)) as c from TurkishMarketSales where DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) >= "' + start_time + '" AND DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) <  "' + end_time + '" AND  (region NOT LIKE "Akdeniz") AND (region NOT LIKE "Ege") AND (region NOT LIKE "Karadeniz")  group by DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")), CITY'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()

    counts,predicate_list=[],[]
    for t, count_trip in result_set:
        predicate_list.append(t)
        counts.append(int(count_trip))
    meann = mean(counts)
    std = stdev(counts)
    threshold =  meann+3*std
    cursor.close()
    cnx.close()
    counts = [x for _, x in sorted(zip(predicate_list, counts))]
    predicate_list = sorted(predicate_list)
    return predicate_list, counts, threshold

def average_item_number(start_time,end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    crnt_query = 'SELECT CITY, AVG(AMOUNT) as c from TurkishMarketSales where DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) >= "' + start_time + '" AND DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) <  "' + end_time + '" AND  (region NOT LIKE "Akdeniz") AND (region NOT LIKE "Ege") AND (region NOT LIKE "Karadeniz")  group by DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")), CITY'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()

    counts,predicate_list=[],[]
    for t, count_trip in result_set:
        predicate_list.append(t)
        counts.append(int(count_trip))
    meann = mean(counts)
    std = stdev(counts)
    threshold =  meann+3*std
    cursor.close()
    cnx.close()
    counts = [x for _, x in sorted(zip(predicate_list, counts))]
    predicate_list = sorted(predicate_list)
    return predicate_list, counts, threshold

def sales_volume_women(start_time,end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB) 
    crnt_query = 'SELECT CITY, COUNT(*) as c from TurkishMarketSales where DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) >= "' + start_time + '" AND DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) <  "' + end_time + '" AND GENDER="K" AND  (region NOT LIKE "Akdeniz") AND (region NOT LIKE "Ege") AND (region NOT LIKE "Karadeniz")  group by DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")), CITY'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()

    counts,predicate_list=[],[]
    for t, count_trip in result_set:
        predicate_list.append(t)
        counts.append(int(count_trip))
    meann = mean(counts)
    std = stdev(counts)
    threshold =  meann+3*std
    cursor.close()
    cnx.close()
    counts = [x for _, x in sorted(zip(predicate_list, counts))]
    predicate_list = sorted(predicate_list)
    return predicate_list, counts, threshold


##per product
def transaction_per_category(start_time,end_time):
    ##Per cATEGORY (81) per day for 14 days 81x14=1134 predicates
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    crnt_query = 'SELECT CATEGORY_NAME2, COUNT(*) as c from TurkishMarketSales where DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) >= "' + start_time + '" AND DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) <  "' + end_time + '" AND  (region NOT LIKE "Akdeniz") AND (region NOT LIKE "Ege") AND (region NOT LIKE "Karadeniz") group by DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")), CATEGORY_NAME2' 
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()

    counts,predicate_list=[],[]
    for t, count_trip in result_set:
        predicate_list.append(t)
        counts.append(int(count_trip))
    meann = mean(counts)
    std = stdev(counts)
    threshold =  meann+3*std
    cursor.close()
    cnx.close()
    counts = [x for _, x in sorted(zip(predicate_list, counts))]
    predicate_list = sorted(predicate_list)
    return predicate_list, counts, threshold

def avg_total_per_category(start_time,end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB) 
    crnt_query = 'SELECT CATEGORY_NAME2, AVG(LINENETTOTAL) as c from TurkishMarketSales where DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) >= "' + start_time + '" AND DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) <  "' + end_time + '"AND  (region NOT LIKE "Akdeniz") AND (region NOT LIKE "Ege") AND (region NOT LIKE "Karadeniz")  group by DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")), CATEGORY_NAME2'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()

    counts,predicate_list=[],[]
    for t, count_trip in result_set:
        if t == None:
            continue
        predicate_list.append(t)
        counts.append(int(count_trip))
    meann = mean(counts)
    std = stdev(counts)
    threshold =  meann+3*std
    cursor.close()
    cnx.close()
    counts = [x for _, x in sorted(zip(predicate_list, counts))]
    predicate_list = sorted(predicate_list)
    return predicate_list, counts, threshold

def avg_itemprice_per_category(start_time,end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    crnt_query = 'SELECT CATEGORY_NAME2, AVG(price) as c from TurkishMarketSales where DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) >= "' + start_time + '" AND DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")) <  "' + end_time + '" AND  (region NOT LIKE "Akdeniz") AND (region NOT LIKE "Ege") AND (region NOT LIKE "Karadeniz") group by DATE(STR_TO_DATE(STARTDATE,"%c/%e/%y %k:%i")), CATEGORY_NAME2'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()

    counts,predicate_list=[],[]
    for t, count_trip in result_set:
        predicate_list.append(t)
        counts.append(int(count_trip))
    meann = mean(counts)
    std = stdev(counts)
    threshold =  meann+3*std
    cursor.close()
    cnx.close()
    counts = [x for _, x in sorted(zip(predicate_list, counts))]
    predicate_list = sorted(predicate_list)
    return predicate_list, counts, threshold
