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
import queries.creds as creds

def query_taxi_count(start_time, end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    crnt_query = 'select floor(PULocationID/8)  as hashed_id, count(*) as c  from taxi_dataset_march_20 where DATE(tpep_pickup_datetime) >= "' + start_time + '" and DATE(tpep_pickup_datetime) < "' + end_time + '" group by DATE(tpep_pickup_datetime), hashed_id'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()
   
    counts = []
    predicate_list = []
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

def query_taxi_count_location(start_time, end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    crnt_query = 'select floor(PULocationID/8)  as hashed_id, count(*) as c  from taxi_dataset_march_20 where DATE(tpep_pickup_datetime) >= "' + start_time + '" and DATE(tpep_pickup_datetime) < "' + end_time + '" and floor(DOLocationID/8) = 20 group by DATE(tpep_pickup_datetime), hashed_id'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()
    
    counts = []
    predicate_list = []
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


def query_taxi_count_payment_type(start_time, end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    crnt_query = 'select floor(PULocationID/8)  as hashed_id, count(*) as c  from taxi_dataset_march_20 where DATE(tpep_pickup_datetime) >= "' + start_time + '" and DATE(tpep_pickup_datetime) < "' + end_time + '" and payment_type=1 group by DATE(tpep_pickup_datetime), hashed_id'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()
    counts = []

    predicate_list = []
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

def query_taxi_count_flag(start_time, end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    crnt_query = 'select floor(PULocationID/8)  as hashed_id, count(*) as c  from taxi_dataset_march_20 where DATE(tpep_pickup_datetime) >= "' + start_time + '" and DATE(tpep_pickup_datetime) < "' + end_time + '"  and store_and_fwd_flag="N" group by DATE(tpep_pickup_datetime), hashed_id'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()
    counts = []
    predicate_list = []
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


def query_taxi_fareamount(start_time, end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    test_query = 'select floor(PULocationID/8)  as hashed_id, count(*) as c  from taxi_dataset_march_20 where DATE(tpep_pickup_datetime) >= "' + start_time + '" and DATE(tpep_pickup_datetime) < "' + end_time + '"and fare_amount>10 group by DATE(tpep_pickup_datetime), hashed_id'
    cursor = cnx.cursor()
    cursor.execute(test_query)
    result_set = cursor.fetchall()

    counts = []
    predicate_list = []
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

def query_taxi_total_amount(start_time, end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    test_query = 'select floor(PULocationID/8)  as hashed_id, count(*) as c  from taxi_dataset_march_20 where DATE(tpep_pickup_datetime) >= "' + start_time + '" and DATE(tpep_pickup_datetime) < "' + end_time + '"and total_amount>30 group by DATE(tpep_pickup_datetime), hashed_id' 
    cursor = cnx.cursor()
    cursor.execute(test_query)
    result_set = cursor.fetchall()
      
    counts = []
    predicate_list = []
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

def query_taxi_tip_amount(start_time, end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    crnt_query = 'select floor(PULocationID/8)  as hashed_id, avg(tip_amount) as c  from taxi_dataset_march_20 where DATE(tpep_pickup_datetime) >= "' + start_time + '" and DATE(tpep_pickup_datetime) < "' + end_time + '" group by DATE(tpep_pickup_datetime), hashed_id'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()

    counts = []
    predicate_list = []
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

def query_congestion_amount(start_time, end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    crnt_query = 'select floor(PULocationID/8)  as hashed_id, avg(congestion_surcharge) as c  from taxi_dataset_march_20 where DATE(tpep_pickup_datetime) >= "' + start_time + '" and DATE(tpep_pickup_datetime) < "' + end_time + '"  group by DATE(tpep_pickup_datetime), hashed_id'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()
   
    counts = []
    predicate_list = []
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

def query_taxi_tolls_amount(start_time, end_time):
    cnx = mysql.connector.connect(user=creds.USERNAME, password=creds.PASSWORD, host=creds.HOST,
                                port=creds.PORT, database=creds.DB)
    crnt_query = 'select floor(PULocationID/8)  as hashed_id, avg(tolls_amount) as c  from taxi_dataset_march_20 where DATE(tpep_pickup_datetime) >= "' + start_time + '" and DATE(tpep_pickup_datetime) < "' + end_time + '"  group by DATE(tpep_pickup_datetime), hashed_id'
    cursor = cnx.cursor()
    cursor.execute(crnt_query)
    result_set = cursor.fetchall()
    
    counts = []
    predicate_list = []
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