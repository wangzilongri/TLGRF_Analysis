#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import numpy as np
import pandas as pd
import ast
import glob
import pickle
import dask
import os
import itertools


#from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score


from multiprocessing import Pool, cpu_count

import dask
import dask.dataframe as dd
from dask.distributed import Client
#client = Client(n_workers=20, memory_limit="10GB", interface='lo')
from concurrent.futures import ThreadPoolExecutor

import dask_ml.cluster as dask_cluster

from pprint import pprint
import os


# In[2]:


pd.set_option('display.max_columns', None)


# ### Define Weighing Function

# In[3]:


def mu_helper(j, delta):
    assert isinstance(delta, int) and delta >= 2, "delta must be integer larger than 2"
    
    numerator = np.sum([i - (delta-1)*0.5 for i in range(j, delta)])
    denominator = np.sum([i*(i-(delta-1)*0.5) for i in range(delta)])
    
    weight = numerator/denominator
    
    return weight

def mu(delta):
    return np.asarray([mu_helper(j, delta) for j in range(1, delta)])


# In[4]:


delta_weights = {}
delta_list = [2,3,4,7,14]
for delta in delta_list:
    delta_weights[delta] = mu(delta)
    print("Weights for delta={} are {}".format(delta,delta_weights[delta]))


# ### Reimport Historical TLGRF Estimates

# In[5]:


merged_TLGRF_results = dd.read_csv("../generate_benchmark_TLGRF_dataset/benchmark_TLGRF_dataset.csv", assume_missing=True).compute()
merged_TLGRF_results["date"] = pd.to_datetime(merged_TLGRF_results["date"])
merged_TLGRF_results = merged_TLGRF_results.sort_values(by=["fips","date"])


# In[6]:


# check for gaps
grouped = merged_TLGRF_results.groupby('fips')

for fips, group in grouped:
    missing_days = group['days_from_start'].diff().gt(1).sum()
    if missing_days > 0:
        print(f"Gap(s) found in 'days_from_start' for fips {fips}: {missing_days} gap(s)")


# In[7]:


merged_TLGRF_results.columns


# ### Generate Composite `r_TLGRF(delta)` and subsequent predictions

# In[8]:


cols_to_keep = ["fips", "county", "state", "date", "days_from_start", "r_TLGRF", "log_rolled_cases", "shifted_log_rolled_cases", "TLGRF_predicted_log_rolled_cases"]
kept_merged_TLGRF_results = merged_TLGRF_results[cols_to_keep]
kept_merged_TLGRF_results["date"] = pd.to_datetime(kept_merged_TLGRF_results["date"])
# Generate the past 13 days of r
for i in range(0,14):
    col_name = "r_TLGRF_{}_days_before".format(i)
    kept_merged_TLGRF_results[col_name] = kept_merged_TLGRF_results.groupby("fips")["r_TLGRF"].shift(i)
# Gnerate the composite TLGRF_r(delta) and the prediction 7 days later
for delta in delta_list:
    col_name = "r_TLGRF_delta={}".format(delta)
    kept_merged_TLGRF_results[col_name] = 0
    rs = kept_merged_TLGRF_results[["r_TLGRF_{}_days_before".format(i) for i in range(delta-1)]].values
    kept_merged_TLGRF_results[col_name] = rs @ delta_weights[delta]
    
    kept_merged_TLGRF_results["TLGRF_delta={}_predicted_log_rolled_cases".format(delta)] = kept_merged_TLGRF_results[col_name] * 7 + kept_merged_TLGRF_results["log_rolled_cases"]

#kept_merged_TLGRF_results = kept_merged_TLGRF_results.dropna(subset=["TLGRF_delta={}_predicted_log_rolled_cases".format(delta_list[-1])])
kept_merged_TLGRF_results = kept_merged_TLGRF_results[kept_merged_TLGRF_results["log_rolled_cases"] >= np.log(20 + 1.1)]
kept_merged_TLGRF_results = kept_merged_TLGRF_results[kept_merged_TLGRF_results["date"] <= "2022-12-31"]


# In[9]:


kept_merged_TLGRF_results.head()


# ### Evaluate Daily Mean RMSE and MAE

# In[10]:


TLGRF_delta_performance = pd.DataFrame()
# Define

for delta in delta_list:
    TLGRF_delta_RMSE = kept_merged_TLGRF_results.groupby("date").apply(lambda x: np.sqrt(np.nanmean((x['shifted_log_rolled_cases'] - x['TLGRF_delta={}_predicted_log_rolled_cases'.format(delta)])**2)))
    TLGRF_delta_MAE = kept_merged_TLGRF_results.groupby("date").apply(lambda x: np.nanmean(np.abs(x['shifted_log_rolled_cases'] - x['TLGRF_delta={}_predicted_log_rolled_cases'.format(delta)])))
    
    TLGRF_delta_performance["RMSE_delta={}".format(delta)] = TLGRF_delta_RMSE
    TLGRF_delta_performance["MAE_delta={}".format(delta)] = TLGRF_delta_MAE
TLGRF_delta_performance.index = pd.to_datetime(TLGRF_delta_performance.index)


# In[11]:


TLGRF_delta_performance


# In[12]:


#for delta in delta_list[1:]:
def plot_comparison(delta1=2, delta2=3, metric="RMSE"):
    plt.clf()
    fig, (ax1) = plt.subplots(1, 1, figsize=(15,5))
    ax1.plot(TLGRF_delta_performance["{}_delta={}".format(metric, delta1)], label="TLRF $\delta$={}".format(delta1), color="red")
    ax1.plot(TLGRF_delta_performance["{}_delta={}".format(metric, delta2)], label="TLRF $\delta$={}".format(delta2), linestyle="dashed")
    
    full_metric = "Mean Absolute Error (MAE)"
    if metric == "RMSE":
        "Root Mean Square Error (RMSE)"

    ax1.set(title='{} in One-Week Ahead COVID Case Predictions'.format(full_metric), xlabel='Date', ylabel=metric)
    #ax1.set_xticks(TLGRF_delta_performance.index[TLGRF_delta_performance.index.is_month_start])
    #ax1.set_xticklabels(TLGRF_delta_performance.index[TLGRF_delta_performance.index.is_month_start].strftime('%Y-%m-%d'), rotation=45)
    #ax1.tick_params(axis='x', which='both', bottom=False, top=False)
    ax1.set_xlim(pd.to_datetime("2020-03-15"), pd.to_datetime("2023-01-01"))
    ylim_dict = {"RMSE": 1.0, "MAE": 0.6}
    ax1.set_ylim(0, ylim_dict[metric])
    ax1.legend()
    plt.tight_layout()
    plt.show()
    ax1.figure.savefig("TLGRF_Delta_{}_{{{}x{}}}.png".format(metric,delta1,delta2))
    



# In[13]:


list(itertools.product([3,4,7,14],["RMSE","MAE"]))


# In[14]:


for delta2, metric in itertools.product([3,4,7,14],["RMSE","MAE"]):
    plot_comparison(delta1=2, delta2=delta2, metric=metric)


# ### Plot $\delta=2$ against others

# In[15]:


fig, axes = plt.subplots(len(delta_list[1:]), 1, figsize=(15,20))
for i, delta in enumerate(delta_list[1:]):
    axes[i].plot(TLGRF_delta_performance["RMSE_delta={}".format(2)], label="TLRF $\delta$={}".format(2), color="red")
    axes[i].plot(TLGRF_delta_performance["RMSE_delta={}".format(delta)], label="TLRF $\delta$={}".format(delta), linestyle="dashed")    
    axes[i].set(title='Root Mean Square Error (RMSE) in One-Week Ahead COVID Case Predictions', xlabel='Date', ylabel="RMSE")
    #axes[i].set_xticks(TLGRF_delta_performance.index[TLGRF_delta_performance.index.is_month_start])
    #axes[i].set_xticklabels(TLGRF_delta_performance.index[TLGRF_delta_performance.index.is_month_start].strftime('%Y-%m-%d'), rotation=45)
    #axes[i].tick_params(axis='x', which='both', bottom=False, top=False)
    axes[i].set_xlim(pd.to_datetime("2020-03-15"), pd.to_datetime("2023-01-01"))
    axes[i].set_ylim(0, 1.0)
    axes[i].legend()
    #axes[i].figure.savefig("TLGRF_Delta_RMSE_{{{}x{}}}.png".format(2,delta))
plt.tight_layout()
plt.savefig("TLGRF_Delta_RMSE.png")
plt.show()



# In[16]:


fig, axes = plt.subplots(len(delta_list[1:]), 1, figsize=(15,20))
for i, delta in enumerate(delta_list[1:]):
    axes[i].plot(TLGRF_delta_performance["MAE_delta={}".format(2)], label="TLRF $\delta$={}".format(2), color="red")
    axes[i].plot(TLGRF_delta_performance["MAE_delta={}".format(delta)], label="TLRF $\delta$={}".format(delta), linestyle="dashed")
    axes[i].set(title='Mean Absolute Error (MAE) in One-Week Ahead COVID Case Predictions', xlabel='Date', ylabel='MAE')
    axes[i].set_xlim(pd.to_datetime("2020-03-15"), pd.to_datetime("2023-01-01"))
    axes[i].set_ylim(0, 0.6)
    axes[i].legend()
plt.tight_layout()
plt.savefig("TLGRF_Delta_MAE.png")
plt.show()


# In[ ]:





# In[ ]:





# ### Table of Median of Daily RMSE and MAE for each Delta

# In[17]:


TLGRF_delta_performance_pivot = TLGRF_delta_performance.copy()
TLGRF_delta_performance_dict = {}

TLGRF_delta_performance_dict["$\delta$"] = delta_list
TLGRF_delta_performance_dict["Median MAE"] = [TLGRF_delta_performance_pivot["MAE_delta={}".format(delta)].median() for delta in delta_list]
TLGRF_delta_performance_dict["Median RMSE"] = [TLGRF_delta_performance_pivot["RMSE_delta={}".format(delta)].median() for delta in delta_list]
TLGRF_delta_performance_dict

TLGRF_delta_performance_pivot = pd.DataFrame(TLGRF_delta_performance_dict)
TLGRF_delta_performance_pivot = TLGRF_delta_performance_pivot.sort_values(by="$\delta$", ascending=False)
TLGRF_delta_performance_pivot
#TLGRF_pivot_table = TLGRF_delta_performance.pivot_table(index=TLGRF_delta_performance.index, columns='delta', aggfunc='median')
#TLGRF_pivot_table.index = ['MAE', 'RMSE']
#TLGRF_pivot_table
print(TLGRF_delta_performance_pivot.to_string(index=False))


# In[18]:


#display(TLGRF_delta_performance_pivot)


# In[19]:


latex_table = TLGRF_delta_performance_pivot.to_latex(index=False, column_format='c|c|c', escape=False, header=['$\delta$', 'Median MAE', 'Median RMSE'], float_format='%.3f')
centered_table = '\\begin{center}\n' + latex_table + '\\end{center}'


# In[20]:


print(centered_table)


# In[21]:


TLGRF_delta_performance.to_csv("TLGRF_delta_performance.csv")


# In[ ]:




