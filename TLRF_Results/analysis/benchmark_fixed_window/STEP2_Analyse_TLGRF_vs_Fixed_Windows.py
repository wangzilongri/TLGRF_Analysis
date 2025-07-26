#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.api as sm

import concurrent.futures


import numpy as np
import pandas as pd
import ast
import glob
import pickle
import dask
import os
import itertools

import pickle

#from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from statsmodels.regression.rolling import RollingOLS

from tqdm.notebook import tqdm

from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
#from tqdm import tqdm
from collections import Counter
from functools import reduce


import dask
import dask.dataframe as dd
from dask.distributed import Client
from dask.diagnostics import ProgressBar

#client = Client(n_workers=20, memory_limit="10GB", interface='lo')
from concurrent.futures import ThreadPoolExecutor

import dask_ml.cluster as dask_cluster

from pprint import pprint
import os

pd.set_option('display.max_columns', None)

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# ### Load TLGRF Benchmark Dataset

# In[2]:


benchmark_TLGRF_dataset = dd.read_csv("../generate_benchmark_TLGRF_dataset/benchmark_TLGRF_dataset.csv", assume_missing=True).compute()
benchmark_TLGRF_dataset["date"] = pd.to_datetime(benchmark_TLGRF_dataset["date"])
benchmark_TLGRF_dataset = benchmark_TLGRF_dataset[benchmark_TLGRF_dataset["log_rolled_cases"] >= np.log(21.1)]
benchmark_TLGRF_dataset = benchmark_TLGRF_dataset[benchmark_TLGRF_dataset["date"] <= "2022-12-31"]
benchmark_TLGRF_dataset = benchmark_TLGRF_dataset.sort_values(by=["fips","date"])
#display(benchmark_TLGRF_dataset)


# In[3]:


TLGRF_MAE = benchmark_TLGRF_dataset.groupby("date").apply(lambda x: np.nanmean(abs(x["TLGRF_predicted_log_rolled_cases"]- x["shifted_log_rolled_cases"])))
TLGRF_RMSE = benchmark_TLGRF_dataset.groupby("date").apply(lambda x: np.sqrt(np.nanmean( (x["TLGRF_predicted_log_rolled_cases"]- x["shifted_log_rolled_cases"])**2 )))


# In[4]:


TLGRF_MAE


# ### Load Augmented DF Benchmark Dataset

# In[5]:


augmented_df = dd.read_csv("../../data/augmented_us-counties_latest.csv", assume_missing=True).compute()
augmented_df["date"] = pd.to_datetime(augmented_df["date"])
augmented_df["fips"] = augmented_df["fips"].astype(int)
augmented_df["days_from_start"] = augmented_df["days_from_start"].astype(int)
augmented_df["log_rolled_cases"] = np.log(augmented_df["rolled_cases"] + 1.1)
augmented_df = augmented_df.sort_values(by=["fips","date"])
augmented_df["shifted_log_rolled_cases"] = augmented_df.groupby("fips")["log_rolled_cases"].shift(-7)

# Check for gaps
gt_columns = ["fips", "days_from_start", "date", "log_rolled_cases", "shifted_log_rolled_cases"]
augmented_df_gt = augmented_df[gt_columns]
grouped = augmented_df_gt.groupby('fips')

for fips, group in grouped:
    missing_days = group['days_from_start'].diff().gt(1).sum()
    if missing_days > 0:
        print(f"Gap(s) found in 'days_from_start' for fips {fips}: {missing_days} gap(s)")


#df = augmented_df.copy()
window_sizes = list(range(2,15))
fips_list = augmented_df_gt["fips"].unique()


# ### Load Benchmark Data of $wsize \in \{2,3,4,\dotsc,13,14\}$

# In[6]:


directory = "Fixed_Window_dfs"
file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith(".csv")]
def read_csv(file_path):
    return pd.read_csv(file_path)
with tqdm(total=len(file_paths), desc="Processing") as pbar:
    beta_results = Parallel(n_jobs=-1)(delayed(read_csv)(file_path) for file_path in file_paths)


# ### Check for Duplicates

# In[7]:


present_fips_wsize_dict = {}
for fname in file_paths:
    fips = int(float(fname.split("=")[1].split("_")[0]))
    window_size = int(float(fname.split("=")[2].split(".")[0]))
    if fips not in present_fips_wsize_dict.keys():
        present_fips_wsize_dict[fips] = Counter()
    if window_size in present_fips_wsize_dict[fips].keys():
        print("Duplicate detected fips={} window_size={}".format(fips, window_size))
        present_fips_wsize_dict[fips][window_size] += 1
    else:
        present_fips_wsize_dict[fips].update({window_size : 1})


# ### Remove fips with too few entries

# In[8]:


sorted_beta_results = sorted(beta_results, key = lambda x: (x["fips"].unique()[0], x.columns[2] ))

beta_result_dict_wsize_fips = {}
for beta_df in tqdm(sorted_beta_results):
    fips = int(beta_df["fips"].unique()[0])
    window_size = int(beta_df.columns[2].split("=")[1])
    if fips not in beta_result_dict_wsize_fips.keys():
        beta_result_dict_wsize_fips[fips] = {window_size:beta_df}
    else:
        beta_result_dict_wsize_fips[fips][window_size] = beta_df
        

counter = 0
problematic_fips = {}
for fips in tqdm(beta_result_dict_wsize_fips.keys()):
    counter += 1
    expected_shape = list(beta_result_dict_wsize_fips[fips].items())[0][1].shape
    #print("fips={} has expected shape {}".format(fips, expected_shape))
    have_windows = set(beta_result_dict_wsize_fips[fips].keys())
    missing_window_sizes = set(window_sizes) - set(have_windows)
    if len(missing_window_sizes):
        print("fips={} has missing window_sizes of {}".format(fips, missing_window_sizes))
        problematic_fips[fips] = missing_window_sizes
    for window_size in beta_result_dict_wsize_fips[fips].keys():
        current_shape = beta_result_dict_wsize_fips[fips][window_size].shape
        if current_shape != expected_shape:
            print("fips={}, wsize={} has shape {}, differing from expected {}".format(fips, window_size, expected_shape, current_shape))
        
print(len(fips_list) - counter)


# In[9]:


beta_result_dict_wsize_fips[1001]


# ### Merge Results

# In[10]:


beta_result_dict = {window_size:[] for window_size in window_sizes}
for beta_df in tqdm(sorted_beta_results):
    fips = int(beta_df["fips"].unique()[0])
    window_size = int(beta_df.columns[2].split("=")[1])
    if fips not in problematic_fips.keys() and window_size in window_sizes:
        beta_result_dict[window_size].append(beta_df)

concatenated_beta_result_dict = {}
for window_size, beta_df_list in tqdm(beta_result_dict.items()):
    concatenated_beta_result_dict[window_size] = pd.concat(beta_df_list)
#updated_df.to_csv("TLGRF_w_Fixed_Windows.csv", index=False)


# In[11]:


concatenated_beta_result_dict


# In[12]:


beta_df_big = pd.DataFrame()
#for window_size, fips_beta_df in tqdm(concatenated_beta_result_dict.items()):
#    if not beta_df_big.shape[0]:
#        beta_df_big = fips_beta_df.copy()
#        continue
#    beta_df_big = pd.merge(beta_df_big, fips_beta_df, on="fips", how="outer")

beta_df_big = reduce(lambda left, right: pd.merge(left, right, on=['fips','days_from_start'], how='outer'), concatenated_beta_result_dict.values())


updated_df = pd.merge(augmented_df_gt, beta_df_big,on=['fips','days_from_start'], how="outer").sort_values(by=["fips", 'days_from_start'])
filtered_updated_df = updated_df[updated_df["date"] <= "2022-12-31"]
#filtered_updated_df = pd.merge(filtered_updated_df, augmented_df_gt, on=["fips","days_from_start"], how="left")

#display(filtered_updated_df)


# In[13]:


predictor_columns = ["beta_wsize={}".format(window_size) for window_size in window_sizes]


# In[ ]:





# In[14]:


MAE_df = pd.DataFrame()
RMSE_df = pd.DataFrame()
MAE_df["r_TLGRF"] = TLGRF_MAE
RMSE_df["r_TLGRF"] = TLGRF_RMSE
for predictor_column in tqdm(predictor_columns):
    predictor_MAE = filtered_updated_df.groupby("date").apply(lambda x: np.nanmean(abs(x[predictor_column]*7+x["log_rolled_cases"]- x["shifted_log_rolled_cases"])))
    predictor_RMSE = filtered_updated_df.groupby("date").apply(lambda x: np.sqrt(np.nanmean( (x[predictor_column]*7+x["log_rolled_cases"] - x["shifted_log_rolled_cases"])**2 )))
    MAE_df[predictor_column] = predictor_MAE
    RMSE_df[predictor_column] = predictor_RMSE


# In[ ]:





# In[15]:


MAE_df


# In[16]:


RMSE_df


# In[17]:


metrics_comparison_df = pd.DataFrame()

row_names = ["TLRF"] + ["Fixed Window {}".format(window_size) for window_size in window_sizes]

metrics_comparison_df["MAE"] = MAE_df.median()
metrics_comparison_df["RMSE"] = RMSE_df.median()
metrics_comparison_df.index = row_names
metrics_comparison_df = metrics_comparison_df[::-1]
metrics_comparison_df


# In[18]:


latex_table = metrics_comparison_df.to_latex(column_format='c'*len(metrics_comparison_df.columns), float_format='%.3f', escape=False)
print(latex_table)


# In[21]:


fig = plt.figure(figsize=(18,10))

ax1 = fig.add_subplot(2, 1, 1)
# Create the bottom left subplot
ax2 = fig.add_subplot(2, 2, 3)
# Create the bottom right subplot
ax3 = fig.add_subplot(2, 2, 4)

# Adjust the spacing between subplots
#fig.subplots_adjust(hspace=0.3)

wsizes = [2,7,14]
colors_lm = ["red", "blue", "xkcd:dark turquoise", "magenta"]
colors_lm = list(reversed(colors_lm))

linestyles_lm = ["-","dotted","dashed","dotted"]
linestyles_lm = list(reversed(linestyles_lm))


plot_columns =  ["r_TLGRF"] + ["beta_wsize={}".format(window_size) for window_size in [2,7,14]]
plot_columns = reversed(plot_columns)

plot_row_names = ["TLRF"] + ["Fixed Window {}".format(window_size) for window_size in [2,7,14]]
plot_row_names = list(reversed(plot_row_names))

#plot_columns =  ["r_TLGRF"] + ["beta_wsize={}".format(window_size) for window_size in [2]]
ax1_line_handles = []
for i,plot_column in tqdm(enumerate((plot_columns))):
    ax1_line_handles.append(ax1.plot(MAE_df[plot_column], label=plot_row_names[i], color=colors_lm[i], linestyle=linestyles_lm[i]))
    ax2.plot(MAE_df[plot_column], label=plot_row_names[i], color=colors_lm[i], linestyle=linestyles_lm[i])
    ax3.plot(MAE_df[plot_column], label=plot_row_names[i], color=colors_lm[i], linestyle=linestyles_lm[i])
    
handles = [handle[0] for handle in ax1_line_handles]
labels = [handle[0].get_label() for handle in ax1_line_handles]
locator = mdates.DayLocator(interval=15)
formatter = mdates.DateFormatter('%Y-%m-%d')



ax1.set_xlabel("Date")
ax1.set_ylabel("MAE")
ax1.set_title("(A): MAE of Fixed Window Sizes vs TLRF")
ax1.set_xlim(pd.to_datetime("2020-03-15"), pd.to_datetime("2023-01-01"))
ax1.set_ylim(0,0.6)
ax1.legend(handles, labels, loc='upper right')

ax2.set_xlabel("Date")
ax2.set_ylabel("MAE")
ax2.set_title("(B): Initial Lockdown Period")
ax2.set_xlim(pd.to_datetime("2020-04-30"), pd.to_datetime("2020-08-01"))
ax2.set_ylim(0,0.6)
#ax2.xaxis.set_major_locator(locator)
#ax2.xaxis.set_major_formatter(formatter)


ax3.set_xlabel("Date")
ax3.set_ylabel("MAE")
ax3.set_title("(C): Delta Variant")
ax3.set_xlim(pd.to_datetime("2021-07-01"), pd.to_datetime("2021-08-31"))
ax3.set_ylim(0,0.6)
ax3.xaxis.set_major_locator(locator)
ax3.xaxis.set_major_formatter(formatter)


#plt.legend()
#fig.legend(handles, labels, loc='upper ')

# Adjust the layout
plt.tight_layout()
plt.savefig("new_lm_grf_mae_together.png")

plt.show()


# In[ ]:


[handle[0].get_label() for handle in ax1_line_handles]


# In[ ]:





# In[22]:


fig = plt.figure(figsize=(18,10))

ax1 = fig.add_subplot(2, 1, 1)
# Create the bottom left subplot
ax2 = fig.add_subplot(2, 2, 3)
# Create the bottom right subplot
ax3 = fig.add_subplot(2, 2, 4)

# Adjust the spacing between subplots
#fig.subplots_adjust(hspace=0.3)

plot_columns =  ["r_TLGRF"] + ["beta_wsize={}".format(window_size) for window_size in [2,7,14]]
plot_columns = reversed(plot_columns)

plot_row_names = ["TLRF"] + ["Fixed Window {}".format(window_size) for window_size in [2,7,14]]
plot_row_names = list(reversed(plot_row_names))

colors_lm = ["red", "blue", "xkcd:dark turquoise", "magenta"]
colors_lm = list(reversed(colors_lm))

linestyles_lm = ["-","dotted","dashed","dotted"]
linestyles_lm = list(reversed(linestyles_lm))


plot_columns =  ["r_TLGRF"] + ["beta_wsize={}".format(window_size) for window_size in [2,7,14]]
plot_columns = reversed(plot_columns)

plot_row_names = ["TLRF"] + ["Fixed Window {}".format(window_size) for window_size in [2,7,14]]
plot_row_names = list(reversed(plot_row_names))


#plot_columns =  ["r_TLGRF"] + ["beta_wsize={}".format(window_size) for window_size in [2]]
ax1_line_handles = []
for i,plot_column in tqdm(enumerate((plot_columns))):
    ax1_line_handles.append(ax1.plot(RMSE_df[plot_column], label=plot_row_names[i], color=colors_lm[i], linestyle=linestyles_lm[i]))
    ax2.plot(RMSE_df[plot_column], label=plot_row_names[i], color=colors_lm[i], linestyle=linestyles_lm[i])
    ax3.plot(RMSE_df[plot_column], label=plot_row_names[i], color=colors_lm[i], linestyle=linestyles_lm[i])
    
handles = [handle[0] for handle in ax1_line_handles]
labels = [handle[0].get_label() for handle in ax1_line_handles]
locator = mdates.DayLocator(interval=15)
formatter = mdates.DateFormatter('%Y-%m-%d')



ax1.set_xlabel("Date")
ax1.set_ylabel("RMSE")
ax1.set_title("(A): RMSE of Fixed Window Sizes vs TLRF")
ax1.set_xlim(pd.to_datetime("2020-03-15"), pd.to_datetime("2023-01-01"))
ax1.set_ylim(0,1.0)
ax1.legend(handles, labels, loc='upper right')

ax2.set_xlabel("Date")
ax2.set_ylabel("RMSE")
ax2.set_title("(B): Initial Lockdown Period")
ax2.set_xlim(pd.to_datetime("2020-04-30"), pd.to_datetime("2020-08-01"))
ax2.set_ylim(0,1.0)
#ax2.xaxis.set_major_locator(locator)
#ax2.xaxis.set_major_formatter(formatter)


ax3.set_xlabel("Date")
ax3.set_ylabel("RMSE")
ax3.set_title("(C): Delta Variant")
ax3.set_xlim(pd.to_datetime("2021-07-01"), pd.to_datetime("2021-08-31"))
ax3.set_ylim(0,1.0)
ax3.xaxis.set_major_locator(locator)
ax3.xaxis.set_major_formatter(formatter)


#plt.legend()
#fig.legend(handles, labels, loc='upper ')

# Adjust the layout
plt.tight_layout()
plt.savefig("new_lm_grf_rmse_together.png")

plt.show()


# In[ ]:





# In[ ]:





# ### Save the MAE and RMSE Dataframes

# In[ ]:


RMSE_df.to_csv("Fixed_windows_RMSE_df.csv")
MAE_df.to_csv("Fixed_windows_MAE_df.csv")


# In[ ]:


MAE_df[MAE_df.index <= "2021-09-12"].median()


# In[ ]:


RMSE_df[RMSE_df.index <= "2021-09-12"].median()


# In[ ]:


#break


# In[ ]:


filtered_updated_df.to_csv("Fixed_windows_all_beta.csv", index=False)


# In[ ]:




