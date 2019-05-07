#!/usr/bin/env python3

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fname_train = "csv/train_1M.csv"
df_train = pd.read_csv(fname_train, dtype={
                       'acoustic_data': np.int16, 'time_to_failure': np.float64})
print("train shape", df_train.shape)
pd.set_option("display.precision", 15)  # show more decimals
print(df_train.head())

#fig, ax1 = plt.subplots(figsize=(16, 8))
#plt.plot(df_train['acoustic_data'], color='b')
#ax2 = ax1.twinx()
#plt.plot(df_train['time_to_failure'], color='r')
# plt.show()

# Sub-samples of 150k rows
# 629M / 150k = 4194


chunk_size = 150000
n_samples = df_train.shape[0] // chunk_size

samples = np.array_split(df_train, n_samples)
for sample in samples:
    print(type(sample))

from scipy.fftpack import fft, ifft

samples_fft = []
for i in range(n_samples):
    x = samples[i]['acoustic_data']
    X = fft(x)
    samples_fft += [ pd.DataFrame(data=abs(X) ) ] # amplitude
print("FFT shape:", samples_fft[0].shape)
n_freq = samples_fft[0].shape[0]//2

n_rows = 2
n_cols = 5
fig, ax = plt.subplots(2, 5, sharex='col', sharey='row', figsize=(15, 7))
for i in range(n_rows):
    for j in range(n_cols):
        k = n_rows*i+j
        ax[i, j].plot(samples_fft[k])
        ax[i, j].set_ylim(0.,3*1e4)
        ax[i, j].set_xlim(0,n_freq)
fig.show()
