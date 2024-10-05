import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import math

from scipy import stats

from os import walk
from scipy import signal
from scipy.io import wavfile
from matplotlib import cm

def moon():
    cat_directory = './space_apps_2024_seismic_detection/data/lunar/training/catalogs/'
    cat_file = cat_directory + 'apollo12_catalog_GradeA_final.csv'
    cat = pd.read_csv(cat_file)
    print(cat)
    for i in range(20):
        row = cat.iloc[i]
        arrival_time_rel = row['time_rel(sec)']
        print(arrival_time_rel)
        test_filename = row.filename
        print(test_filename)
        data_directory = './space_apps_2024_seismic_detection/data/lunar/training/data/S12_GradeA/'
        csv_file = f'{data_directory}{test_filename}.csv'
        data_cat = pd.read_csv(csv_file)
        print(data_cat)

        mseed_file = f'{data_directory}{test_filename}.mseed'
        st = read(mseed_file)
        print(st)
        print(st[0].stats)
        tr = st.traces[0].copy()

        st_nonfilt = st.copy()
        tr_nonfilt = st_nonfilt.traces[0].copy()
        tr_times_nonfilt = tr_nonfilt.times()
        tr_data_nonfilt = tr_nonfilt.data
        fn, tn, sxxn = signal.spectrogram(tr_data_nonfilt, tr_nonfilt.stats.sampling_rate)

        minfreq = 0.5
        maxfreq = 1.0
        st_filt = st.copy()
        st_filt.filter('bandpass',freqmin=minfreq,freqmax=maxfreq)
        tr_filt = st_filt.traces[0].copy()
        tr_times_filt = tr_filt.times()
        tr_data_filt = tr_filt.data
        
        tr_data_filt_abs = np.abs(tr_data_filt)
        tr_data_filt_replaced = [0.000000001 if value < 0.000000001 else value for value in tr_data_filt_abs]
        f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)

        k, b, r, p, std_err = stats.linregress(tr_times_filt, tr_data_filt_replaced)
        def myfunc(tr_data_filt_replaced):
            return k * tr_data_filt_replaced + b
        
        mymodel = list(map(myfunc, tr_data_filt_replaced))
        regMymodel = [2 * value for value in mymodel]
        negMymodel = [-1 * value for value in mymodel]
        regNegMymodel = [-2 * value for value in mymodel]

        starttime = tr.stats.starttime.datetime
        arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'],'%Y-%m-%dT%H:%M:%S.%f')
        arrival = (arrival_time - starttime).total_seconds()

        # Plot the time series and spectrogram
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(2, 2, 1)
        # Plot trace
        ax.plot(tr_times_filt,tr_data_filt)
        ax.plot(tr_times_filt, mymodel, color = 'g', linestyle = 'solid', label = "Data")
        ax.plot(tr_times_filt, negMymodel, color = 'g', linestyle = 'solid', label = "Data")
        
        ax.plot(tr_times_filt, regMymodel, color = 'g', linestyle = 'dashed', label = "Data")
        ax.plot(tr_times_filt, regNegMymodel, color = 'g', linestyle = 'dashed', label = "Data")
        # Mark detection
        ax.axvline(x = arrival, color='red',label='Detection')
        ax.legend(loc='upper left')
        # Make the plot pretty
        ax.set_xlim([min(tr_times_filt),max(tr_times_filt)])
        ax.set_ylabel('Velocity (m/s)')
        ax.set_xlabel('Time (s)')

        ax2 = plt.subplot(2, 2, 2)
        # Plot trace
        ax2.plot(tr_times_nonfilt,tr_data_nonfilt)
        # Mark detection
        ax2.axvline(x = arrival, color='red',label='Detection')
        ax2.legend(loc='upper left')
        # Make the plot pretty
        ax2.set_xlim([min(tr_times_nonfilt),max(tr_times_nonfilt)])
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_xlabel('Time (s)')

        ax3 = plt.subplot(2, 2, 3)
        vals = ax3.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
        ax3.set_xlim([min(tr_times_filt),max(tr_times_filt)])
        ax3.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
        ax3.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax3.axvline(x=arrival, c='red')
        cbar = plt.colorbar(vals, orientation='horizontal')
        cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

        ax4 = plt.subplot(2, 2, 4)
        vals = ax4.pcolormesh(tn, fn, sxxn, cmap=cm.jet, vmax=5e-17)
        ax4.set_xlim([min(tr_times_nonfilt),max(tr_times_nonfilt)])
        ax4.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
        ax4.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax4.axvline(x=arrival, c='red')
        cbar = plt.colorbar(vals, orientation='horizontal')
        cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

        plt.savefig(f'./plots/moon/{test_filename}.png')

def moon2(grade):
    filenames = next(walk(f'./space_apps_2024_seismic_detection/data/lunar/test/data/{grade}/'), (None, None, []))[2]
    for i in range(int(len(filenames)/2)):
        test_filename = filenames[i*2]
        print(test_filename)
        data_directory = f'./space_apps_2024_seismic_detection/data/lunar/test/data/{grade}/'
        csv_file = f'{data_directory}/{test_filename}'
        data_cat = pd.read_csv(csv_file)
        print(data_cat)

        mseed_file = f'{data_directory}{test_filename[0:len(test_filename)-4]}.mseed'
        st = read(mseed_file)
        print(st)
        print(st[0].stats)
        tr = st.traces[0].copy()

        st_nonfilt = st.copy()
        tr_nonfilt = st_nonfilt.traces[0].copy()
        tr_times_nonfilt = tr_nonfilt.times()
        tr_data_nonfilt = tr_nonfilt.data
        fn, tn, sxxn = signal.spectrogram(tr_data_nonfilt, tr_nonfilt.stats.sampling_rate)

        minfreq = 0.5
        maxfreq = 1.0
        st_filt = st.copy()
        st_filt.filter('bandpass',freqmin=minfreq,freqmax=maxfreq)
        tr_filt = st_filt.traces[0].copy()
        tr_times_filt = tr_filt.times()
        tr_data_filt = tr_filt.data
        tr_data_filt_abs = np.abs(tr_data_filt)
        tr_data_filt_replaced = [0.000000001 if value < 0.000000001 else value for value in tr_data_filt_abs]
        f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)

        k, b, r, p, std_err = stats.linregress(tr_times_filt, tr_data_filt_replaced)
        def myfunc(tr_data_filt_replaced):
            return k * tr_data_filt_replaced + b
        
        mymodel = list(map(myfunc, tr_data_filt_replaced))
        regMymodel = [2 * value for value in mymodel]
        negMymodel = [-1 * value for value in mymodel]
        regNegMymodel = [-2 * value for value in mymodel]

        starttime = tr.stats.starttime.datetime

        # Plot the time series and spectrogram
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(2, 2, 1)
        # Plot trace
        ax.plot(tr_times_filt, tr_data_filt)
        ax.plot(tr_times_filt, mymodel, color = 'g', linestyle = 'solid', label = "Data")
        ax.plot(tr_times_filt, negMymodel, color = 'g', linestyle = 'solid', label = "Data")
        
        ax.plot(tr_times_filt, regMymodel, color = 'g', linestyle = 'dashed', label = "Data")
        ax.plot(tr_times_filt, regNegMymodel, color = 'g', linestyle = 'dashed', label = "Data")

        i = 0
        findedpick = False
        starttimeofquake = 0
        while i <= len(tr_data_filt_abs)-2:
            if tr_data_filt_abs[i] > regMymodel[i] or findedpick:
                findedpick = True
                if starttimeofquake == 0:
                    starttimeofquake = i
                timedelta = 0
                for j in range(i, len(tr_times_filt)):
                    if tr_times_filt[i] + 100 <= tr_times_filt[j]:
                        timedelta = j
                        break
                if timedelta == 0:
                    timedelta = len(tr_times_filt)
                if max(tr_data_filt_abs[i:timedelta]) < mymodel[i]:
                    plt.scatter(tr_times_filt[starttimeofquake], max(tr_data_filt_abs[starttimeofquake:timedelta]), color = 'r', linestyle = 'dashed', marker = '.',label = "Data")
                    plt.scatter(tr_times_filt[i], max(tr_data_filt_abs[i:timedelta]), color = 'r', linestyle = 'dashed', marker = '.',label = "Data")
                    ax.plot([tr_times_filt[starttimeofquake], tr_times_filt[i]], [max(tr_data_filt_abs[starttimeofquake:timedelta]), max(tr_data_filt_abs[i:timedelta])], color = 'r', linestyle = 'solid', label = "Data")
                    a = (tr_times_filt[i]-tr_times_filt[starttimeofquake])/100
                    b = (max(tr_data_filt_abs[starttimeofquake:timedelta])-max(tr_data_filt_abs[i:timedelta]))*10000000000
                    print(a, b)
                    
                    findedpick = False
                    starttimeofquake = 0
                i = timedelta
            i+=1
        
        # Mark detection
        # Make the plot pretty
        ax.set_xlim([min(tr_times_filt),max(tr_times_filt)])
        ax.set_ylabel('Velocity (m/s)')
        ax.set_xlabel('Time (s)')

        ax2 = plt.subplot(2, 2, 2)
        # Plot trace
        ax2.plot(tr_times_nonfilt,tr_data_nonfilt)
        # Mark detection
        # Make the plot pretty
        ax2.set_xlim([min(tr_times_nonfilt),max(tr_times_nonfilt)])
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_xlabel('Time (s)')

        ax3 = plt.subplot(2, 2, 3)
        vals = ax3.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
        ax3.set_xlim([min(tr_times_filt),max(tr_times_filt)])
        ax3.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
        ax3.set_ylabel('Frequency (Hz)', fontweight='bold')
        cbar = plt.colorbar(vals, orientation='horizontal')
        cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

        ax4 = plt.subplot(2, 2, 4)
        vals = ax4.pcolormesh(tn, fn, sxxn, cmap=cm.jet, vmax=5e-17)
        ax4.set_xlim([min(tr_times_nonfilt),max(tr_times_nonfilt)])
        ax4.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
        ax4.set_ylabel('Frequency (Hz)', fontweight='bold')
        cbar = plt.colorbar(vals, orientation='horizontal')
        cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

        plt.savefig(f'./plots/moon/{grade}/{test_filename}.png')

def mars():
    cat_directory = './space_apps_2024_seismic_detection/data/mars/training/catalogs/'
    cat_file = cat_directory + 'Mars_InSight_training_catalog_final.csv'
    cat = pd.read_csv(cat_file)
    print(cat)
    for i in range(20):
        row = cat.iloc[i]
        arrival_time_rel = row['time_rel(sec)']
        print(arrival_time_rel)
        test_filename = row.filename
        print(test_filename)
        data_directory = './space_apps_2024_seismic_detection/data/mars/training/data/'
        csv_file = f'{data_directory}{test_filename}'
        data_cat = pd.read_csv(csv_file)
        print(data_cat)

        mseed_file = f'{data_directory}{test_filename[0:len(test_filename)-4]}.mseed'
        st = read(mseed_file)
        print(st)
        print(st[0].stats)
        tr = st.traces[0].copy()

        st_nonfilt = st.copy()
        tr_nonfilt = st_nonfilt.traces[0].copy()
        tr_times_nonfilt = tr_nonfilt.times()
        tr_data_nonfilt = tr_nonfilt.data
        fn, tn, sxxn = signal.spectrogram(tr_data_nonfilt, tr_nonfilt.stats.sampling_rate)

        minfreq = 0.5
        maxfreq = 1.0
        st_filt = st.copy()
        st_filt.filter('bandpass',freqmin=minfreq,freqmax=maxfreq)
        tr_filt = st_filt.traces[0].copy()
        tr_times_filt = tr_filt.times()
        tr_data_filt = tr_filt.data
        f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)

        starttime = tr.stats.starttime.datetime
        arrival_time = datetime.strptime(row['time_abs(%Y-%m-%dT%H:%M:%S.%f)'],'%Y-%m-%dT%H:%M:%S.%f')
        arrival = (arrival_time - starttime).total_seconds()

        # Plot the time series and spectrogram
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(2, 2, 1)
        # Plot trace
        ax.plot(tr_times_filt,tr_data_filt)
        # Mark detection
        ax.axvline(x = arrival, color='red',label='Detection')
        ax.legend(loc='upper left')
        # Make the plot pretty
        ax.set_xlim([min(tr_times_filt),max(tr_times_filt)])
        ax.set_ylabel('Velocity (c/s)')
        ax.set_xlabel('Time (s)')

        ax2 = plt.subplot(2, 2, 2)
        # Plot trace
        ax2.plot(tr_times_nonfilt,tr_data_nonfilt)
        # Mark detection
        ax2.axvline(x = arrival, color='red',label='Detection')
        ax2.legend(loc='upper left')
        # Make the plot pretty
        ax2.set_xlim([min(tr_times_nonfilt),max(tr_times_nonfilt)])
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_xlabel('Time (s)')

        ax3 = plt.subplot(2, 2, 3)
        vals = ax3.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
        ax3.set_xlim([min(tr_times_filt),max(tr_times_filt)])
        ax3.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
        ax3.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax3.axvline(x=arrival, c='red')
        cbar = plt.colorbar(vals, orientation='horizontal')
        cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

        ax4 = plt.subplot(2, 2, 4)
        vals = ax4.pcolormesh(tn, fn, sxxn, cmap=cm.jet, vmax=5e-17)
        ax4.set_xlim([min(tr_times_nonfilt),max(tr_times_nonfilt)])
        ax4.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
        ax4.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax4.axvline(x=arrival, c='red')
        cbar = plt.colorbar(vals, orientation='horizontal')
        cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

        plt.show()

moon2("S15_GradeB")