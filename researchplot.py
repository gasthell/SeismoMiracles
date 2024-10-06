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

class Moon:
    def __init__(self, grade):
        self.grade = grade
        self.directory = f'./space_apps_2024_seismic_detection/data/lunar/test/data/{self.grade}/'
        self.mymodel, self.regMymodel, self.negMymodel, self.regNegMymodel = [[],[],[],[]]
    
    def main(self):
        filenames = next(walk(f'./space_apps_2024_seismic_detection/data/lunar/test/data/{self.grade}/'), (None, None, []))[2]
        for i in range(int(len(filenames)/2)):
            self.predict(filenames[i*2])

    def isExp(self, tr_data):
        tr_av = 0
        counter = 0
        for j in range(0, len(tr_data)-101):
            counter+=1
            tr_av = max(tr_data[j:j+100])

        return tr_av/counter

    def predict(self, filename):
        self.TraceVerification(filename)

    def spectogramShakeDetection(self, f, t, sxx):
        minfIndex = 0
        for i in range(len(f)):
            if f[i] >= 0.5:
                minfIndex = i
                break
        maxfIndex = 0
        for i in range(len(f)):
            if f[i] >= 1:
                maxfIndex = i
                break
        search = False
        minY = 1
        maxY = 0
        detectX = 100000
        detectEntryX = []
        for i in range(len(t)-1):
            for j in range(minfIndex, maxfIndex):
                if sxx[j][i] > 1.5e-18:
                    if maxY-minY >= 0.47:
                        if t[detectX] not in detectEntryX:
                            if len(detectEntryX) >= 1:
                                if detectEntryX[len(detectEntryX)-1] + 2000 <= t[detectX]:
                                    detectEntryX.append(t[detectX])
                            else:
                                detectEntryX.append(t[detectX])
                    minY = min(minY, j)
                    maxY = max(maxY, j)
                    if not search:
                        search = True
                        detectX = i
                if detectX + 50 <= i:
                    detectX = 100000
                    search = False
                    minY = 1
                    maxY = 0
                    z = 0
        return detectEntryX
            
    def TraceVerification(self, filename):
        mseed_file = f'{self.directory}{filename[0:len(filename)-4]}.mseed'
        st = read(mseed_file)

        # Non filtered data
        st_nonfilt = st.copy()
        tr_nonfilt = st_nonfilt.traces[0].copy()
        tr_times_nonfilt = tr_nonfilt.times()
        tr_data_nonfilt = tr_nonfilt.data
        fn, tn, sxxn = signal.spectrogram(tr_data_nonfilt, tr_nonfilt.stats.sampling_rate)

        # Filtered data
        tr_times_filt, tr_data_filt, tr_data_filt_abs, tr_data_filt_replaced, f, t, sxx = self.filterData(st)

        #Maximized background and reg line
        self.maximizedBackgroundNRegLine(tr_times_filt, tr_data_filt_replaced)

        # Initialize figure
        fig = plt.figure(figsize=(20, 10))
        ax = plt.subplot(2, 2, 1)
        self.plotTrace(ax, tr_times_filt, tr_data_filt)
        self.plotMaximizedBackgroundNRegLine(ax, tr_times_filt)

        SeismoEntry = []

        EntryX = self.spectogramShakeDetection(f, t, sxx)

        for elem in EntryX:
            ax.axvline(x = elem, color='pink', linestyle = 'dashed', label='Detection', linewidth=2)
            startX = 0
            for i in range(len(tr_times_filt)):
                if tr_times_filt[i] >= elem:
                    startX = i
                    break
            timedelta = 1000
            for j in range(startX, len(tr_times_filt)):
                if tr_times_filt[startX] + 200 <= tr_times_filt[j]:
                    timedelta = j
            
            if max(tr_data_filt_abs[startX:timedelta]) > self.regMymodel[startX]:
                i = 0
                findedpick = False
                starttimeofquake = 0

                while startX <= len(tr_data_filt_abs)-1:
                    if tr_data_filt_abs[startX] > self.regMymodel[startX] or findedpick:
                        findedpick = True
                        if starttimeofquake == 0:
                            starttimeofquake = startX
                        timedelta = 0
                        
                        for j in range(startX, len(tr_times_filt)):
                            if tr_times_filt[startX] + 200 <= tr_times_filt[j]:
                                timedelta = j
                                break
                        if timedelta == 0:
                            timedelta = len(tr_times_filt)

                        BottomY = max(tr_data_filt_abs[startX:timedelta])
                        if BottomY < self.mymodel[startX]:
                            BottomX = tr_times_filt[startX]

                            TopY = max(tr_data_filt_abs[starttimeofquake:timedelta])
                            TopX = tr_times_filt[starttimeofquake]
                            plt.scatter(TopX, TopY, color = 'pink', linestyle = 'dashed', marker = '.')
                            plt.scatter(BottomX, BottomY, color = 'pink', linestyle = 'dashed', marker = '.')
                            ax.plot([TopX, BottomX], [TopY, BottomY], color = 'pink', linestyle = 'solid')
                            
                            a = (BottomX-TopX)/100
                            b = (TopY-BottomY)*10000000000
                            if(a > 6 and a < 80):
                                SeismoEntry.append(elem)
                                ax.axvline(x = elem, color='red',label='Detection')
                            
                            findedpick = False
                            starttimeofquake = 0
                            break
                        startX = timedelta
                    startX+=1
        print(SeismoEntry)
        '''
        i = 0
        findedpick = False
        starttimeofquake = 0

        while i <= len(tr_data_filt_abs)-1:
            if tr_data_filt_abs[i] > self.regMymodel[i] or findedpick:
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

                BottomY = max(tr_data_filt_abs[i:timedelta])
                if BottomY < self.mymodel[i]:
                    BottomX = tr_times_filt[i]

                    TopY = max(tr_data_filt_abs[starttimeofquake:timedelta])
                    TopX = tr_times_filt[starttimeofquake]
                    plt.scatter(TopX, TopY, color = 'pink', linestyle = 'dashed', marker = '.')
                    plt.scatter(BottomX, BottomY, color = 'pink', linestyle = 'dashed', marker = '.')
                    ax.plot([TopX, BottomX], [TopY, BottomY], color = 'pink', linestyle = 'solid')
                    
                    a = (BottomX-TopX)/100
                    b = (TopY-BottomY)*10000000000
                    if(a > 8):
                        SeismoEntry.append(TopX)
                        ax.axvline(x = TopX, color='red',label='Detection')
                    
                    findedpick = False
                    starttimeofquake = 0
                i = timedelta
            i+=1
        '''

        # Initialize figure
        ax2 = plt.subplot(2, 2, 2)
        self.plotTrace(ax2, tr_times_nonfilt, tr_data_nonfilt)

        # Initialize figure
        ax3 = plt.subplot(2, 2, 3)
        self.plotSpectogram(ax3, tr_times_filt, f, t, sxx)
        for elem in EntryX:
            ax3.axvline(x = elem, color='pink', linestyle = 'dashed', label='Detection', linewidth=2)
        for elem in SeismoEntry:
            ax3.axvline(x = elem, color='red',label='Detection')

        # Initialize figure
        ax4 = plt.subplot(2, 2, 4)
        self.plotSpectogram(ax4, tr_times_filt, fn, tn, sxxn)

        plt.savefig(f'./plots/moon/{self.grade}/{filename}.png')
        plt.close()

    def maximizedBackgroundNRegLine(self, tr_times_filt, tr_data_filt_replaced):
        k, b, r, p, std_err = stats.linregress(tr_times_filt, tr_data_filt_replaced)
        def myfunc(tr_data_filt_replaced):
            return k * tr_data_filt_replaced + b
        
        maxCof = max(tr_data_filt_replaced)
        self.mymodel = list(map(myfunc, tr_data_filt_replaced))
        
        kdelta = maxCof/self.mymodel[int(len(tr_data_filt_replaced)/2)] * 0.6

        self.regMymodel = [kdelta * value for value in self.mymodel]
        self.negMymodel = [-1 * value for value in self.mymodel]
        self.regNegMymodel = [-kdelta * value for value in self.mymodel]
    
    def filterData(self, st):
        minfreq = 0.5
        maxfreq = 1.0
        st_filt = st.copy()
        st_filt.filter('bandpass',freqmin=minfreq,freqmax=maxfreq)
        tr_filt = st_filt.traces[0].copy()
        tr_times_filt = tr_filt.times()
        tr_data_filt = tr_filt.data
        tr_data_filt_abs = np.abs(tr_data_filt)
        tr_data_filt_replaced = [0.0000000008 if value < 0.0000000008 else value for value in tr_data_filt_abs]
        f, t, sxx = signal.spectrogram(tr_data_filt, tr_filt.stats.sampling_rate)
        
        return tr_times_filt, tr_data_filt, tr_data_filt_abs, tr_data_filt_replaced, f, t, sxx
    
    def plotTrace(self, ax, tr_times, tr_data):
        ax.plot(tr_times, tr_data)
        ax.set_xlim([min(tr_times),max(tr_times)])
        ax.set_ylabel('Velocity (m/s)')
        ax.set_xlabel('Time (s)')

    def plotMaximizedBackgroundNRegLine(self, ax, tr_times):
        ax.plot(tr_times, self.mymodel, color = 'g', linestyle = 'solid')
        ax.plot(tr_times, self.negMymodel, color = 'g', linestyle = 'solid')
        ax.plot(tr_times, self.regMymodel, color = 'g', linestyle = 'dashed')
        ax.plot(tr_times, self.regNegMymodel, color = 'g', linestyle = 'dashed')

    def plotSpectogram(self, ax, tr_times, f, t, sxx):
        vals = ax.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
        ax.set_xlim([min(tr_times),max(tr_times)])
        ax.set_xlabel(f'Time (Day Hour:Minute)', fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontweight='bold')
        cbar = plt.colorbar(vals, orientation='horizontal')
        cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')


moon = Moon("S16_GradeB")
moon.main()