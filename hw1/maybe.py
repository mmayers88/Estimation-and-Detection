import argparse
import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from scipy import stats
from scipy.fft import fft, ifft, rfft,fftfreq
from scipy.signal import butter,filtfilt,convolve, freqz, find_peaks,firwin,lfilter,argrelextrema
import scipy.signal as sg
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
import seaborn as sns

#sample rate is 125Hz
samplerate = 32

def find_val(arr,height):
    arr = -arr
    return find_peaks(arr,height=height)

def returnMax(data):
    bigMax = np.amax(data)
    peaks, _ = find_peaks(data, height=bigMax*.1)
    maxx = np.array([data[i] for i in peaks])
    maxx =  maxx.mean()
    return maxx

def find_bounds(data,thresh):
    abov = [i for i,v in enumerate(data) if v > thresh]
    begi = []
    edd = []
    for i in abov:
        if i != 0 and (i-1 not in abov):
            begi.append(i)
        if i != len(data)-1 and (i+1 not in abov) and begi:
            edd.append(i)
    #print(len(begi),len(edd))
    if len(begi) > len(edd):
        begi = begi[:-1]
    return begi,edd

def filt(data, samplerate, ct =  60, nt = 11):
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
    #b = firwin(nt, cutoff=ct, fs=samplerate, window='hamming')
    b = firwin(nt, cutoff=ct, fs=samplerate, window='boxcar')
    filtered = filtfilt(b, 1.0, data)
    return filtered

def turn(fileName,ct,nt,pr,thr,gra,verb,sav):
    if fileName[-4:] != '.csv':
        fileName =  fileName+'.csv'
    if os.path.isfile(fileName):
        data = np.loadtxt(fileName)
    else:
        print("no file with that name!")
        return
    time = range(0,len(data),1)
    time = [x * (1/samplerate) for x in time]
    X = fft(data)
    freqs= fftfreq(len(X)) * samplerate

    filtered = filt(data,samplerate,ct,nt)
    
    if pr > 0:
        for x in range(pr):
            filtered = filt(filtered, samplerate,ct,nt)
    Y = fft(filtered)
    freqs2 = fftfreq(len(Y)) * samplerate
    if gra == 1:
        fig, ax = plt.subplots(2)
        
        ax[0].plot(np.abs(freqs), np.abs(X))
        ax[0].set_xlabel('Frequency in Hertz [Hz]')
        ax[0].set_ylabel('Frequency Domain (Spectrum) Magnitude')
        ax[0].set_title("Unfiltered Signal")
        
        ax[1].plot(np.abs(freqs2), np.abs(Y))
        ax[1].set_xlabel('Frequency in Hertz [Hz]')
        ax[1].set_ylabel('Frequency Domain (Spectrum) Magnitude')
        ax[1].set_title("Filtered Signal")
        plt.show()
    '''
    filtered = data['velocity'].rolling(ct,min_periods=1).mean()
    filtered = np.roll(filtered,-1*ct//2)
    '''
    returnstff = []
    #print(data)
    iBegin,iEnd = find_bounds(filtered,returnMax(filtered)*.4)
    jBegin,jEnd = find_bounds(-filtered,returnMax(filtered)*.4)
    iBegin.extend(jBegin)
    iEnd.extend(jEnd)
    iBegin = np.divide(iBegin,samplerate)
    iEnd = np.divide(iEnd,samplerate)
    plt.figure(figsize=(20,14))
    if verb != 0:
        print(fileName[:-4]+':')
    for i,t in enumerate(iBegin):
        if verb != 0:
            print("Turn Length {}: {}s".format(i+1,round(iEnd[i]-iBegin[i],3)))
        plt.axvspan(iBegin[i],iEnd[i],facecolor='r', alpha=0.5)
        returnstff.append(round(iEnd[i]-iBegin[i],3))
    print(returnstff)
    plt.title(fileName[:-4])
    plt.plot(time,data)
    plt.plot(time, filtered, label="Filtered")
    plt.xlabel('Time[s]')
    plt.ylabel('Velocity[m/s]')
    if sav == 1:
        path = os.path.join(os.getcwd(), "pix", fileName[:-4]+".png")
        plt.savefig(path,dpi=199)
    plt.show()
    
    return returnstff
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fName', '-f', default="Archive\Data\Accelerometers-20200716-10015369_Walk.csv", type=str,
                        help='name of txt file')
    parser.add_argument('--cutoff', '-c', default=.5, type=int,
                        help='Number of days')
    parser.add_argument('--numt', '-n', default=19, type=int,
                        help='numtaps')
    parser.add_argument('--passr', '-p', default=100, type=int,
                        help='how many filter passes')
    parser.add_argument("--thresh", '-t', default= 3, type=float,
                        help='threshold percentage between 1 - 10')
    parser.add_argument("--graphs", '-g', default= 0, type=int,
                        help='extra graphs')
    parser.add_argument("--verbose", '-v', default= 0, type=int,
                        help='Print turn times')
    parser.add_argument("--save", '-s', default= 0, type=int,
                        help='save file')


    args = parser.parse_args()
    turn(args.fName,args.cutoff,args.numt,args.passr,args.thresh,args.graphs,args.verbose,args.save)


    #https://stackoverflow.com/questions/13717463/find-the-indices-of-elements-greater-than-x




    '''
    data = np.subtract(data, grav)
    data = -data
    data = np.add(data, grav)
    step = np.array([])
    on = 0
    for i,t in enumerate(data):
        if on == 1:
            if t < data[i-1]:
                on = 0
        if t > (thresh*.75) and on == 0 and t > data[i-1]:
            #print(i)
            step = np.append(step,i)
            on = 1
        else:
            continue
    #print(step)
    return step.astype(int)
    '''