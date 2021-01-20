from operator import truediv
import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import argparse
import os, sys
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter,filtfilt,convolve, freqz, find_peaks,firwin,lfilter,argrelextrema
from scipy.fft import fft, ifft,fftfreq
import math
from tqdm import tqdm 
import windows as win
  


samplerate = 500
grav = 9.6        #gravity
num_box = 20

def sensAspec(dici,reg,data_points):
    if 0 not in  dici:
        dici[0] = 0
    if 1 not in  dici:
        dici[1] = 0
    for i in range(2,20):
        if i in dici:
            dici[1] += dici[i]

    fn = dici[0]
    tn = data_points-(reg)
    tp = int(dici[1])
    fp = (dici['pred']) - (tp) 
    if fp < 0:
        fp = 0
    print('fn:',fn,'tn:',tn,'tp:',tp,'fp:',fp)
    sens =  tp/(tp + fn)
    spec = 1-(fp/(fp+tn))
    print('Sensitivity: ',sens,'Specificity: ',spec)
    return sens, spec

def my_stats(arr,att):
    unique, counts = np.unique(arr, return_counts=True)
    stats = dict(zip(unique, counts))
    stats['pred'] = len(att)
    print(stats)
    return stats

def stepInBin(beg,end,steps):
    #print(beg,end)
    #beg=np.multiply(beg,samplerate)
    #end=np.multiply(end,samplerate)
    bins = [0] * len(beg)
    for i in range(len(beg)):
        for j in steps:
            #print(beg[i],j,end[i])
            if j <= end[i] and j >= beg[i]:
                bins[i] += 1
    #print(bins)            
    return bins

def makeRange(fName):
    fName = fName[0:4] +'-Annotations'
    mydir = os.getcwd()
    fol = "Archive/Data/"
    # Join various path components  
    fName = os.path.join(mydir,fol,fName) 
    print(fName)
    if fName[-4:] != ".csv":
        fName = fName + '.csv'
    if os.path.isfile(fName) == False:
        print("no file with that name!")
        return
    truePoints=np.genfromtxt(fName,delimiter=',')
    #print(truePoints)
    start  = np.array([])
    end  = np.array([])
    for i in truePoints:
        #check within a second
        start = np.append(start,(i-250))
        end = np.append(end,(i+250))
    return start,end

def midpoint(p1, p2):
    return p1 + (p2-p1)/2

def midpoints(start,end):
    points = np.array([])
    for i in range(start):
        points = np.append(points,midpoint(start[i],end[i]))
    return points
    

def checkCough(myMax,thresh):
    if myMax > thresh:
        return True
    return False

def coughDet(df):
    try:
        Y = fft(df.to_numpy())
    except:
        Y = fft(df)
    freqs = fftfreq(len(Y)) * samplerate
    loc = np.argmax(np.abs(Y))
    mymax = np.amax(np.abs(Y))
    #print(mymax)
    return mymax,loc

def specdft(df):
    fig, ax = plt.subplots()
    fig.set_size_inches(20.0,10.0)
    try:
        Y = fft(df.to_numpy())
    except:
        Y = fft(df)
    freqs = fftfreq(len(Y)) * samplerate
    ax.stem(np.abs(freqs), np.abs(Y),label='norm')
    ax.set_xlabel('Frequency in Hertz [Hz]')
    ax.set_ylabel('Magnitude')
    ax.legend()
    plt.show()

def logScalespecdft(df):
    fig, ax = plt.subplots()
    fig.set_size_inches(20.0,10.0)
    try:
        Y = fft(df.to_numpy())
    except:
        Y = fft(df)
    freqs = fftfreq(len(Y)) * samplerate
    ax.semilogy(np.abs(freqs), np.abs(Y),label='norm')
    ax.set_xlabel('Frequency in Hertz [Hz]')
    ax.set_ylabel('Magnitude(dB)')
    ax.legend()
    plt.show()

def morePower(df):
    fig, ax = plt.subplots()
    fig.set_size_inches(20.0,10.0)
    f, Pxx_den = signal.welch(df, samplerate,nperseg=100)
    ax.semilogy(f, Pxx_den)
    #ax.set_ylim([0.5e-3, 1])
    ax.axhline(.5e-5)
    countAbov = 0
    countbel = 0
    for i in Pxx_den:
        if i < .5e-5:
            countAbov += 1
        else:
            countbel += 1
    if countbel > countAbov:
        print("\ncough")

    #ax.set_xlabel('frequency [Hz]')
    #ax.set_ylabel('PSD [V**2/Hz]')
    #plt.show()

def printFinal(df,time,start,end):
    fig, ax = plt.subplots()
    fig.set_size_inches(20.0,10.0)
    for i in range(len(start)):
         #plt.axvspan(start[i],end[i],facecolor='orange', alpha=0.5)
         ax.axvline(end[i], label = 'Coughs',color='black')
    ax.plot(time,df['x'],label='x')
    ax.plot(time,df['y'],label='y')
    ax.plot(time,df['z'],label='z')
    ax.plot(time,df['norm'],label='L2 Norm')
    ax.set_title("Sensor")
    ax.set_xlabel('time (s)')
    ax.set_ylabel('acceleration')
    #plt.legend()
    plt.show()


def printOG(df,time):
    fig, ax = plt.subplots()
    fig.set_size_inches(20.0,10.0)
    ax.plot(time,df['x'],label='x')
    ax.plot(time,df['y'],label='y')
    ax.plot(time,df['z'],label='z')
    ax.plot(time,df['norm'],label='L2 Norm')
    ax.set_title("Sensor")
    ax.set_xlabel('time (s)')
    ax.set_ylabel('acceleration')
    plt.legend()
    plt.show()

def l2Norm(df):
    x = df.iloc[:, 0]
    y = df.iloc[:, 1] 
    z = df.iloc[:, 2]
    dat = {'ave':np.sqrt(x**2 + y**2 + z**2)}
    df_ret = pd.DataFrame(dat)
    return df_ret

def getDat(fName):
    col = ['x','y','z']
    df = pd.read_csv(fName, names = col)
    df['norm'] = l2Norm(df)
    time = np.linspace(0, df.shape[0]/samplerate,df.shape[0])
    return time ,df


def predictSamp(df,co,thresh):
    df = df[:5000]
    start = np.array([])
    end= np.array([])
    myrange = (df.shape[0])
    for i in tqdm (range(0,myrange,100), desc="Finding Coughs...  "):
        df2 = df[i:i+100]
        time = (np.linspace(0, df2.shape[0]/samplerate,df2.shape[0]))+i/samplerate
        fig, ax = plt.subplots()
        fig.set_size_inches(20.0,10.0)
        ax.plot(time,df2['x'],label='x')
        ax.plot(time,df2['y'],label='y')
        ax.plot(time,df2['z'],label='z')
        ax.plot(time,df2['norm'],label='L2 Norm')
        ax.set_title("Sensor")
        ax.set_xlabel('time (s)')
        ax.set_ylabel('acceleration')
        #ax.set_xlim([i,(i+1000)])
        plt.legend()
        plt.show()
        filtered = filt(df2['z'],10,co)
        #specdft(df2['z'])
        specdft(filtered)
        #logScalespecdft(filtered)
        #morePower(df2['norm'])
        myMax, loc = coughDet(filtered)
        #specdft(filtered)
        if checkCough(myMax,thresh) == True:
            print('\n' + str((i/samplerate)),str(((i+100)/samplerate)))
            start = np.append(start,(i/samplerate))
            end = np.append(end,((i+100)/samplerate))
            #printOG(df2,time)
    return start,end

def predict(df,co,thresh):
    start = np.array([])
    end= np.array([])
    myrange = (df.shape[0]//1000) * 1000
    for i in tqdm (range(0,myrange,100), desc="Finding Coughs...  "):
        df2 = df[i:i+100]
        #time = np.linspace(0, df2.shape[0]/samplerate,df2.shape[0])
        time = (np.linspace(0, df2.shape[0]/samplerate,df2.shape[0]))+i//samplerate
        #printOG(df2,time)
        filtered = filt(df2['z'],co,100)
        #specdft(filtered)
        myMax, loc = coughDet(filtered)
        #specdft(filtered)
        if checkCough(myMax,thresh) == True:
            #print('\n'+str((i+100)/samplerate))
            start = np.append(start,(i/samplerate))
            end = np.append(end,((i+100)/samplerate))
            #printOG(df2,time.round(decimals = 4))
    return start,end

def filt(sig,numtaps,f):
    sos = signal.butter(numtaps, f, 'hp', fs=samplerate, output='sos')
    filtered = signal.sosfilt(sos, sig)
    return filtered
'''
def filt(data,numtaps,f):
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
    #b = signal.firwin(20, cutoff=1, fs=samplerate, window='boxcar')
    #band_seg = filtfilt(b, 1.0, data)
    b,x = win.h('gauss',num_box)
    band_seg = np.convolve(b,data,'same')
    return band_seg
'''


def main(fName,co,thresh):
    mydir = os.getcwd()
    fol = "Archive/Data/"
    # Join various path components  
    fNames = os.path.join(mydir,fol,fName) 
    print(fNames)
    if fNames[-4:] != ".csv":
        fNames = fNames + '.csv'
    if os.path.isfile(fNames) == False:
        print("no file with that name!")
        return
    time,df = getDat(fNames)
    #printOG(df,time)
    tStart,tEnd = makeRange(fName=fName)
    start,end = predict(df,co,thresh)
    

    printFinal(df,time,start,end)
    print("Number of Coughs:", len(end))
    shift_end = np.multiply(end,samplerate)
    bins = stepInBin(tStart,tEnd,shift_end)
    stats = my_stats(bins,shift_end)
    sens,spec = sensAspec(stats,(tEnd[0]-tStart[0]),len(df['norm']))
    print(shift_end,tStart,tEnd)

    
    
    
    

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fileName', '-f', default='R001-Accelerometer.csv', type=str,
                        help='The file name we want to process. Default: ^DJI.csv')

    parser.add_argument('--coef', '-c', default=15, type=int,
                        help='coeffiecnets for filter')
    parser.add_argument('--thresh', '-t', default=15, type=int,
                        help='Threshold number')
    args = parser.parse_args()

    main(args.fileName,args.coef,args.thresh)
    #15,15
    #1,5