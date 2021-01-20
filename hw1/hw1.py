import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import argparse
import os, sys
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter,filtfilt,convolve, freqz, find_peaks,firwin,lfilter,argrelextrema
from scipy.fft import fft, ifft,fftfreq
import windows as win
import math

samplerate = 32
grav = 9.6        #gravity
num_box = 20

def specdft(left,right,wrist):
    print(left)
    fig, ax = plt.subplots(3)
    fig.set_size_inches(20.0,10.0)
    Y = fft(left['ave'].to_numpy())
    freqs = fftfreq(len(Y)) * samplerate
    ax[0].plot(np.abs(freqs), np.abs(Y),label='left')
    ax[0].set_xlabel('Frequency in Hertz [Hz]')
    ax[0].set_ylabel('Magnitude')
    Y = fft(right['ave'].to_numpy())
    freqs = fftfreq(len(Y)) * samplerate
    ax[1].plot(np.abs(freqs), np.abs(Y),label='right')
    ax[1].set_xlabel('Frequency in Hertz [Hz]')
    ax[1].set_ylabel('Magnitude')
    Y = fft(wrist['ave'].to_numpy())
    freqs = fftfreq(len(Y)) * samplerate
    ax[2].plot(np.abs(freqs), np.abs(Y),label='wrist')
    ax[2].set_xlabel('Frequency in Hertz [Hz]')
    ax[2].set_ylabel('Magnitude')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    plt.show()

def sensAspec(dicil,dicir,regl,regr,data_points):
    if 0 not in  dicil:
        dicil[0] = 0
    if 0 not in  dicir:
        dicir[0] = 0
    if 1 not in  dicil:
        dicil[1] = 0
    if 1 not in  dicir:
        dicir[1] = 0

    fn = dicil[0] + dicir[0]
    tn = data_points-(regl+regr)
    tp = dicil[1] + dicir[1]
    fp = (dicil['pred'] + dicir['pred']) - (tp) 
    if fp < 0:
        fp = 0
    print('fn:',fn,'tn:',tn,'tp:',tp,'fp:',fp)
    sens =  tp/(tp + fn)
    spec = 1-(fp/(fp+tn))
    print('sens',sens,'spec',spec)
    return sens, spec

def my_stats(arr,att):
    unique, counts = np.unique(arr, return_counts=True)
    stats = dict(zip(unique, counts))
    stats['pred'] = len(att)
    print(stats)
    return stats

def stepInBin(beg,end,steps):
    #print(beg,end)
    beg=np.multiply(beg,samplerate)
    end=np.multiply(end,samplerate)
    bins = [0] * len(beg)
    for i in range(len(beg)):
        for j in steps:
            #print(beg[i],j,end[i])
            if j <= end[i] and j >= beg[i]:
                bins[i] += 1
    #print(bins)            
    return bins

def localMin_left(data,peak):
    for i,t in enumerate(np.flip(data[:peak])):
        if t > data[i+1]:
            continue
        else:
            return peak+i

def localMin_right(data,peak):
    for i,t in enumerate(data[peak:]):
        if t > data[i+1]:
            continue
        else:
            return peak+i
    



#green
def find_lim_right(data,thresh):
    peaks,_ = find_peaks(data,height= thresh)
    step = np.array([])
    for i,t in enumerate(peaks):
        step = np.append(step,t-3)
    #print(step)
    return step.astype(int)

#orange
def find_lim_left(data,thresh):
    peaks,_ = find_peaks(data,height= thresh)
    step = np.array([])
    for i,t in enumerate(peaks):
        step = np.append(step,t+6)
    #print(step)
    return step.astype(int)
    

def get_span(foot):
    base  =  np.bincount(foot.astype(int)).argmax()
    thresh = ((np.amax(foot)-base) * .25) + base
    peaks,_ = find_peaks(foot,height= thresh)
    beg = np.array([])
    end = np.array([])
    for i,t in enumerate(peaks):

        try:
            dis = (peaks[i+1]-t)//3
        except:
            dis = (t-peaks[i-1])//3
        #print(i,t,dis)
        beg = np.append(beg,t-dis)
        end = np.append(end,t+dis)
    beg = np.divide(beg,samplerate)
    end = np.divide(end,samplerate)
    return beg,end
    


def find_val(arr,height):
    arr = np.subtract(arr, grav)
    arr = -arr
    arr = np.add(arr, grav)
    print(arr)
    return find_peaks(arr,height=height)




def filt(data, samplerate, ct =  3, nt = 20):
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
    b,x = win.h('gauss',num_box)
    #data = np.pad(data, (num_box//2, num_box//2), 'constant')
    filtered = filtfilt(b,1.0,data)
    return filtered

def printOG(df_wrist,df_left,df_right,time):
    fig, ax = plt.subplots(3)
    fig.set_size_inches(20.0,10.0)
    for i in df_wrist.columns:
        ax[0].plot(time,df_wrist[i])
    for i in df_left.columns:
        ax[1].plot(time,df_left[i])
    for i in df_right.columns:
        ax[2].plot(time,df_right[i])
    ax[0].set_title("wrist")
    ax[0].set_xlabel('time (s)')
    ax[0].set_ylabel('acceleration')
    ax[1].set_title("left")
    ax[1].set_xlabel('time (s)')
    ax[1].set_ylabel('acceleration')
    ax[2].set_title("right")
    ax[2].set_xlabel('time (s)')
    ax[2].set_ylabel('acceleration')
    plt.show()

def l2Norm(df):
    x = df.iloc[:, 0]
    y = df.iloc[:, 1] 
    z = df.iloc[:, 2]
    dat = {'ave':np.sqrt(x**2 + y**2 + z**2)}
    df_ret = pd.DataFrame(dat)
    return df_ret

def delta(y):
    #this will return 1 or 0
    return

def getDat(fName):
    col = ['x Wrist','y Wrist','z Wrist','x Left','y Left','z Left','x Right','y Right','z Right']
    df = pd.read_csv(fName, names = col)
    df_wrist = df.filter(regex='Wrist')
    df_wrist = pd.concat([df_wrist,l2Norm(df_wrist)],axis =1)
    df_left = df.filter(regex='Left')
    df_left = pd.concat([df_left,l2Norm(df_left)],axis =1)
    df_right = df.filter(regex='Right')
    df_right = pd.concat([df_right,l2Norm(df_right)],axis =1)
    time = np.linspace(0, df.shape[0]/samplerate,df.shape[0])
    return time,df_wrist, df_left, df_right
def predict(time,df_wrist, df_left, df_right,fName):
    base  =  np.bincount(df_wrist['ave'].astype(int)).argmax()
    baseThresh = ((np.amax(df_wrist['ave'])-base) * .20) + base
    print("BaseThresh:",baseThresh)
   # specdft(df_left['ave'],df_right,df_wrist)
   
    plt.figure(figsize=(20,10))
    filt_wrist = filt(df_wrist['ave'],samplerate)
    
    #peaks,_ = find_peaks(filt_wrist,height= baseThresh)
    peaks = find_lim_right(filt_wrist,baseThresh)
    #vals,_ = find_val(filt_wrist,height= baseThresh)
    vals = find_lim_left(filt_wrist,baseThresh)
    plt.plot(time,filt_wrist, label = "wrist")
    filt_left = filt(df_left['ave'],samplerate)
    plt.plot(time,filt_left,color='orange',label = "left foot")
    filt_right = filt(df_right['ave'],samplerate)
    plt.plot(time,filt_right,color='green',label = "right foot")

    r_beg,r_end = get_span(filt_right)
    for i in range(len(r_beg)):
       plt.axvspan(r_beg[i],r_end[i],facecolor='g', alpha=0.5)

    l_beg,l_end = get_span(filt_left)
    for i in range(len(l_beg)):
       plt.axvspan(l_beg[i],l_end[i],facecolor='orange', alpha=0.5)

    r_bin = stepInBin(r_beg,r_end,peaks)
    st_r=my_stats(r_bin,peaks)

    l_bin = stepInBin(l_beg,l_end,vals)
    st_l = my_stats(l_bin,vals)
    
    sens,spec = sensAspec(st_l,st_r,len(r_beg),len(l_beg),len(filt_wrist))
    plt.plot(np.divide(peaks,samplerate), filt_wrist[peaks], "o", label="r_detect",color = 'green')
    plt.plot(np.divide(vals,samplerate), filt_wrist[vals], "o", label="l_detect",color = "orange")
    plt.xlabel("time (s)")
    plt.ylabel("Acceleration")
    plt.legend()
    plt.title(fName)
    plt.show()
    return sens,spec

def main(fName):
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
    time,df_wrist, df_left, df_right = getDat(fName)
    printOG(df_wrist,df_left,df_right,time)
    specdft(df_left,df_right,df_wrist)

    predict(time,df_wrist, df_left, df_right,fName)
   

    
    
    
    

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fileName', '-f', default='Accelerometers-20200716-11053032_Walk.csv', type=str,
                        help='The file name we want to process. Default: ^DJI.csv')

    args = parser.parse_args()

    main(args.fileName)