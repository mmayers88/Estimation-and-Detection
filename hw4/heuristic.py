import argparse
import numpy as np
from scipy import signal
from scipy.signal import freqz, find_peaks
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, rfft,fftfreq
from scipy.io.wavfile import read
import findMean as fm

fileName = "MER0106"
samplerate = 22000

def rideWave(file):
    a = read(file)
    return a[0], np.array(a[1],dtype=float)

def myPlot(data,filt,samplerate):
    length = data.shape[0] / samplerate
    time = np.linspace(0., length, data.shape[0])
    plt.plot(time, data, label="Brainwave Data")
    plt.plot(time, filt, label="Filtered")
    plt.legend(loc = 'best')
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()

def filt(data, samplerate, ct =  1000, nt = 11):
    '''
    sos = signal.butter(3, 4100, 'highpass', fs=samplerate, output='sos')
    filtered = signal.sosfilt(sos, data)
    b = signal.firwin(11, cutoff=4200, fs=samplerate, pass_zero=False, window=20)
    filtered = signal.lfilter(b, [1.0], data)
    '''
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.firwin.html
    b = signal.firwin(nt, cutoff=ct, fs=samplerate, pass_zero=False, window='hamming')
    filtered = signal.lfilter(b, [1.0], data)
    return filtered


def run(fileN,ct,nt,passr = 0, thresh = 1000, ffte = 0, verb = 0):
    #samplerate,data = rideWave(file)
    data,_ = fm.getData(fileN)
    data = data.flatten()
    #print("Sample Rate: ",samplerate)
    filtered = filt(data, samplerate,ct,nt)
    X = fft(filtered)
    freqs = fftfreq(len(X)) * samplerate

    if verb != 0:
        #print(filtered)
        #print(thresh)
        newthresh  = np.max(filtered) * .5
        peaks, _ = find_peaks(filtered, height=newthresh)
        #print("Peak Number: ",len(peaks))
        time = np.linspace(0,len(data)/samplerate,len(data))
        '''
        plt.figure(figsize=(20,10))
        plt.plot(time,filtered, label="Filtered Data")
        plt.vlines(peaks/samplerate, -.25,.25,color='orange', label="Detections")
        #plt.plot(np.zeros_like(filtered), "--", color="gray")
        plt.legend(loc = 'best')
        plt.title(fileN)
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.show()
        '''
        
        return data, peaks

    


if __name__ == "__main__":
    # execute only if run as a script
    parser = argparse.ArgumentParser()
    parser.add_argument('--cutoff', '-c', default=1000, type=int,
                        help='Number of days')
    parser.add_argument('--numt', '-n', default=11, type=int,
                        help='numtaps')
    parser.add_argument('--passr', '-p', default=1, type=int,
                        help='how many filter passes')
    parser.add_argument('--thresh', '-t', default=.1, type=float,
                        help='minimum for peaks')
    parser.add_argument('--fft', '-f', default=0, type=int,
                        help='print fft')
    parser.add_argument('--verb', '-v', default=1, type=int,
                        help='this is for peaks')
                        
    args = parser.parse_args()
    run(fileName,args.cutoff, args.numt,args.passr, args.thresh, args.fft, args.verb)

    #.\hw2.py -c 1000 -p 2 -t 1000