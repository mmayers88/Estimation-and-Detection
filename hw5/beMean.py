import scipy.io as sio
from scipy.io.wavfile import read
import os,sys
from os import listdir
from os.path import isfile, join
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def getSamp(data,mid,sizze):
    samp = np.array(data[mid:mid+sizze//2])
    sampEnd = np.array(data[mid-sizze//2:mid])
    samp = np.append(sampEnd,samp)
    return samp

def retMids(spikes,samplerate):
    mids = np.array([])
    for i,t in enumerate(spikes):
        if i % 2 == 0:
            dist = (spikes[i+1] - t) * samplerate
            mids = np.append(mids,t*samplerate + dist//2)
    #print(mids.astype(int))
    return mids.astype(int)


def meanAve(spikes,samplerate):
    dist = 0
    counts = 0
    for i,t in enumerate(spikes):
        if i % 2 == 0:
            counts += 1
            dist +=(spikes[i+1] - t) * samplerate
    return int(dist//counts)

def getData(fName):
    mydir = os.getcwd()
    fol = "Archive/Data/"
    # Join various path components  
    fNames = os.path.join(mydir,fol,fName) 
    if fNames[-4:] != ".csv":
        fNames = fNames + '.csv'
    print(fNames)
    if os.path.isfile(fNames) == False:
        print("no file with that name!")
        return
    data = np.loadtxt(fNames, delimiter=',')
    timeUnit = data[:,0][1] - data[:,0][0]
    samplerate = 1/timeUnit
    #print(samplerate)
    return data[:,2], samplerate
    

def getSpike(fName):
    mydir = os.getcwd()
    fol = "Archive/Events/"
    # Join various path components  
    fNames = os.path.join(mydir,fol,fName) 
    if fNames[-4:] != ".csv":
        fNames = fNames + '.csv'
    print(fNames)
    if os.path.isfile(fNames) == False:
        print("no file with that name!")
        return
    dataS = np.loadtxt(fNames, delimiter=',').flatten()
    dataS = dataS.reshape((-1,2))[: : 2,:].flatten()
    dataS = dataS[~np.isnan(dataS)]
    #print(dataS)
    return dataS

def plotNow(data,samplerate,spikes = None):
    time = np.linspace(0,len(data)/samplerate,len(data))
    plt.figure(figsize=(16,8))
    if spikes is not None:
        plt.vlines(spikes,np.amin(data),np.amax(data),colors='orange')
    plt.plot(time, data)
    plt.xlabel("time (s)")
    plt.ylabel("velocity (m/s)")
    plt.show()
    return

def fileRead():
    mypath = os.getcwd()
    fol = "Archive/Data/"
    mypath = os.path.join(mypath,fol)
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    #onlyfiles.pop(5)
    #onlyfiles.pop(10)
    #onlyfiles=onlyfiles[:10]
    #print(onlyfiles)
    for fi in onlyfiles:
        if fi[-4:] != '.csv':
            continue
        else:
            daMean = main(fi)
            mySavepath = os.getcwd()
            folS = "/means/"
            Savepath = os.path.join(mySavepath,folS,fi)
            mySavepath = mySavepath+Savepath
            if isfile(mySavepath):
                os.remove(mySavepath)
            np.savetxt(mySavepath, daMean, delimiter=",")
            

def main(fName):
    data,samplerate = getData(fName)
    spikes = getSpike(fName)
    #plotNow(data,samplerate,spikes)
    sampSize = meanAve(spikes,samplerate)
    mids = retMids(spikes,samplerate)
    
    for i,t in enumerate(mids):
        try:
            dataS = getSamp(data,int(t),sampSize)
            stuff = np.vstack((stuff,dataS))
                
        except:
            dataS = getSamp(data,int(t),sampSize)
            stuff = np.array(dataS)
            time = np.linspace(0,dataS.shape[0]-2,dataS.shape[0])
    #print(time.shape,dataS.shape)

    plt.figure(figsize=(16,8))
    for i in stuff:
        plt.plot(time/samplerate,i,alpha=.5,color="skyblue")
    daMean = np.mean(stuff, axis=0)
    
    '''
    plt.plot(time/samplerate,daMean,color='r')
    plt.title("Mean of 10")
    plt.xlabel("time")
    plt.ylabel("amplitude")
    '''
    #plt.show()
    return daMean
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fileName', '-f', default='Subject000', type=str,
                        help='The file name we want to process. Default: ^DJI.csv')
    args = parser.parse_args()

    #main(args.fileName)
    fileRead()