import scipy.io as sio
from scipy.io.wavfile import read
import os,sys
from os import listdir
from os.path import isfile, join
import argparse
import numpy as np
import matplotlib.pyplot as plt
import heuristic as hu
import seaborn as sns

samplerate = 22000
sampSize = 40

def rideWave(file):
    a = read(file)
    return a[0], np.array(a[1],dtype=float)

def getData(fName):
    mydir = os.getcwd()
    fol = "Archive/Data/"
    # Join various path components  
    fNames = os.path.join(mydir,fol,fName) 
    if fNames[-4:] != ".mat":
        fNames = fNames + '.mat'
    if os.path.isfile(fNames) == False:
        print("no file with that name!")
        return
    #print(fNames)
    data = sio.loadmat(fNames)
    #print(data.keys())
    #print(data['x'].shape[1])
    if data['x'].shape[1] is not 1:
        datax = data['x'].flatten()
    else:
        datax = data['x']
    spikes = data['si']
    return datax, spikes

def getSpike(data, spike,sizze):
    spike = int(spike)
    dataS = np.array(np.append(data[spike - sizze: spike],data[spike: spike + sizze]))
    time = np.linspace(0,len(dataS),len(dataS))
    #plt.plot(time, dataS)
    #plt.vlines(len(dataS)//2,np.min(data),np.max(data),colors='red')
    #plt.show()
    return dataS

def fileRead():
    mypath = os.getcwd()
    fol = "Archive/Data/"
    mypath = os.path.join(mypath,fol)
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    #onlyfiles.pop(5)
    #onlyfiles.pop(10)
    onlyfiles=onlyfiles[:10]
    print(onlyfiles)
    for fi in onlyfiles:
        if fi[-4:] != '.mat':
            continue
        else:
            try:
                stuff = np.vstack((stuff,main(fi)))
            except:
                dataS = main(fi)
                stuff = np.array(dataS)
                time = np.linspace(0,sampSize-1,sampSize)
    plt.figure(figsize=(20,10))
    for i in stuff:
        plt.plot(time/samplerate,i,alpha=.5,color="skyblue")
    daMean = np.mean(stuff, axis=0)
    
    plt.plot(time/samplerate,daMean,color='r')
    plt.title("Mean of 10")
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.show()
    #np.savetxt("meanBitch20.csv", daMean, delimiter=",") ###########################
    for i,t in enumerate(stuff):
        stuff[i] = t- daMean
    plt.figure(figsize=(20,10))
    for i in stuff:
        plt.plot(time/samplerate,i,alpha=.5,color="skyblue")
    plt.title("removed signal")
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.show()
    #counts, bins = np.histogram(stuff[:,35:])
    #plt.hist(bins[:-1], bins, weights=counts)
    sns.histplot(data=stuff[:,35:], kde=True)
    plt.title("Noise Histogram")
    plt.show()
    daSTD = np.std(stuff[:,35:], axis=0)
    print("daSTD:", np.mean(daSTD))
def main(fName):
    
    #data,spikes = getData(fName)
    data,spikes = hu.run(fName,1000,11,1,.1,0,1)
    print(len(spikes))
    time = np.linspace(0,len(data),len(data))
    #plt.plot(time, data)
    #plt.vlines(spikes[0],np.min(data),np.max(data),colors="red")
    #plt.show()
    for i in spikes:
        #dataS = getSpike(data,i,50)
        try:
            dataS = getSpike(data,i,sampSize//2)
            if dataS.shape[0] is sampSize:
                stuff = np.vstack((stuff,dataS))
                
        except:
            dataS = getSpike(data,i,sampSize//2)
            if dataS.shape[0] is sampSize:
                stuff = np.array(dataS)
    return stuff


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fileName', '-f', default='MER0102', type=str,
                        help='The file name we want to process. Default: ^DJI.csv')
    args = parser.parse_args()

    #main(args.fileName)
    fileRead()