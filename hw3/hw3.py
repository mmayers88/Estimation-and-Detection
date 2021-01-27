import numpy as np
import matplotlib.pyplot as plt


#=====================================================
# General Settings
#=====================================================
samplerate = 10
windowSize = 25

#=====================================================
#Gaussian noise settings
#=====================================================
mu, sigma = 0, 1

#=====================================================
#ML Settings
#=====================================================
pi_0 = .5
pi_1 = .5
c_00 = 0
c_01 = 0
c_10 = 0
c_11 = 0


def veri(TP,pred):
    bins = [0] * len(TP)
    for i in range(len(TP)):
        for j in pred:
            #print(beg[i],j,end[i])
            if TP[i] == j:
                bins[i] += 1
           
    return bins

def likly(window, sig):
    LHS = 0
    for i,k in enumerate(window):
        LHS += k*(sig[i])
    return (LHS)
    #return (LHS/sigma)

def RHS(window,sig):
    rhs = np.inner(sig,window)/2
    return rhs

def get_eta(sig):
    eta = np.array([x/2 for x in sig])
    return eta


def magic(window,sig):
    LHS = likly(window,sig)
    Right_Side = RHS(sig,sig)
    #print(LHS,Right_Side)
    if LHS >= Right_Side:
        #print(LHS,Right_Side)
        return True
    return False

def analyze(signal,sig):
    prediction = np.array([])
    for i in range((len(signal)-windowSize)):
        window = np.array(signal[ i : (i+windowSize) ]) #make a window
        if magic(window,sig) ==  True:
            #print(i,(i+windowSize))
            prediction = np.append(prediction,i)
    return prediction

def make_sig(sig):
    signal = np.zeros(10000)
    randnums= np.random.randint(1,9975,10)
    randnums = np.sort(randnums, axis=None)
    dropPoints = np.array([])
    for i in randnums:
        testSig = np.array(signal[ i : (i+windowSize) ])
        if np.all(testSig == 0) == True:
            for j in range(len(sig)):
                signal[i+j] = sig[j]
            print(i,(i+windowSize))
            dropPoints = np.append(dropPoints,i)
        else:
            continue
    return signal, dropPoints


def filthify(sig): 
    noise = np.random.normal(mu, sigma, len(sig)) 
    dirtySig =  sig + noise
    return dirtySig

def graph(sig,time):
    plt.figure(figsize=(19,9))
    plt.plot(time,sig)
    plt.show()
    return

def small_sample():
    sig = np.array([.1,.2,.3,.4,.5,1,1,-1,-1,-.5,-.4,-.3,-.2,-.1,.3,.4,1,-1,-.1,-.2,-.5,.6,.9,.2,-1])
    return sig

def main():
    sig = small_sample()
    tim = np.linspace(0,len(sig),len(sig))
    #graph(sig,tim)
    signal,TP = make_sig(sig)
    time = tim = np.linspace(0,len(signal),len(signal))
    dirty_signal = filthify(signal)
    #graph(dirty_signal,time)
    plt.figure(figsize=(19,9))
    plt.plot(time,dirty_signal,label = "dirty")
    plt.plot(time,signal,label="clean signal")
    plt.legend()
    plt.show()
    pred = analyze(dirty_signal,sig)
    print(veri(TP,pred))
    #print("Correctness: {}%".format(veri(TP,pred)))
    return

if __name__ == "__main__":
    main()