import numpy as np
import matplotlib.pyplot as plt
import scipy
import argparse


#=====================================================
# General Settings
#=====================================================
samplerate = 10
windowSize = 25

#=====================================================
#Gaussian noise settings
#=====================================================
mu = 0

#=====================================================
#ML Settings
#=====================================================
pi = [0,0]
pi[0] = .5
pi[1] = .5
c = [[0, 0],[0 ,0]]
c[0][0] = 0 #TN
c[0][1] = 1 #FP
c[1][0] = 10 #FN
c[1][1] = 0 #TP



def SNR(clean,noise):
    x = 0
    y = 0
    for i in clean:
        x += i**2
    for j in noise:
        y += j**2
    x = x/len(clean)
    y = y/len(noise)
    snr = x/y
    return snr


def bayes_risk(pf,pd):
    R = c[0][0]*pi[0] +  c[0][1]*pi[1] + (c[1][0]-c[0][0])*pf *pi[0] +(c[1][1]-c[0][1])*pd* pi[1]
    return R

def E_half(sig):
    E = np.inner(sig,sig)
    E = np.sqrt(E)
    return E

def makeRange(truePoints):
    start  = np.array([])
    end  = np.array([])
    for i in truePoints:
        #check within a second
        start = np.append(start,(i-6))
        end = np.append(end,(i+6))
    return start,end

def Theta():
    theta = ((c[1][0]-c[0][0])*pi[0])/((c[0][1]-c[1][1])*pi[1])
    theta_log = np.log(theta)
    return theta_log

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
    #print('fn:',fn,'tn:',tn,'tp:',tp,'fp:',fp)
    sens =  tp/(tp + fn)
    spec = 1-(fp/(fp+tn))
    PPV = tp/(tp+fp)
    NPV = tn/(tn+fn)
    print('Sensitivity: ',sens,'Specificity: ',spec)
    print("PPV: ",PPV,"NPV: ",NPV)
    return sens, spec, (1-PPV)

def my_stats(arr,att):
    unique, counts = np.unique(arr, return_counts=True)
    stats = dict(zip(unique, counts))
    stats['pred'] = len(att)
    #print(stats)
    return stats

def veri(beg,end,steps):
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

def likly(window, sig):
    E=E_half(sig)
    LHS = 0
    for i,k in enumerate(window):
        LHS += k*(sig[i])
    #return (LHS)
    return (LHS/E)

def RHS(window,sig,sigma):
    E = E_half(sig)
    theta = Theta()
    rhs = (E/2)+((sigma/E)*theta)
    return rhs

def get_eta(sig):
    eta = np.array([x/2 for x in sig])
    return eta


def magic(window,sig,sigma):
    LHS = likly(window,sig)
    Right_Side = RHS(sig,sig,sigma)
    #Right_Side =  Theta()
    #print(LHS,Right_Side)
    if LHS >= Right_Side:
        #print(LHS,Right_Side,RHS(sig,sig,sigma))
        return True
    return False

def analyze(signal,sig,sigma):
    prediction = np.array([])
    for i in range((len(signal)-windowSize)):
        window = np.array(signal[ i : (i+windowSize) ]) #make a window
        if magic(window,sig,sigma) ==  True:
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
            #print(i,(i+windowSize))
            dropPoints = np.append(dropPoints,i)
        else:
            continue
    return signal, dropPoints


def filthify(sig,sigma): 
    noise = np.random.normal(mu, sigma, len(sig)) 
    dirtySig =  sig + noise
    return dirtySig,noise

def graph(sig,time):
    plt.figure(figsize=(19,9))
    plt.plot(time,sig)
    plt.show()
    return

def small_sample():
    x = 0
    sig = np.array([.1,.2,.3,.4,.5,1,1,-1,-1,-.5,-.4,-.3,-.2,-.1,.3,.4,1,-1,-.1,-.2,-.5,.6,.9,.2,-1])
    for i in sig:
        x += i**2
    x = x / len(sig)
    print("Average Power: ",x)
    return sig

def predict(sig,dirty_signal,TP,sigma):
    pred = analyze(dirty_signal,sig,sigma)
    start,end = makeRange(TP)
    bins = veri(start,end,pred)
    #print("Correctness: {}%".format(veri(TP,pred)))
    stats = my_stats(bins,pred)
    sens,spec,NPV = sensAspec(stats,10,10000)
    return sens,pred,NPV

def loopy():
    sig = small_sample()
    tim = np.linspace(0,len(sig),len(sig))
    graph(sig,tim)
    signal,TP = make_sig(sig)
    time = tim = np.linspace(0,len(signal),len(signal))
    graph(signal,time)
    y = np.array([])
    x = np.array([])
    for i in range(100,2,-1):
        sigma = i/100
        
        dirty_signal,noise = filthify(signal,sigma)
        #graph(dirty_signal,time)
        sens,pred,NPV = predict(sig,dirty_signal,TP,sigma)
        br =bayes_risk(NPV,sens)
        y = np.append(y,br)
        x = np.append(x,SNR(signal,noise))
        print("SNR: ",SNR(signal,noise), "Bayes Risk: ", br)
    plt.figure(figsize=(19,9))
    plt.plot(x,y,label = "BR/SNR")
    plt.xlabel("SNR")
    plt.ylabel("Bayes Risk")
    plt.legend()
    plt.show()

def main(sigma):
    sig = small_sample()
    tim = np.linspace(0,len(sig),len(sig))
    graph(sig,tim)
    signal,TP = make_sig(sig)
    time = np.linspace(0,len(signal),len(signal))
    dirty_signal,noise = filthify(signal,sigma)
    #graph(dirty_signal,time)
    plt.figure(figsize=(19,9))
    plt.plot(time,dirty_signal,label = "dirty")
    plt.plot(time,signal,label="clean signal")
    plt.legend()
    plt.show()

    sens,pred = predict(sig,dirty_signal,TP,sigma)
    plt.figure(figsize=(19,9))
    plt.plot(time,dirty_signal,label = "dirty")
    plt.plot(time,signal,label="clean signal")
    plt.vlines(pred,1,4,colors="red")
    plt.legend()
    plt.show()

    print("SNR: ", SNR(signal,noise))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sigma', '-s', default=.1, type=float,
                        help='STD')
    args = parser.parse_args()

    #main(args.sigma)
    loopy()