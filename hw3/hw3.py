import numpy as np
import matplotlib.pyplot as plt

samplerate = 10

def filthify(sig):
    mu, sigma = 0, 0.1 
    noise = np.random.normal(mu, sigma, len(sig)) 
    dirtySig =  sig + noise
    return dirtySig

def graph(sig,time):
    plt.plot(time,sig)
    plt.show()
    return

def make_sig():
    sig = np.array([.1,.2,.3,.4,.5,2,2,-2,-2,-.5,-.4,-.3,-.2,-.1])
    return sig

def main():
    signal = make_sig()
    time = np.linspace(0,len(signal),len(signal))
    
    graph(signal,time)
    
    return

if __name__ == "__main__":
    main()