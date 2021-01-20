import numpy as np
import matplotlib.pyplot as plt
import random as rand
import argparse
from scipy import signal


def zer_pad(x):
    length =  len(x)
    xx = x
    xx = np.delete(xx, np.where(xx == 0))
    xxx = np.split(xx,2)
    x1 = xxx[0]
    x2 = xxx[1]
    x1 =  x1 - 1
    x2 = x2 + 1
    x = np.append(x1,x)
    x =  np.append(x,x2)
    return x

def triangles(length, throwaway1,throwaway2):
    x = np.linspace(0,(length//2),length//2)
    if length % 2 == 0:
        y1 = x + 1
        y2 = np.flip(y1)
        y =  np.append(y1,y2)
        x = np.linspace(-1,1,length)
        print(x,y)
        #x = zer_pad(x)
        #y = np.pad(y,(length//2,length//2), 'constant')
    else:
        y1 = x + 1
        y2 = np.flip(y1)
        #y2 = y2[:-1]
        ys = np.array([length])
        y1 =  np.append(y1,ys)
        y =  np.append(y1,y2)
        x = np.linspace(-1,1,length) 
        print(x,y)
        #x = zer_pad(x)
        #y = np.pad(y,(length//2,length//2), 'constant')
    return y,x

def gaussian(length, mu, sig):
    x = np.linspace(-2,2,length)
    g = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    #x = zer_pad(x)
    #g = np.pad(g,(length//2,length//2), 'constant')
    return g,x

def box(length,throwaway1, throwaway2):
    x = np.linspace(-1,1,length) 
    g = np.full(length,1)
    #x = zer_pad(x)
    #g = np.pad(g,(length//2,length//2), 'constant')
    return g,x

def rando(length,throwaway1, throwaway2):
    x = np.linspace((0-length//2),(length//2),length)
    if length % 2 == 0:
        y1 = np.random.randint(100,size=length//2)
        y2 = np.flip(y1)
        y =  np.append(y2,y1)
        x = np.linspace(-1,1,length) 
        #x = zer_pad(x)
        #y = np.pad(y,(length//2,length//2), 'constant')
    else:
        y1 = np.random.randint(100,size=length//2+1)
        y2 = np.flip(y1)
        y2 = y2[:-1]
        y =  np.append(y2,y1)
        x = np.linspace(-1,1,length) 
        #x = zer_pad(x)
        #y = np.pad(y,(length//2,length//2), 'constant')
    return y,x

def h(myfunc,length=7):
    #even symmetry
    #apply function
    #print(length)
    g,x = funcdict[myfunc](length,0,1)
    #must sum to 1
    g = g/g.sum(axis=0,keepdims=1)
    #g = np.pad(g,(length//4,length//4), 'constant')
    print('Filter sum:',np.sum(g))
    #print(g)
    #print(len(g),len(x))
    return g,x

funcdict = {
  'gauss': gaussian,
  'box' : box,
  'triang' : triangles,
  'rand' : rando
}

def main(funcN,sizze):
    for i in range(1,9):
        i = (i + 2)
        y,x = h(funcN,i)
        w, H = signal.freqz(y)
        lab = funcN+': ' +str(i)
        plt.plot(w,abs(H),label=lab)
    plt.title('Magnitude: '+ funcN)
    plt.xlim([0,np.pi])
    plt.ylim([0,1*1.05])
    plt.ylabel('Magnitude')
    plt.legend()
    plt.axhline((1/2**.5),ls='--', label='-3db')          ##cutoff frequency
    plt.axvline((np.pi * .1),ls='--', label='.1 pi') 
    plt.axvline((np.pi * .2),ls='--', label='.2 pi') 
    plt.axvline((np.pi * .3),ls='--', label='.3 pi')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fName', '-f', default="rand", type=str,
                        help='name of function')
    parser.add_argument('--sizze', '-s', default=7, type=int,
                        help='size of the window')

    args = parser.parse_args()
    main(args.fName, args.sizze)