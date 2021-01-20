import hw1 as hw
import os
from os import listdir
from os.path import isfile, join,exists
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt,convolve, freqz, find_peaks,firwin,lfilter,argrelextrema
import csv


outputTxt = "output.csv"
samplerate = 32
base = 9.7       #gravity
mydir = os.getcwd()
fol = "Archive/Data/"
# Join various path components  
mydir = os.path.join(mydir,fol)
onlyfiles = [join(mydir, f) for f in listdir(mydir) if isfile(join(mydir, f))]
stuff = []

if os.path.exists(outputTxt) ==  True:
    os.remove(outputTxt)

for fi in onlyfiles:
    if fi[-4:] != '.csv':
        continue
    else:
        
        time,df_wrist, df_left, df_right =  hw.getDat(fi)
        #hw.specdft(df_left,df_right,df_wrist)
        sens,spec = hw.predict(time,df_wrist, df_left, df_right,fi)
        if os.path.exists(outputTxt):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not
        with open(outputTxt,append_write) as fd:
            line = str(sens) +',' +str(spec) + '\n'
            fd.write(line)
