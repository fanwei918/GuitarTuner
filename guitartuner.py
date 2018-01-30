#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 13:09:17 2017

@author: weif
"""

import numpy as np
import struct
import pyaudio
import matplotlib.pyplot as plt
#from termcolor import colored, cprint
import librosa


CHUNK = 4410  # num of data samples read per 
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

RECORD_SECONDS = 2
PREDICTION_SECONDS = 2


BLOCK = RECORD_SECONDS * RATE  # BLOCK is the prediction window
GAP = (PREDICTION_SECONDS * RATE) / CHUNK  # GAP should be int; it is the prediction gap
i = 0

# Create the audio stream

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Initialize the data container

frames = bytes(str([]), 'utf-16')


def crop(y, thres, window):
    n_win = y.shape[0]//window
    keep_index = np.ones_like(y)
    start = 0
    stop = 0
    for n in range(n_win//3):
        chunk = y[n*window:(n+1)*window]
        if np.abs(chunk).mean() < thres:
            stop = (n+1)*window
    keep_index[start:stop] = 0 
    start = y.shape[0]
    stop = y.shape[0]
    for n in range(n_win//3):
        chunk = y[y.shape[0]-(n+1)*window : y.shape[0]-n*window]
        if np.abs(chunk).mean() < thres:
            stop = y.shape[0]-(n+1)*window
    keep_index[stop:start] = 0   
    
    return y[keep_index==1]
            
            
            


def autocor(y,lowlim = 1,uplim = 250):
    result = np.zeros((uplim,))
    for shift in range (lowlim,uplim):
        y1 = np.zeros_like(y)
        y1[shift:] = y[0: y.shape[0]-shift ] 
        result[shift] = (y*y1).sum()/((y*y).sum() + 1000)
#    plt.plot(y)
#    plt.plot(y1)
#    plt.plot(result)
#    plt.show()
    return result


#def autocor(y,lowlim = 40,uplim = 200, winlen = 512):
#    n_win = (y.shape[0]-uplim)//winlen
#    result = np.zeros((uplim,n_win))
#    for n in range(n_win):
#        y_piece = y[uplim+n*winlen : uplim+(n+1)*winlen]
#        for shift in range (lowlim,uplim):
#            y_piece1 = np.zeros_like(y_piece)
#            y_piece1 = y[uplim+n*winlen-shift : uplim+(n+1)*winlen-shift]
#            result[shift,n] = (y_piece * y_piece1).mean()
#        
##    plt.plot(y)
##    plt.plot(y1)
##    plt.plot(result.max(axis = 1))
#    plt.imshow(result)
#    plt.show()
#    return result



'''
Starting recording
'''

print("* recording")



try:
    while True:    
        record = stream.read(CHUNK, exception_on_overflow=False)
        frames = frames + record
        
        i = i + 1
        if i % GAP == 1 and i > BLOCK/CHUNK:
            frame_len = len(frames)
            if frame_len > (BLOCK * 2):  #the bytes type has 2 times length of the real data it representes
                frames = frames[frame_len - BLOCK*2:frame_len]
                data_form = "%dh" % (BLOCK)
                y = (np.array(struct.unpack(data_form, frames))+0.5) / ((0x7FFF + 0.5))
                
                y= (librosa.resample(y,RATE,16000)*32768).astype(int)
#                plt.plot(y)
#                plt.show()
#                y = 0.03383324*y[0:-4] + 0.24012702 *y[1:-3] + 0.45207947*y[2:-2] + 0.24012702*y[3:-1] + 0.03383324*y[4:]
#                y = y[0::8]
#                y = (y[0:-3]+y[1:-2]+y[2:-1]+y[3:] )/4
                y = (y[0:-1] + y[1:])/2
                
#                D = librosa.stft(y,n_fft=1024, hop_length=1024, win_length=None, window='han', center=False)
#                freq = (np.abs(D)).sum(axis = 1)
#                ceps = librosa.stft(freq,n_fft=512, hop_length=512, win_length=None, window=np.ones((512,)), center=False) 
#                ceps = np.abs(ceps)
#                plt.imshow(np.abs(D[0:]))
                
#                noisefloor = int(np.abs(y).mean())+1
#                y = (y//noisefloor)
                
                
#                plt.plot(y)
#                plt.show()
#                
#                print(np.abs(y).mean())
                
                thre = 400
                
                if np.abs(y).mean()<thre:
                    print("no major sounds heard")
                    continue
                
                y = crop(y,thre,window = 50)
                plt.plot(y)
                plt.show()
                
                result = autocor(y, uplim = 300)
                freq = 0
                maxloc = 0
                maxval = result[10:250].max()*0.9
                for n in range(10,250):
                    if result[n]>result[n-2:n].max() and result[n]>result[n+1:n+3].max() and result[n]>maxval  :
                        maxloc = n
                        break
                if maxloc ==0:
                    print('not guitar sound heard')
                else:
                    print("the tone heard is ")
                    freq = 16000/(maxloc)  
                    print(freq)
#                    print(maxloc)
#                    print(maxval)
                    
except KeyboardInterrupt:
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    
#%%
    
    
import glob
import soundfile as sf
sr_desired = 16000

for filename in glob.glob('Guitar/*.ogg'):
    y,sr = sf.read(filename, dtype = 'float32')
    if y.ndim>1 and y.shape[1]>1:
        y=y[:,0]
    y = (librosa.resample(y,sr,sr_desired)*32768).astype(int)
    
    print(filename)
    
#    np.savetxt(filename.split('_')[0],y)
    
    y = (y[0:-3]+y[1:-2]+y[2:-1]+y[3:] )/4
    
    plt.plot(y)
    plt.show()
    
    result = autocor(y)
    plt.plot(result)
    plt.show()
    
    maxval = result[25:].max() * 0.8
    for n in range(25,result.shape[0]-1):
        if result[n]>result[n-1] and result[n]>result[n+1] and result[n]>maxval  :
            maxloc = n
            break
    freq = 16000/maxloc
    print(maxloc)
    print(freq)
    
#    break
    
#%%
    
    
import glob
import soundfile as sf
sr_desired = 16000

for filename in glob.glob('Guitar/*.ogg'):
    y,sr = sf.read(filename, dtype = 'float32')
    if y.ndim>1 and y.shape[1]>1:
        y=y[:,0]
    y = (librosa.resample(y,sr,sr_desired)*32768).astype(int)
    
    print(filename)
    
#    np.savetxt(filename.split('_')[0],y)
    
    y = y[0::8]
#    y = (y[0:-1]+y[1:])/2
    result = autocor(y)
    plt.plot(result)
    plt.show()
    
    maxval = result[2:].max() * 0.8
    for n in range(2,40):
        if result[n]>result[n-2:n].max() and result[n]>result[n+1:n+3].max() and result[n]>maxval  :
            maxloc = n
            break
    freq = 2000/maxloc
    print(maxloc)
    print(freq)
    
#    break
        
    #%%
    
from scipy import signal

#b = np.array([np.random.random()-0.5,np.random.random()-0.5,np.random.random()-0.5])

b = signal.firwin(2, 0.1)
print(b)
w, h = signal.freqz(b)
plt.plot(w, abs(h), 'b')




    
