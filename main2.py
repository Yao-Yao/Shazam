''' 
Music Identification Program (a.k.a. Shazam/Soundhound) 
Proof of Concept 
Bryant Moquist
'''
from __future__ import print_function
from scipy.io.wavfile import read
import sys
import peakpicker as pp
import fingerprint as fhash
import numpy as np
import tdft
import math

if __name__ == '__main__':

    num_inputs = len(sys.argv)

    #Song files to be hashed into database
    songs = []
    songnames = []
    separator = '.'

    for i in range(1,num_inputs):
        songs.append(read(sys.argv[i]))
        name = sys.argv[i].rsplit(separator,1)[0]
        songnames.append(name)

    #TDFT parameters
    windowsize = 0.008     #set the window size  (0.008s = 64 samples)
    windowshift = 0.004    #set the window shift (0.004s = 32 samples)
    fftsize = 1024         #set the fft size (if srate = 8000, 1024 --> 513 freq. bins separated by 7.797 Hz from 0 to 4000Hz) 
                                            #(if srate = 44100, 1024 --> 513 freq. bins separated by 42.982 Hz from 0 to 22050Hz) 
    
    #Peak picking dimensions 
    f_dim1 = 30
    t_dim1 = 80 
    f_dim2 = 10
    t_dim2 = 20
    percentile = 70
    #base = 70 # lowest frequency bin used (peaks below are too common/not as useful for identification)
    base = 8 # lowest frequency bin used (peaks below are too common/not as useful for identification) = 320Hz
    #watch [20Hz to 20kHz (Human Audio Spectrum)](https://www.youtube.com/watch?v=qNf9nzvnd1k), 60-15kHz for me
    top = 375 # 375*42.982Hz = approx 15kHz
    high_peak_threshold = 75
    low_peak_threshold = 60

    #Hash parameters
    #delay_time = 250      # 250*0.004 = 1 second
    #delta_time = 250*3    # 750*0.004 = 3 seconds
    #delta_freq = 128      # 128*42.982Hz = approx 5500Hz
    #Hash parameters for short audio
    delay_time = 50      # 50*0.004 = 0.2 second
    delta_time = 50*3    # 150*0.004 = 0.6 seconds
    delta_freq = 12      # 12*42.982Hz = approx 516Hz
    
    #Time pair parameters
    TPdelta_freq = 4
    TPdelta_time = 2

    #Construct the audio database of hashes
    database = np.zeros((0,5))
    durations = []

    for i in range(0,len(songs)):
        print('Analyzing id='+str(i)+': '+str(songnames[i]))
        srate = songs[i][0]  #sample rate in samples/second
        audio = songs[i][1]  #audio data        
        print('The srate of the audio is: '+str(srate))

        spectrogram = tdft.tdft(audio, srate, windowsize, windowshift, fftsize)
        time = spectrogram.shape[0]
        freq = spectrogram.shape[1]
        print('The size of the spectrogram is time: '+str(time)+' and freq: '+str(freq))
        duration = time * windowshift
        print('The duration of the song is: %.2f' % duration)
        durations.append(duration)

        threshold = pp.find_thres(spectrogram, percentile, base, top)
        peaks = pp.peak_pick(spectrogram,f_dim1,t_dim1,f_dim2,t_dim2,threshold,base, top)
        print('The initial number of peaks is:'+str(len(peaks)))

        peaks = pp.reduce_peaks(peaks, fftsize, high_peak_threshold, low_peak_threshold)
        print('The reduced number of peaks is:'+str(len(peaks)))
        print('The 3 front element of reduced peaks:'+str(peaks[:3]))

        #Calculate the hashMatrix for the database song file
        songid = i
        hashMatrix = fhash.hashPeaks(peaks,songid,delay_time,delta_time,delta_freq)
        print('hashMatrix of shape:'+str(hashMatrix.shape))

        #Add to the song hash matrix to the database
        database = np.concatenate((database,hashMatrix),axis=0)
        print('database of shape:'+str(database.shape))
        print()

    timepairs = fhash.findTimePairs2(database, TPdelta_freq, TPdelta_time)
    np.set_printoptions(threshold=10000)
    print('timepairs of length:'+str(len(timepairs)))

    #Compute number of matches by song id to determine a match
    numSongs = len(songs)
    songbins = np.zeros((numSongs, numSongs))
    numOffsets = len(timepairs)
    offsets = np.zeros(numOffsets)
    index = 0
    for _t0, t0, _songID, songID in timepairs:
        offsets[index]=_t0-t0
        index = index+1
        songbins[int(_songID)][int(songID)] += 1
    selfsim = np.zeros((numSongs, numSongs))
    for i in range(numSongs):
        selfsim[i][i] = songbins[i][i] / durations[i] / durations[i]
    for i in range(numSongs):
        for j in range(numSongs):
            songbins[i][j] = round(songbins[i][j] / durations[i] / durations[j] / math.sqrt(selfsim[i][i] * selfsim[j][j]), 2)
    print('Final similarity matrix:')
    np.set_printoptions(threshold=10000)
    print(songbins)
    np.savetxt("result.csv", songbins, delimiter=',')

