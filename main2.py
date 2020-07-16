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

if __name__ == '__main__':

    num_inputs = len(sys.argv)

    #Song files to be hashed into database
    songs = []
    songnames = []
    separator = '.'

    for i in range(1,num_inputs):
    	songs.append(read(sys.argv[i]))
    	name = sys.argv[i].split(separator,1)[0]
    	songnames.append(name)

    #TDFT parameters
    windowsize = 0.008     #set the window size  (0.008s = 64 samples)
    windowshift = 0.004    #set the window shift (0.004s = 32 samples)
    fftsize = 1024         #set the fft size (if srate = 8000, 1024 --> 513 freq. bins separated by 7.797 Hz from 0 to 4000Hz) 
    
    #Peak picking dimensions 
    f_dim1 = 30
    t_dim1 = 80 
    f_dim2 = 10
    t_dim2 = 20
    percentile = 70
    base = 70 # lowest frequency bin used (peaks below are too common/not as useful for identification)
    high_peak_threshold = 75
    low_peak_threshold = 60

    #Hash parameters
    #delay_time = 250      # 250*0.004 = 1 second
    #delta_time = 250*3    # 750*0.004 = 3 seconds
    #delta_freq = 128      # 128*7.797Hz = approx 1000Hz
    #Hash parameters for short audio
    delay_time = 50      # 250*0.004 = 1 second
    delta_time = 50*3    # 750*0.004 = 3 seconds
    delta_freq = 24      # 128*7.797Hz = approx 1000Hz
    
    #Time pair parameters
    TPdelta_freq = 4
    TPdelta_time = 2

    #Construct the audio database of hashes
    database = np.zeros((0,5))
    durations = []
    spectrodata = []
    peaksdata = []

    for i in range(0,len(songs)):

    	print('Analyzing '+str(songnames[i]))
    	srate = songs[i][0]  #sample rate in samples/second
    	audio = songs[i][1]  #audio data    	
    	spectrogram = tdft.tdft(audio, srate, windowsize, windowshift, fftsize)
    	time = spectrogram.shape[0]
    	freq = spectrogram.shape[1]

    	threshold = pp.find_thres(spectrogram, percentile, base)

    	print('The size of the spectrogram is time: '+str(time)+' and freq: '+str(freq))
        duration = time * windowshift
        print('The duration of the song is: %.2f' % duration)
        durations.append(duration)
    	spectrodata.append(spectrogram)

    	peaks = pp.peak_pick(spectrogram,f_dim1,t_dim1,f_dim2,t_dim2,threshold,base)

    	print('The initial number of peaks is:'+str(len(peaks)))
    	peaks = pp.reduce_peaks(peaks, fftsize, high_peak_threshold, low_peak_threshold)

    	print('The reduced number of peaks is:'+str(len(peaks)))
        print('The 3 front element of reduced peaks:'+str(peaks[:3]))
    	peaksdata.append(peaks)

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
    songbins= np.zeros((numSongs, numSongs))
    numOffsets = len(timepairs)
    offsets = np.zeros(numOffsets)
    index = 0
    for i in timepairs:
    	offsets[index]=i[0]-i[1]
    	index = index+1
    	songbins[int(i[2])][int(i[3])] += 1
    for i in range(numSongs):
        for j in range(numSongs):
            songbins[i][j] = round(songbins[i][j] / durations[i] / durations[j], 2)
    print('Final similarity matrix:')
    np.set_printoptions(threshold=10000)
    print(songbins)
    np.savetxt("result.csv", songbins, delimiter=',')

