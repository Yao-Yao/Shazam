'''
Hash and Acoustic Fingerprint Functions
Bryant Moquist
'''

import numpy as np

def findAdjPts(index,A,delay_time,delta_time,delta_freq):
    "Find the three closest adjacent points to the anchor point"    
    t0, f0, amp0 = A[index]
    adjPts = []
    low_x = t0+delay_time
    high_x = low_x+delta_time
    low_y = f0-delta_freq/2
    high_y = f0+delta_freq/2
    
    for i in A:
        t, f, amp = i
        if ((t>low_x and t<high_x) and (f>low_y and f<high_y)):
            adjPts.append(i)
            
    return adjPts
    
def hashPeaks(A,songID,delay_time,delta_time,delta_freq):
    "Create a matrix of peaks hashed as: [[freq_anchor, freq_other, delta_time], time_anchor, songID]"
    hashMatrix = np.zeros((len(A)*100,5))  #Assume size limitation
    index = 0
    numPeaks = len(A)
    for i in range(0,numPeaks):
        t0, f0, amp0 = A[i]
        adjPts = findAdjPts(i,A,delay_time,delta_time,delta_freq)
        adjNum=len(adjPts)
        for j in range(0,adjNum):
            t, f, amp = adjPts[j]
            hashMatrix[index][0] = f0
            hashMatrix[index][1] = f
            hashMatrix[index][2] = t-t0
            hashMatrix[index][3] = t0
            hashMatrix[index][4] = songID
            index=index+1
    
    hashMatrix = hashMatrix[~np.all(hashMatrix==0,axis=1)]
    hashMatrix = np.sort(hashMatrix,axis=0)
        
    return hashMatrix

def hashSamplePeaks(A,delay_time,delta_time,delta_freq):
    "Create a matrix of peaks hashed as: [[freq_anchor, freq_other, delta_time],time_anchor]"
    hashMatrix = np.zeros((len(A)*100,4))
    index = 0
    numPeaks = len(A)
    for i in range(0,numPeaks):
        t0, f0, amp0 = A[i]
        adjPts = findAdjPts(i,A,delay_time,delta_time,delta_freq)
        adjNum = len(adjPts)
        for j in range(0,adjNum):
            t, f, amp = adjPts[j]
            hashMatrix[index][0] = f0
            hashMatrix[index][1] = f
            hashMatrix[index][2] = t-t0
            hashMatrix[index][3] = t0
            index=index+1

    hashMatrix = hashMatrix[~np.all(hashMatrix==0,axis=1)]
    hashMatrix = np.sort(hashMatrix,axis=0)
        
    return hashMatrix

def findTimePairs(hash_database,sample_hash,deltaTime,deltaFreq):
    "Find the matching pairs between sample audio file and the songs in the database"

    timePairs = []

    for i in sample_hash:
        f0, f, dt, t0 = i
        for j in hash_database:
            _f0, _f, _dt, _t0, _songID = j
            if(f0 > (_f0-deltaFreq) and f0 < (_f0 + deltaFreq)):
                if(f > (_f-deltaFreq) and f < (_f + deltaFreq)):
                    if(dt > (_dt-deltaTime) and dt < (_dt + deltaTime)):
                        timePairs.append((_t0,t0,_songID))
                    else:
                        continue
                else:
                    continue
            else:
                continue
            
    return timePairs

def findTimePairs2(hash_database,deltaTime,deltaFreq):
    "Find the matching pairs between sample audio file and the songs in the database"

    timePairs = []

    for i in hash_database:
        f0, f, dt, t0, songID = i
        for j in hash_database:
            _f0, _f, _dt, _t0, _songID = j
            if(f0 > (_f0-deltaFreq) and f0 < (_f0 + deltaFreq)):
                if(f > (_f-deltaFreq) and f < (_f + deltaFreq)):
                    if(dt > (_dt-deltaTime) and dt < (_dt + deltaTime)):
                        timePairs.append((_t0,t0,_songID,songID))
                    else:
                        continue
                else:
                    continue
            else:
                continue
            
    return timePairs
