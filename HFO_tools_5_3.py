import warnings
warnings.filterwarnings('ignore')

import mne as mn

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from numpy import random

from tkinter import messagebox

from mne.time_frequency import tfr_array_stockwell as stockwell

from utils.preprocessing_11_28 import find_channel_subset_annotations
from utils.general import plot_selected


################# STOCKWELL TRANSFORM ##################################

def stockwell_transform(signal,fs,fmin,fmax):
    signal_epoch=np.expand_dims(signal,axis=(0,1))
    S, itc, freqs=stockwell(signal_epoch,fs,fmin=fmin,fmax=fmax)
    S=np.squeeze(S,axis=0)
    return S


from tkinter import Tk, ttk, Frame,Label, Button, Entry,Label, Checkbutton,BooleanVar,filedialog,StringVar,messagebox,ttk,DoubleVar,Frame,LEFT,TOP,RIGHT,BOTTOM,BOTH,Menu,Toplevel,PhotoImage

from tkinter.ttk import Progressbar

from matplotlib.widgets import Slider

from tabulate import tabulate

import scipy.special
from scipy import signal

##################################################################################################
##################################################################################################
################  HILBERT-STOCKWELL METHOD WRITTEN BY CODY DEAN####################################


# -----------MAIN HFO FUNCTION-------------------
#-----------------------------------------------------------

#automatic time-frequency algorithm for detection of HFOs
#This was translated to python from the origianl publication at http://www.plosone.org/article/info%3Adoi%2F10.1371%2Fjournal.pone.0094381
# =========================================================================

#==== IN THIS VERSION, THE SAMPLING FREQUENCY AND BANDPASS FILTER FREQUENCIES ARE PASSED AS ARGUMENTS.
def _func():
    global x
    x=3

def HFO_detect(Signal,fs,hp,lp):

    time_thr = np.ceil(0.006*fs)

    #parameters for filtering
    Fst1 = (hp-10)/(fs/2)
    Fp1 = hp/(fs/2)
    Fp2 = lp/(fs/2)
    Fst2 = (lp+10)/(fs/2)
    Ast1 = 40
    Ap = 0.5
    Ast2 = 40

    #merge IoEs
    maxIntervalToJoin = 0.01*fs #10 ms

    #reject events with less than 6 peaks
    minNumberOscillations = 6
    dFactor = 2

    #Stage 2
    bound_min_peak = 40 #Hz, minimum boundary for the lowest ("deepest") point
    ratio_thr = 0.5 #threshold for ratio
    min_trough = 0.2 #20%

    ############# CHANGED to .9*fs/2
    #limit_fr = 500
    limit_fr=int(.9*fs/2)

    start_fr = 80 #limits for peak frequencies

    # 1. filtering ---------------------------------------------------------------------a
    #see end of notebook for notes on conversion
    wp = [Fp1,Fp2]
    ws = [Fst1,Fst2]
    gpass = Ap
    gstop = Ast1

    #bandpass filter
    [b,a] = signal.iirdesign(wp, ws, gpass, gstop, analog=False, ftype='ellip')
    Signal_filtered = signal.filtfilt(b,a,Signal)


    # 2. envelope ---------------------------------------------------------------------
    env = np.abs(signal.hilbert(Signal_filtered))


    # 3. threshold ---------------------------------------------------------------------

    #==== I'VE EXPERIMENTED A LOT WITH SETTING THE THRESHOLDS. ALL IS FINE FOR THE 30-SECOND FILE,
    ## BUT I'M HAVING PROBLEMS WITH THE TWO-HOUR ONE.

    #THR = 3 * np.std(env) + np.mean(env) #- use this threshold for 30sec file

    #======ONE ATTEMPT FOR TWO-HOUR FILE, WHERE THE THRESHOLD IS CHANNEL DEPENDENT (UNDESIRABLE)

    THR = 3 * np.std(env) + np.mean(env)

    #THR=np.percentile(env,99.5)

    #=====ANOTHER ATTEMPT FOR TWO-HOUR FILE. I ALSO USED THR=18 BUT THAT DIDN'T WORK.

    #THR = 18 #ANOTHER ATTEMPT FOR TWO-HOUR FILE (was based upon pyedflib) yields about 12.6 K HFOs

    #THR=27*10**(-6)

    #======THIRD ATTEMPT FOR TWO-HOUR FILE, BASED ROUGHLY UPON
    #MEAN AND STANDARD DEVIATION OF HILBERT ENVELOPE VALUES OVER ALL CHANNELS. GIVEN THAT THE MEDIAN
    #OVERALL CHANNELS IS OF THE ORDER 10^(-5), THIS RESULTS IN NO HFOS.

    #mean_envelope_all_channels=.00010394
    #std_envelope_all_channels=.00012638
    #THR=mean_envelope_all_channels+2*std_envelope_all_channels





    # 4. Stage 1 - detection of EoIs ---------------------------------------------------
    #assign the first and last positions at 0 point
    env[0] = 0
    env[-1] = 0

    pred_env = np.zeros(len(env))
    pred_env[1:len(env)] = env[0:len(env)-1]
    pred_env[0] = pred_env[1]

    if np.size(pred_env,0) != np.size(env,0):
        pred_env = pred_env.T

    #np.where is python equivalent to matlab find(), returns index where conditions are true
    #for some reason returns array in one cell, explore later, but if we take the first element that gives us what we want
    t1 = np.where((pred_env < (THR/2)) & (env >= (THR/2)))[0] #find zero crossings rising
    t2 = np.where((pred_env > (THR/2)) & (env <= (THR/2)))[0] #find zero crossings falling

    trig = np.where((pred_env < THR) & (env >= THR))[0] #check if envelope crosses the THR level rising


    trig_end = np.where((pred_env >= THR) & (env < THR))[0] #check if envelope crosses the THR level falling

    Detections = pd.DataFrame(0.0, index=range(len(trig)), columns=['start','peak','stop','peakAmplitude'])

    nDetectionCounter = 0


    # check every trigger point, where envelope crosses the threshold,
    # find start and end points (t1 and t2), t2-t1 = duration of event;
    # start and end points defined as the envelope crosses half of the
    # threshold for each EoIs

    for i in np.arange(0,len(trig)):

        if ((trig_end[i] - trig[i]) >= time_thr):
            nDetectionCounter = nDetectionCounter + 1
            #not sure why it is buried in layers of arrays, add the index at the end to get out int
            k = np.where((t1 <= trig[i]) & (t2 >= trig[i]))[0][0] #find the starting and end points of envelope

            #check if it does not start before 0 moment
            if t1[k] > 0:
                Detections['start'][nDetectionCounter-1] = t1[k]
            else:
                Detections['start'][nDetectionCounter-1] = 1

            #check if it does not end after last moment
            if t2[k] <= len(env):
                Detections['stop'][nDetectionCounter-1] = t2[k]
            else:
                Detections['stop'][nDetectionCounter-1] = len(env)

            #calculate the max and where it occurs
            peakAmplitude = np.max(env[t1[k]:t2[k]])
            ind_peak = np.argmax(env[t1[k]:t2[k]])

            Detections['peak'][nDetectionCounter-1] = (ind_peak + t1[k])
            Detections['peakAmplitude'][nDetectionCounter-1] = peakAmplitude



    if (nDetectionCounter > 0):

        Detections = Detections[Detections['start'] > 0]

        joinedDetections = joinDetections(Detections,trig,fs)

        checkedOscillations = checkOscillations(joinedDetections, Signal_filtered)
        checkedOscillations = checkedOscillations[checkedOscillations['start'] > 0]

        PSvalidated = PS_validation_all(checkedOscillations, Signal, env,fs)


    else:
        PSvalidated = Detections


    return PSvalidated;


# =========================================================================
# Any EoI that are close to each other and almost indistinguishable, are merged into one EoI
def joinDetections(Detections,trig,fs):

    warnings.filterwarnings("ignore")
    #Merge EoIs with inter-event-interval less than 10 ms into one EoI
    nOrigDetections = len(Detections)
    #merge IoEs
    maxIntervalToJoin = 0.01*fs #10 ms

    #fill result with first detection
    joinedDetections = pd.DataFrame(0.0, index=range(len(trig)), columns=['start','peak','stop','peakAmplitude'])
    joinedDetections['start'][0] = Detections['start'][0]
    joinedDetections['stop'][0] = Detections['stop'][0]
    joinedDetections['peak'][0] = Detections['peak'][0]
    joinedDetections['peakAmplitude'][0] = Detections['peakAmplitude'][0]
    nDetectionCounter = 0

    for n in np.arange(1,nOrigDetections):

        #join detection
        if (Detections['start'][n] > joinedDetections['start'][nDetectionCounter]):
            nDiff = Detections['start'][n] - joinedDetections['stop'][nDetectionCounter]

            if (nDiff < maxIntervalToJoin):
                joinedDetections['stop'][nDetectionCounter] = Detections['stop'][n]

                if (joinedDetections['peakAmplitude'][nDetectionCounter] < Detections['peakAmplitude'][n]):
                    joinedDetections['peakAmplitude'][nDetectionCounter] = Detections['peakAmplitude'][n]
                    joinedDetections['peak'][nDetectionCounter] = Detections['peak'][n]

            else:
                #initialize struct
                nDetectionCounter = nDetectionCounter + 1
                joinedDetections['start'][nDetectionCounter] = Detections['start'][n]
                joinedDetections['stop'][nDetectionCounter] = Detections['stop'][n]
                joinedDetections['peak'][nDetectionCounter] = Detections['peak'][n]
                joinedDetections['peakAmplitude'][nDetectionCounter] = Detections['peakAmplitude'][n]

    #clear out zero rows, probably not efficient code, just trying to get everything working at this point
    joinedDetections = joinedDetections.replace(0,np.nan).dropna()
    #should go back and figure out why these go to floats
    joinedDetections['start'] = joinedDetections['start'].astype(int)
    joinedDetections['stop'] = joinedDetections['stop'].astype(int)
    joinedDetections['peak'] = joinedDetections['peak'].astype(int)


    return joinedDetections;


# =========================================================================
# HFO needs to have a sufficient number of oscillations - this funtion verifies and cleans out ones that don't
def checkOscillations(Detections, Signal):


    # Reject events not having a minimum of 6 peaks above 2 SD
    # ---------------------------------------------------------------------
    # set parameters
    #reject events with less than 6 peaks
    minNumberOscillations = 3
    dFactor = 2
    nDetectionCounter = -1
    AbsoluteMean = np.mean(np.abs(Signal))
    AbsoluteStd = np.std(np.abs(Signal))
    checkedOscillations = pd.DataFrame(0.0, index=range(len(Detections)), columns=['start','stop','peak',
                                                                         'peakHFOFrequency','troughFrequency','peakLowFrequency','peakAmplitude'])
    for n in np.arange(len(Detections)):
        #get EEG for interval
        #add 1 since python indexes to 1 before last value
        intervalEEG = Signal[Detections['start'][n]:Detections['stop'][n]+1]
        # compute abs values for oscillation interval
        absEEG = np.abs(intervalEEG)
        #look for zeros
        zeroVec = np.where(np.multiply(intervalEEG[0:-1],intervalEEG[1:])<0)[0]
        nZeros = np.size(zeroVec)

        nMaxCounter = 0;

        if (nZeros > 0):
            #look for maxima with sufficient amplitude between zeros; insert start/stop/peak into empty data frame
            #checked oscillations
            for ii in np.arange(nZeros-1):

                lStart = zeroVec[ii]
                lEnd = zeroVec[ii+1]
                dMax = np.max(absEEG[lStart:lEnd])

                if (dMax > AbsoluteMean + dFactor * AbsoluteStd):

            #========ORIGINALLY ABOVE LINE HAD dFactor plus AbsoluteStd and not dFactor times AbsoluteStd. I MADE THE
            #CHANGE AND THINGS WORKED FINE.

                    nMaxCounter = nMaxCounter + 1

        if (nMaxCounter >= minNumberOscillations):
            nDetectionCounter = nDetectionCounter + 1
            checkedOscillations['start'][nDetectionCounter] = Detections['start'][n]
            checkedOscillations['stop'][nDetectionCounter] = Detections['stop'][n]
            checkedOscillations['peak'][nDetectionCounter] = Detections['peak'][n]
            checkedOscillations['peakHFOFrequency'][nDetectionCounter] = 0
            checkedOscillations['troughFrequency'][nDetectionCounter] = 0
            checkedOscillations['peakLowFrequency'][nDetectionCounter] = 0
            checkedOscillations['peakAmplitude'][nDetectionCounter] = Detections['peakAmplitude'][n]


    if (nDetectionCounter < 0):
        checkedOscillations['start'][0] = -1
        checkedOscillations['stop'][0] = -1
        checkedOscillations['peak'][0] = -1
        checkedOscillations['peakHFOFrequency'][0] = 0
        checkedOscillations['troughFrequency'][0] = 0
        checkedOscillations['peakLowFrequency'][0] = 0
        checkedOscillations['peakAmplitude'][0] = 0



    #checkedOscillations = checkedOscillations.iloc[checkedOscillations['start'].nonzero()[0]]
    #should go back and figure out why these go to floats

    if (nMaxCounter >= minNumberOscillations):
        checkedOscillations['start'] = checkedOscillations['start'].astype(int)
        checkedOscillations['stop'] = checkedOscillations['stop'].astype(int)
        checkedOscillations['peak'] = checkedOscillations['peak'].astype(int)

    #=====INDICATE THE SHAPE OF checkedOscillations if we get to this stage.

    #print('Checked oscillations has size '+str(checkedOscillations.shape))

    return checkedOscillations;

# -----------------------------------------------------------------------------
# Stage 2 - recognition of HFOs among EoIs
# -----------------------------------------------------------------------------
#=========================================================================
def PS_validation_all(Detections, Signal, env,fs):

    # set parameters
    bound_min_peak = 40 #Hz, minimum boundary for the lowest ("deepest") point
    ratio_thr = 0.5 #threshold for ratio
    min_trough = 0.2 #20%

    ############# CHANGED to .9*fs/2
    #limit_fr = 500
    limit_fr=int(.9*fs/2)

    start_fr = 80 #limits for peak frequencies

    THR = 3 * np.std(env) + np.mean(env)

    #THR=np.percentile(env,99.5)

    #=====ATTEMPT FOR TWO-HOUR FILE. I ALSO USED THR=18 BUT THAT DIDN'T WORK.

    #THR =18+ 0*27.19761355 #ANOTHER ATTEMPT FOR TWO-HOUR FILE
    #THR=27
    #THR=27*10**(-6) #based upon 27 used by Cody and 10^6 scale between pyedflib and MNE

    #mean_envelope_all_channels=.00010394
    #std_envelope_all_channels=.00012638
    #THR=mean_envelope_all_channels+2*std_envelope_all_channels

    nDetectionCounter = -1
    PSvalidated = pd.DataFrame(0.0, index=range(len(Detections)), columns=['start','stop','peak',
                                                                              'peakHFOFrequency','troughFrequency',
                                                                              'peakLowFrequency','peakAmplitude'])

    for n in np.arange(len(Detections)):

        if ((Detections['peak'][n] != -1) & ((Detections['stop'][n]-Detections['start'][n]) < fs*1)):
            #find the sec interval where the peak occurs
            det_start = Detections['peak'][n]-Detections['start'][n]
            det_stop  = Detections['stop'][n]-Detections['peak'][n]

            #define 0.5 sec interval where HFOs occur and take for
            #analysis 0.1 sec before + interval (0.5 sec) + 0.4 sec after
            #in total 1 sec around an HFO is analyzed

            if (np.floor(Detections['peak'][n]/(fs/2)) == 0):   #peak occured in first 0.5 sec

                det_peak = Detections['peak'][n]
                intervalST = Signal[0:fs]
                interval_env = env[0:fs]

            elif (np.floor(Detections['peak'][n]/(fs/2)) == len(Signal)/(fs/2)-1):  #peak occured last 0.5 sec

                det_peak = np.mod(Detections['peak'][n], fs)
                intervalST = Signal[(len(Signal)-fs) : len(Signal)]
                interval_env = env[(len(Signal)-fs) : len(Signal)]

            else:                                                     #peak occured in middle of signal

                det_peak = int(np.mod(Detections['peak'][n], (fs/2))+np.floor(0.1*fs))
                t_peak_interval = int(np.floor(Detections['peak'][n] / (fs/2)))
                intervalST = Signal[int(t_peak_interval*fs/2-np.floor(0.1*fs)) : int(t_peak_interval*fs/2+np.ceil(0.9*fs))]
                interval_env = env[int(t_peak_interval*fs/2-np.floor(0.1*fs)) : int(t_peak_interval*fs/2+np.ceil(0.9*fs))]



#  START STOCKWELL TRANSFORM
#*******************
#*******************
#*******************


            #--------------------------------------------------------------------------
            # Python version uses stockwell transform package from github https://github.com/claudiodsf/stockwell.git.  Verify with MATLAB version is within 1-2Hz
            #STSignal = st.st(intervalST, 0, limit_fr)

            # limit_fr=500; fs is sampling frequency

            STSignal=stockwell_transform(intervalST,fs,0,limit_fr)

            #***********************
            #*****ADDED AN UPPER LIMIT TO INDICES SO WE WEREN'T ACCESSING THE
            #ST TRANSFORM OR ENVELOPE OUTSIDE ITS LENGTH
            #*******************
            upper_index=len(interval_env)




            #-----------------------------------------------------------------------------
            # analyze instantaneous power spectra
            true_HFO = 0 # counter for recognized HFOs


            for tcheck in np.arange(np.max([det_peak-det_start,]), np.min([det_peak+det_stop,upper_index])):
                #print('computing frequencies')

                tcheck=int(tcheck)

                #check if the envelope is above half of the peak+threshold
                if (interval_env[tcheck] > (0.5*(Detections['peakAmplitude'][n] + THR))):


                    #for maximum upper start_f frequency

                    maxV = np.max(np.abs(STSignal[start_fr:,tcheck])) #HFO peak
                    maxF = np.argmax(np.abs(STSignal[start_fr:,tcheck]))
                    maxF = maxF + start_fr+1

                    #search for minimum before found maximum
                    minV = np.min(np.abs(STSignal[bound_min_peak:maxF, tcheck])) #the trough
                    minF = np.argmin(np.abs(STSignal[bound_min_peak:maxF, tcheck])) #the trough
                    minF = minF+bound_min_peak+1

                    #print(tcheck,minF,minV,maxF,maxV)

                    #check for sufficient difference
                    #set signal to look through to x so we can pull peaks out if it
                    x = STSignal[0:minF, tcheck]
                    if np.size(np.abs(x)) == 0:
                        peaks=[]
                    else:
                        peak_loc = signal.find_peaks(np.abs(x))[0] #this find the location of the peak, to find the actual value you need x[peaks]
                        peaks = x[peak_loc]

                    if np.size(peaks) == 0:
                        fpeaks=np.floor(minF/2)
                        peaks = np.abs(STSignal[fpeaks, tcheck])
                        ratio_HFO=0
                        ratio_LowFr=0
                    else:
                        ratio_HFO = float(10*np.log10(maxV) - 10*np.log10(minV)) #ratio between HFO peak and the trough
                        ratio_LowFr = float(10*np.log10(peaks[-1]) - 10*np.log10(minV)) #ratio between Low Frequency peak and the trough



                    #check the difference and check for sufficient trough
                    if ((upper_index>0)&(ratio_HFO>(ratio_thr*ratio_LowFr))&(ratio_HFO>(min_trough*10*np.log10(maxV)))&(maxF<500)):
                        true_HFO=true_HFO+0
                    else:
                        true_HFO=true_HFO+1

            if ((upper_index > 0) & (true_HFO==0)): #all conditions are satisfied
                #search for peak
                tcheck = det_peak


                tcheck=int(tcheck)

                maxF = np.argmax(np.abs(STSignal[start_fr:,tcheck]))
                maxF = maxF + start_fr+1 #try to understand why change -1 to +1, maybe python indexing?

                #search for minimum before found maximum
                minF = np.argmin(np.abs(STSignal[bound_min_peak:maxF, tcheck])) #the trough
                minF = minF+bound_min_peak+1

                #check for sufficient difference
                #fpeaks=[]
                #fpeaks.append(signal.find_peaks_cwt(np.abs(STSignal[0:minF, tcheck]),np.arange(1, 2))[0])#low frequency peak
                fpeaks = signal.find_peaks(np.abs(STSignal[0:minF, tcheck]))[0]




# END OF STOCKWELL PORTION
#*******************
#*******************
#*******************
#*******************

                nDetectionCounter = nDetectionCounter + 1

                #times are translates to seconds
                PSvalidated['start'][nDetectionCounter] = Detections['start'][n]/fs
                PSvalidated['stop'][nDetectionCounter] = Detections['stop'][n]/fs
                PSvalidated['peak'][nDetectionCounter] = Detections['peak'][n]/fs
                PSvalidated['peakHFOFrequency'][nDetectionCounter] = maxF
                PSvalidated['troughFrequency'][nDetectionCounter] = minF

                if (np.size(fpeaks) != 0):
                    PSvalidated['peakLowFrequency'][nDetectionCounter] = fpeaks[-1]
                else:
                    PSvalidated['peakLowFrequency'][nDetectionCounter] = 0

                PSvalidated['peakAmplitude'][nDetectionCounter] = Detections['peakAmplitude'][n]

        #from tabulate import tabulate
        #print(tabulate(Detections, headers='keys', tablefmt='psql'))
        #print('NUM DETECTIONS: '+str(nDetectionCounter))
        if (nDetectionCounter < 0):
            PSvalidated['start'][0] = -1
            PSvalidated['stop'][0] = -1
            PSvalidated['peak'][0] = -1
            PSvalidated['peakHFOFrequency'][0] = 0
            PSvalidated['troughFrequency'][0] = 0
            PSvalidated['peakLowFrequency'][0] = 0
            PSvalidated['peakAmplitude'][0] = 0

    return PSvalidated;



##################################################################################
##################################################################################
################  END HILBERT-STOCKWELL METHOD #######################################



def HFOs_all_channels(root,eeg,X,channels,hp,lp,fs):
    plot_selected(root,eeg,annotations=False)

    df_combined_results = pd.DataFrame()
    num_channels=len(channels)

    all_annot=find_channel_subset_annotations(eeg.raw,eeg.fs,channels)

    onsets=all_annot.onset
    durations=all_annot.duration

    progress_window = Toplevel(root)
    progress_window.geometry('300x100')
    progress_window.attributes('-topmost', 'true')
    progress_window.configure(bg='lightgrey')
    progress_window.wm_title("Progress")



    progress_var = DoubleVar()
    progressbar = Progressbar(master=progress_window,variable=progress_var, length=num_channels,maximum=1)
    progressbar.pack(side=TOP,ipady=5,fill=BOTH, expand=True)
    progress_var.set(0)
    progressbar.update()
    progressbar_label=Label(master=progress_window,text='Channel',bg='lightgray')
    progressbar_label.pack(side=TOP,pady=5,fill=BOTH, expand=True)

    killed=False
    def _stop():
        global killed
        killed=True
        progress_window.destroy()


    terminate_button=Button(master=progress_window, text="Cancel",command=_stop)
    terminate_button.pack(side=TOP,pady=5,fill=BOTH, expand=True)

#==========NOW LOOP OVER TIME INTERVALS AND CHANNELS TO COMPUTE HFOS, APPENDING df_combined_results AT EACH PASS.
    progress_var.set(0)
    progressbar.update()

    win_count=0


    for i in range(num_channels):
        if killed:
            break
        progress_var.set(i/(num_channels))
        progressbar_label.config(text='Channels: '+str(round(win_count/num_channels*100))+'%')
        #progressbar_label.config(text='Channels: '+str(round(i/(num_channels-1)*100))+'%')
        progressbar.update()
        win_count+=1

        Signal=X[:,i]

        try:
            df_channel=HFO_detect(Signal,fs,hp,lp)
            df_channel = df_channel[df_channel['start'] > 0] # 'start' measured in seconds
            df_channel = df_channel[df_channel['peakHFOFrequency'] <=lp] # peak cannot exceed one-half fs

        except:
            df_channel=pd.DataFrame()
            #progress_window.destroy()
            #messagebox.showerror('Error','Method failed to yield HFOs. Try modifying interval or using RMS method instead')
            #return

        if not df_channel.empty:
            df_channel['start']=np.round(df_channel['start'],2)
            df_channel['stop']=np.round(df_channel['stop'],2)

            df_channel.insert(0,'channel',[channels[i] for j in range(df_channel.shape[0])])
            df_channel.reset_index(inplace=True)

            results_starts=df_channel['start'].tolist()
            # determine indices of above hfos that start inside an annotation
            bad_result_indices=[]
            for k in range(len(results_starts)):
                bad_indices=[j for j in range(len(onsets)) if  onsets[j]<=results_starts[k] and results_starts[k]<=onsets[j]+durations[j]]
                if len(bad_indices)>=1:
                    bad_result_indices.append(k)
            # remove from results those whose indices correspond to artifact intervals
            #results=[results_starts[k] for k in range(len(results_starts)) if k not in set(bad_result_indices)]

            df_channel.drop(bad_result_indices,inplace=True) #,axis=0)
            #df_channel.reset_index()

        df_combined_results=df_combined_results.append(df_channel,ignore_index=True)

    df_combined_results.reset_index(inplace=True)
    #from tabulate import tabulate
    #print(tabulate(df_combined_results, headers='keys', tablefmt='psql'))

    if not df_combined_results.empty:
        counts=df_combined_results['channel'].value_counts()
        counts=list(counts.reindex(channels).fillna(0).astype(int))
    else:
        counts=[0 for i in range(len(channels))]

    progress_window.destroy()

    return df_combined_results,counts


''' EXAMPLE

from utils.animation_tools import scrolling_matrix_viewer
raw=mn.io.read_raw_edf('11.edf', preload=True)
fs=int(raw.info['sfreq'])
data,times=raw[:,:]

data_matrix=data.T
data_matrix=data_matrix[:,0:3]

channels=raw.info["ch_names"][0:3]
num_channels=len(channels)

hp = 80
lp = .9*fs/2

df_combined_results,counts, M_ch_time=HFOs_all_channels(root,eeg,data_matrix,channels,hp,lp,fs)

from tabulate import tabulate
print(tabulate(df_combined_results, headers='keys', tablefmt='psql'))


'''
