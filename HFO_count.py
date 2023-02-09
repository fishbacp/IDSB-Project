import sys

import pandas as pd
import numpy as np

from mne import create_info
from mne.io import RawArray
from mne_hfo import RMSDetector


from tkinter import Toplevel, TOP,BOTH,DoubleVar,Button,Label,Tk, Frame,Label, Button, Entry,Label, Checkbutton,BooleanVar,filedialog,StringVar,messagebox,ttk,DoubleVar,Frame,LEFT,TOP,RIGHT,BOTTOM,BOTH,Menu,Toplevel

from tkinter.ttk import Progressbar

from HFO_tools_5_3  import HFO_detect #(Signal,fs,hp,lp)

from utils.preprocessing_11_28 import find_channel_subset_annotations


################### FUNCTION TO COUNT HFOS BASED UPON SPECIFIED METHOD#################


def hfo_count(root,eeg,X,fs,channels,hp,lp, method='Burnos'):
    hfo_events = pd.DataFrame()
    num_channels=len(channels)

    all_annot=find_channel_subset_annotations(eeg.raw,eeg.fs,channels)
    onsets=all_annot.onset
    durations=all_annot.duration


    progress_window = Toplevel(root)
    progress_window.attributes('-topmost', 'true')
    progress_window.configure(bg='lightgrey')
    progress_window.wm_title("Progress")
    progress_window_label = Label(root, text="")
    progress_window.geometry('300x100')
    progress_window_label.pack()


    win_progress_var=DoubleVar()
    win_progressbar=Progressbar(master=progress_window,variable=win_progress_var,length=num_channels,maximum=1)
    win_progressbar.pack(side=TOP,ipady=5,fill=BOTH,expand=True)
    win_progress_var.set(0)
    win_progressbar.update()
    win_progressbar_label=Label(master=progress_window,text='Window',bg='lightgray')
    win_progressbar_label.pack(side=TOP,pady=5,fill=BOTH, expand=True)


    killed=False
    def _stop():
        global killed
        killed=True
        progress_window.destroy()

    terminate_button=Button(master=progress_window, text="Cancel",command=_stop)
    terminate_button.pack(side=TOP,pady=5,fill=BOTH, expand=True)

    win_count=0
    win_progress_var.set(0)


    if method=='rms':

        for i in range(num_channels):

            if killed:
                break

            win_progress_var.set((win_count+0)/num_channels)
            win_progressbar_label.config(text='Channels: '+str(round(win_count/num_channels*100))+'%')
            win_progressbar.update()
            win_count+=1


            detector = RMSDetector(sfreq=eeg.fs/2,filter_band=(80, .9*eeg.fs/2))
            raw_temp=eeg.raw_orig.copy()
            raw_temp.crop(eeg.start_time,eeg.end_time)
            raw_temp.pick_channels([channels[i]])
            detector.fit(X=raw_temp)
            results=detector.chs_hfos_list[0]
            results=[(r[0],r[1]) for r in results] # a list of ordered pairs with start, end hfo times in sec


            # find starting time of each hfo
            results_starts=[results[i][0] for i in range(len(results))]

            # determine indices of above hfos that start inside an artifact interval
            bad_result_indices=[]
            for k in range(len(results)):
                bad_indices=[j for j in range(len(onsets)) if  onsets[j]<=results_starts[k] and results_starts[k]<=onsets[j]+durations[j]]
                if len(bad_indices)>=1:
                    bad_result_indices.append(k)
            # remove from results those whose indices correspond to artifact intervals
            results=[results[k] for k in range(len(results)) if k not in set(bad_result_indices)]


            df_rms=pd.DataFrame.from_records(results, columns=['start', 'stop'])
            df_rms['start']=df_rms['start']/1
            df_rms['stop']=df_rms['stop']/1
            df_rms['channel']= pd.Series([channels[i] for j in range(df_rms.shape[0])])
            hfo_events=hfo_events.append(df_rms,ignore_index=True)


    if method=='Burnos':

        for i in range(num_channels):

            if killed:
                break

            win_progress_var.set((win_count+0)/num_channels)
            win_progressbar_label.config(text='Channels: '+str(round(win_count/num_channels*100))+'%')
            win_progressbar.update()
            win_count+=1


            df_burnos=HFO_detect(X[:,i],fs,hp,lp)
            df_burnos=df_burnos[df_burnos['start']>0]
            df_burnos=df_burnos[['start','stop']]

            # results above are in seconds
            results=list(df_burnos.itertuples(index=False, name=None))

            # find starting time of each hfo
            results_starts=[results[i][0] for i in range(len(results))]

            # determine indices of above hfos that start inside an annotation
            bad_result_indices=[]
            for k in range(len(results)):
                bad_indices=[j for j in range(len(onsets)) if  onsets[j]<=results_starts[k] and results_starts[k]<=onsets[j]+durations[j]]
                if len(bad_indices)>=1:
                    bad_result_indices.append(k)
            # remove from results those whose indices correspond to artifact intervals
            results=[results[k] for k in range(len(results)) if k not in set(bad_result_indices)]

            df_burnos=pd.DataFrame.from_records(results, columns=['start', 'stop'])
            df_burnos['channel']= pd.Series([channels[i] for j in range(df_burnos.shape[0])])
            hfo_events=hfo_events.append(df_burnos,ignore_index=True)


    counts=hfo_events['channel'].value_counts()
    counts=list(counts.reindex(channels).fillna(0).astype(int))


    progress_window.destroy()

    return counts
