import os
import numpy as np
import mne

# mne.viz.set_browser_backend('matplotlib')
mne.viz.set_browser_backend('qt')

##from mne.preprocessing import ICA
##from mne import create_info
##from mne.io import RawArray

import sys
##from PIL import ImageTk, Image
##
##import matplotlib
##import matplotlib as mpl
##from matplotlib import pyplot as plt
##from matplotlib.figure import Figure
##from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure

##import matplotlib.backends.backend_tkagg as tkagg
##import matplotlib.ticker as ticker

##matplotlib.use('Qt5Agg',force=True)
##
##from utils.general import get_data, plot_selected,list_channel_pairs, license_window,program_info_window,start_video_tutorial
##
##from utils.preprocessing_11_28 import remove_line_noise,find_channel_subset_annotations, repair_artifacts
##
##from utils.animation_tools_5_3 import heatplot_barplot_animation_combined,scrolling_configuration_matrix_viewer
##from utils.animation_tools_5_3 import bar_plot_connection_reinforcements,bar_plot_with_slider_rms,bar_plot_with_slider_burnos
##
##from utils.menu_options_2_1 import conn_options, update_conn_options,conn_options_reset
##from tkinter import Tk, Frame,Label, Button, Entry,Label, Checkbutton,BooleanVar,filedialog,StringVar,messagebox,ttk,DoubleVar,Frame,LEFT,TOP,RIGHT,BOTTOM,BOTH,Menu,Toplevel,PhotoImage,Canvas
##from tkinter import font as tkFont
##
##from tkmacosx import Button as Message_Button  ### CHANGE TO JUST BUTTON FOR WINDOWS VERSION; DON'T IMPORT tkmacosx

from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QMessageBox, QComboBox, QHBoxLayout, QFileDialog, QMenu, QAction,QErrorMessage
from PyQt5.QtGui import QFont, QIntValidator, QPainter
from PyQt5.QtCore import Qt

import pyqtgraph as pg
from pyqtgraph import PlotWidget, plot, AxisItem

### MAIN HFO FUNCTIONS
##from HFO_tools_5_3 import HFOs_all_channels
##from HFO_count import hfo_count
##
##### MAIN CONNECTIVITY FUNCTIONS
from functional_connectivity_tools import coherence_at_mark ,functional_connectivities #,like_brain_states,clustering_coefficient_matrix
##from effective_connectivity_tools import effective_connectivity,conn_count,swadtf_at_mark

################## FUNCTION TO CLOSE CURRENTLY OPEN WIDGETS
##def resource_path(relative_path):
##    try:
##        base_path = sys._MEIPASS
##    except Exception:
##        base_path = os.path.abspath(".")
##
##    return os.path.join(base_path, relative_path)
##
def all_children (window) :
    # _list = window.winfo_children()
    # for item in _list :
    #     if item.winfo_children() :
    #         _list.extend(item.winfo_children())
    _list = window.children()
    for item in _list:
        if item.children():
            _list.extend(item.children())
    return _list


############### MAIN ROOT WINDOW

# root = Tk()
# myfont = tkFont.Font(family='Helvetica', size=14, weight=tkFont.BOLD)
# root.wm_title("EEG Connectivity Explorer")


# root.geometry('1500x1000')
# root.configure(bg='white')
# style = ttk.Style(root)
#style.theme_use('clam')

app = QApplication([])
# root = QWidget()
root = QtWidgets.QMainWindow()
root.setWindowTitle("EEG Connectivity Explorer")
root.resize(500, 500)

############ FIGURE BACKGROUND AT STARTUP

########### IMAGE ON ROOT WINDOW AT STARTUP
##def startup_background(root):
##
##    if getattr(sys, 'frozen', False):
##        # Frozen application
##        img_path = os.path.join(sys._MEIPASS, 'background.png')
##        config_path=os.path.join(sys._MEIPASS,'config.txt')
##    else:
##        # (Mac version) Original script
##        img_path = 'background.png'
##        config_path='config.txt'


    # Create a Canvas
    # canvas = Canvas(root, width=700, height=3500)

    # canvas.pack(fill=BOTH, expand=True)

    # label = QtWidgets.QLabel(parent = root)
    # canvas = QtGui.QPixmap(700, 3500)
    # canvas.fill(Qt.white)
    # label.setPixmap(canvas)

    # Function to resize the window
##def resize_image(e):
##
##        global image, resized , image2
##        # open image to resize it
##
##        if getattr(sys, 'frozen', False):
##            # Frozen application
##            img_path = os.path.join(sys._MEIPASS, 'background.png')
##            config_path=os.path.join(sys._MEIPASS,'config.txt')
##        else:
##            # (Mac version) Original script
##            img_path = 'background.png'
##            config_path='config.txt'
##
##
##        image = Image.open(img_path)
##
##        #print([e.width,e.height])
##
##        # resize the image with width and height of root
##        resized = image.resize((e.width, e.height), Image.ANTIALIAS)
##
##        image2 = ImageTk.PhotoImage(resized)
##        canvas.create_image(0, 0, image=image2, anchor='nw')
##

    # Bind the function to configure the parent window
    # root.bind("<Configure>", resize_image)

##def show_license(root):

##    if getattr(sys, 'frozen', False):
##        # Frozen application
##        config_path=os.path.join(sys._MEIPASS,'config.txt')
##    else:
##        # (Mac version) Original script
##        config_path='config.txt'
##    license_window(config_path)
##
##startup_background(root)
##show_license(root)


############## EEG CLASS ################################
############# A global variable with numerous attributes and methods
###################################################

class EEG:
    def __init__(self,file):

        raw=mne.io.read_raw(file,preload=True) # raw data as loaded from .edf file
        self.raw=raw

        raw_orig=raw.copy()
        self.raw_orig=raw_orig  # backup copy of original data so its doesn't have to be re-loaded

        self.fs=int(self.raw.info['sfreq'])   # retrieve sampling frequency from recording
        self.channels=raw.info["ch_names"]  # retrieve channel names as a list of strings, e.g. ['Fp1','Fp2',...]

        # Plot duration and start in seconds; Updated in plot_selected as
        # eeg.start_time = eeg.fig.mne.t_start - eeg.fig.mne.first_time
        # eeg.end_time = eeg.start_time + eeg.fig.mne.duration. See "fig" below

        self.start_time=0
        self.end_time=30

        self.plot_duration=30
        self.plot_start=0

        self.root=None

        # fig is the current figure on display at any instant. Currently, I can extract the start time, end time, marked segment from
        # fig. I was able to do this by digging deep into the MNE code.
        fig=self.raw.plot(show=False,block=True,duration=self.plot_duration,bgcolor='w',color='black', bad_color='gray')
        self.fig=fig

        # scalings of channels; currently controlled by +/- keys 
        self.scalings=.00005 

        # current "bad", unselected channels as they appear in eeg.fig
        self.bads=self.fig.mne.info['bads']




        # current location of marked segment
        self.segment_loc=0




        self.plot_options_window=None
        self.conn_options_window=None

        self.canvas_frame=None

        self.source_frame=None
        self.ica=None
        self.eog_indices=None

        self.configuration_matrix=None

        self.annotations=None

        self.artifact_starts=[]
        self.artifact_stops=[]

        self.plot_options_changed=False
        self.segment_options_changed=False
        self.conn_options_changed=False
        self.num_win=1
        self.win_size=30
        self.windowed=False
        self.include_channel_time=False

        # set a few default parameters for analysis.
        self.conn_win_value=.5
        self.fmin_value=5
        self.fmax_value=30
        self.conns_percentile=.9


############ plot method associated with EEG class ####################

    def plot_canvas(self,annotations=False):

##        if self.source_frame in set(all_children(root)):
##            self.source_frame.destroy()

        # annotations are marked segments in the recording where signal artifacts are repaired. LOW-PRIORITY
##        if annotations==False:
##            self.raw.set_annotations(None)
##        else:
##            raw=self.raw
##            channels=self.raw.info['ch_names']
##            fs=self.raw.info['sfreq']
##            all_annot=find_channel_subset_annotations(raw,fs,channels)
##            self.raw.set_annotations(all_annot)
##            self.annotations=all_annot


        mne.set_config('MNE_BROWSE_RAW_SIZE','16,8')

        # fig stores current figure showing in window.
        self.fig=(self.raw).plot(show=False,block=True,duration=self.plot_duration,start=self.plot_start,bgcolor='white',color='b', bad_color='gray',scalings=dict({'eeg':.000050}),n_channels=10,precompute=True,time_format='clock')


        #########  THIS IS WHERE THE RAW DATA IS PLOTTED AND PLACED ON A TKINTER FIGURE CANVAS. 


        # self.canvas_frame=Frame(root)
        # self.canvas_frame.pack(side=TOP,expand=True)
        # self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        # self.canvas.draw()
        # self.canvas.get_tk_widget().pack(side=TOP,fill=BOTH,expand=1)

        self.canvas_frame = QtWidgets.QFrame(parent = root)
        root.setCentralWidget(self.canvas_frame)
        # self.label = QtWidgets.QLabel(parent = self.canvas_frame)
        # self.canvas = QtGui.QPixmap(700, 3500)
        # self.canvas.fill(Qt.white)
        # self.painter = QPainter(self.canvas)
        # self.label.setPixmap(self.canvas)
        # self.canvas = FigureCanvasQTAgg(self.fig)
        # self.canvas.setParent(self.canvas_frame)
        # self.canvas.draw()
        self.fig.setParent(self.canvas_frame)



#############  PLOT ICA SOURCES #################   USED FOR SIGNAL REPAIR--LOW PRIORITY
##    def plot_sources(self,show=True):
##        if self.fig.mne.info['bads']==self.fig.mne.info['ch_names']:
##            messagebox.showerror("Error","No channels selected!!")
##            return
##        if self.canvas_frame in set(all_children(root)):
##            self.canvas_frame.destroy()  # Important to close the eeg.canvas_frame before doing everything below.
##        bads=self.fig.mne.info['bads']
##        channels=self.fig.mne.info['ch_names']
##        to_drop=list(set(channels).intersection(set(bads)))
##        self.raw.drop_channels(to_drop)
##        channels=self.raw.info['ch_names']
##
##        raw_temp=self.raw.copy().crop(tmin=self.start_time,tmax=self.end_time)
##        raw_filt = raw_temp.load_data().filter(l_freq=1., h_freq=None)
##        self.ica = ICA(method='fastica',n_components=len(channels))
##        self.ica.fit(raw_filt,picks='all',decim=1)
##
##        eog_idx, eog_scores=self.ica.find_bads_eog(raw_filt,ch_name=channels,measure='correlation',threshold=.9,reject_by_annotation=False)
##        ecg_idx, ecg_scores=self.ica.find_bads_ecg(raw_filt,ch_name=channels[0],measure='correlation',threshold=.9,reject_by_annotation=False)
##        idx=set(eog_idx).union(set(ecg_idx))
##        excludes_idx=sorted(idx)
##        self.exclude_indices=excludes_idx
##
##        if show:
##            fig=self.ica.plot_sources(raw_filt, show=False,block=True,show_scrollbars=True,title='Latent Sources in Data')
##            self.source_frame=Frame(root,background="white")
##            self.source_frame.pack(side=TOP,expand=True)
##
##            close_button = Button(self.source_frame, text ="Repair Original", fg='black',bg='white',borderwidth=0,command = _repair)
##            close_button.pack(side=TOP,expand=True)
##
##            repair_button = Button(self.source_frame, text ="Close", fg='black',bg='white',borderwidth=0,command = _replot)
##            repair_button.pack(side=TOP,expand=True)
##
##            source_canvas = FigureCanvasTkAgg(fig, master=self.source_frame)
##            source_canvas.draw()
##            source_canvas.get_tk_widget().pack(side=TOP,fill=BOTH,expand=1)
##
##
##### replot using only selected channels
##def _replot():
##    eeg.plot_canvas()

############ OPEN FILE ####################

def _open():
    global eeg
    widget_list = all_children(root)

    # for item in widget_list:
        # item.pack_forget()
        # item.hide()

    # root.filename = filedialog.askopenfilename(title = "Select file",filetypes = (("EDF files","*.edf"),("MNE files","*.fif")))#,("all files","*.*")))
    options = QFileDialog.Options()
    options |= QFileDialog.DontUseNativeDialog
    fileName, _ = QFileDialog.getOpenFileName(root, "Select File", "", "edf (*.edf)", options=options)

##    if fileName=='':
##        startup_background(root)
##        return

    eeg=EEG(fileName)
    
    eeg.raw.info['bads']=eeg.raw.info["ch_names"]
    
    eeg.plot_canvas() 

    enable_menu_pulldowns()


########### PLOT SELECTED CHANNELS CALL-BACK FUNCTION#######################

def plot_selected(eeg):
    if set(eeg.fig.mne.info['bads'])==set(eeg.channels):
        error_dialog = QtWidgets.QErrorMessage(root)
        error_dialog.showMessage('No Channels Selected! Click on desired channels')
        #messagebox.showerror("Error","No channels selected!!")
        return
    bads=eeg.fig.mne.info['bads']
    channels=eeg.fig.mne.info["ch_names"]
    channels=[ch for ch in channels if ch in set(channels)-set(bads)]
    raw_temp=eeg.raw_orig.copy()
    eeg.raw=raw_temp.pick_channels(channels)
    eeg.plot_canvas()
    

############  SELECT ALL CHANNELS CALL-BACK FUNCTION
def _select_all_channels(eeg):
    raw_temp=eeg.raw_orig.copy()
    eeg.raw=raw_temp
    eeg.raw.info['bads']=[]
    eeg.plot_canvas()

####### REFRESH PLOT CALL-BACK FUNCTION
def _refresh(eeg): 
    raw_fresh= eeg.raw_orig.copy()

    raw_fresh.info['bads']=raw_fresh.info['ch_names']
    eeg.raw=raw_fresh
    eeg.plot_start=0
    eeg.plot_duration=30
    eeg.plot_canvas()

######## QUIT CALL-BACK FUNCTION
##def _quit():
##
##    widget_list = all_children(root)
##    for item in widget_list:
##        item.pack_forget()
##
##    root.quit()
##    root.destroy()
##

 #root.protocol("WM_DELETE_WINDOW", enable_menu_pulldowns)


############ REMOVE LINE NOISE CALL-BACK: LOW-PRIORITY #######################

##def _remove_line_noise():
##    remove_line_noise(eeg)


########## MARK BLINK AND MUSCLE Artifacts
'''
def _mark_artifacts():
    # this calls plot_selected; eeg.raw.info['ch_names'] is updated; and only those channels are plotted by eeg.canvas()
    # an error message results here if no channels are selected; artifacts are marked only for selected channels
    plot_selected(root,eeg,annotations=True)
'''

###########  PLOT ICA SOURCES AND REPAIR: LOW PRIORITY ##################

##def _plot_sources():
##    eeg.plot_sources(show=True)
##    return

########################################################################
########################################################################
############## RECONSTRUCT REPAIRED SIGNAL USING CHOSEN ARTIFACTS
###### Create raw_repaired by using ica.apply below on
######### raw_temp=eeg.raw.copy().crop(tmin=eeg.start_time,tmax=eeg.end_time). Then append this in front and back
####### by raw_beginning=eeg.raw.copy().crop(tmin=0,tmax=eeg.start_time) and
####### raw_ending=eeg.raw.copy().crop(tmin=eeg.end_time)

# LOW PRIORITY
##def _repair():
##    eeg.source_frame.destroy()
##
##    raw_temp=eeg.raw.copy().crop(tmin=eeg.start_time,tmax=eeg.end_time)
##    raw_repaired=eeg.ica.apply(raw_temp,exclude=eeg.ica.exclude)
##
##    raw_beginning=eeg.raw.copy().crop(tmin=0,tmax=eeg.start_time)
##    raw_ending=eeg.raw.copy().crop(tmin=eeg.end_time)
##    raw_beginning.append([raw_repaired,raw_ending])
##    eeg.raw=raw_beginning
##
##    eeg.plot_canvas()
##    return



######## UPDATE connectivity Options UNDER Connectivity Menu ############################
#######################################################################################

##### UPDATE CONN OPTIONS #########
##def _update_conn_options():
##    update_conn_options(root,eeg)
##    enable_menu_pulldowns()

########## RESET CONN OPTIONS WINDOW

##def _reset_conn_options():
##    conn_options_reset(root,eeg)
##    enable_menu_pulldowns()


########## CLOSE CONNECTIVITY OPTIONS WINDOW

##def _close_conn_options():
##     ##### ENABLE PULL DOWN MENU
##     eeg.conn_options_window.destroy()
##     enable_menu_pulldowns()

######## SET UP CONNECTIVITY MENU
##def _conn_options():
##    disable_menu_pulldowns()
##    eeg.conn_options_window = Toplevel(bg="lightgray")
##    eeg.conn_options_window.attributes('-topmost','true')
##    eeg.conn_options_window.geometry('400x280')
##    eeg.conn_options_window.wm_title("Connectivity Options")
##    eeg.conn_options_window.protocol("WM_DELETE_WINDOW", _close_conn_options)
##
##    conn_options(root,eeg)
##
############ Update and close Connectivity options

##    update_conn_options_box = ttk.Button(master=eeg.conn_options_window, text="Update",command=_update_conn_options)
##    update_conn_options_box.pack()
##
##    reset_conn_options_box = ttk.Button(master=eeg.conn_options_window, text="Reset",command=_reset_conn_options)
##    reset_conn_options_box.pack()


############### CONNECTIVITY COMPUTATIONS ############################
#######################################################################

### Coherence heatmap using a window of width conn_win_value centered at selected green segment in raw plot

def _coh_mark(eeg):    
    if set(eeg.fig.mne.info['bads'])==set(eeg.channels) or len(eeg.fig.mne.info['ch_names'])-len(eeg.fig.mne.info['bads'])<2:
        error_dialog = QtWidgets.QErrorMessage(root)
        error_dialog.showMessage('Not enough channels! Click on at least two')
        return
    if eeg.fig.mne.vline==None: 
        error_dialog = QtWidgets.QErrorMessage(root)
        error_dialog.showMessage('No time selected! Click at desired time value')
        return
    eeg.segment_loc=eeg.fig.mne.vline.value()
    S=coherence_at_mark(eeg)

    ######  A NEW WIDGET SHOULD APPEAR WITH A HEATMAP FOR ABOVE CHANNELS

def _coh(eeg):
    if set(eeg.fig.mne.info['bads'])==set(eeg.channels) or len(eeg.fig.mne.info['ch_names'])-len(eeg.fig.mne.info['bads'])<2:
        error_dialog = QtWidgets.QErrorMessage(root)
        error_dialog.showMessage('Not enough channels! Click on at least two')
        return
    if eeg.fig.mne.duration<eeg.conn_win_value:
        error_dialog = QtWidgets.QErrorMessage(root)
        error_dialog.showMessage('Selected interval is less than subwindow size from connectivity options! Expand interval!')
        return
    if eeg.fig.mne.duration>120:
        error_dialog = QtWidgets.QErrorMessage(root)
        error_dialog.showMessage('Connectivity animation limited to 120 seconds. Decrease interval length!')
        return    
    S_list,bar_list= functional_connectivities(eeg)    
    #### A NEW WIDGET SHOULD APPEAR. IT WILL HAVE A SLIDER BUTTON AND CREATE AN ANIMATION. EACH FRAME
    #### WILL DISPLAY THE HEATPLOT FOR A MATRIX FROM S_list AND A BAR PLOT FROM bar_list
   


### Sequence of coherence matrices, where each is averaged over a frequency band. PSD for each channel also computed.
##def _coh():
##    eeg.start_time = eeg.fig.mne.t_start - eeg.fig.mne.first_time
##    eeg.end_time = eeg.start_time + eeg.fig.mne.duration
##
##    if  set(eeg.fig.mne.info['bads'])==set(eeg.fig.mne.info['ch_names']) or len(eeg.fig.mne.info['ch_names'])-len(eeg.fig.mne.info['bads'])<2:
##        message_box(eeg.root, 'Error: At least two channels must be selected!',type='error')
##        return
##    if eeg.end_time-eeg.start_time<eeg.conn_win_value:
##        message_box(eeg.root, 'Error: Selected interval is less than subwindow size from connectivity options! Expand interval!',type='error')
##        return
##    if eeg.end_time-eeg.start_time>120:
##        message_box(eeg.root,'Error: Connectivity animation limited to 120 seconds. Decrease interval length!',type='error')
##        return
##
##
##    plot_selected(root,eeg,annotations=False)
##
##    X,fs,channels=get_data(eeg,eeg.raw,eeg.fig,eeg.start_time,eeg.end_time)
##
##    data=X.T
##    start_time=eeg.start_time
##    conn_win_value=eeg.conn_win_value # max(5/(eeg.fmin_value*2),eeg.conn_win_value)
##    M_list,config_matrix,bar_list=functional_connectivities(root,data,channels,fs,conn_win_value,method='coh',fmin=eeg.fmin_value,fmax=eeg.fmax_value)
##    bar_list=bar_list*1/np.max(bar_list) #max(max(bar_list))
##
##    bar_list_max=np.max(bar_list)
##
##    heatplot_barplot_animation_combined(eeg,root,channels,M_list,bar_list,bar_list_max,start_time,conn_win_value,xlabel='channel',ylabel='channel',barlabel='Centrality',title='Coherence and Centrality Values')
##
##
##    eeg.configuration_matrix=config_matrix



########  cluster similar coherence matrices by time windows
##def _coh_time_communities():
##    if  set(eeg.fig.mne.info['bads'])==set(eeg.fig.mne.info["ch_names"]):
##        message_box(eeg.root,'Error: No channels selected!',type='error')
##        #messagebox.showerror('Error','No channels selected!')
##        return
##
##    plot_selected(root,eeg,annotations=False)
##    X,fs,channels=get_data(eeg,eeg.raw,eeg.fig,eeg.start_time,eeg.end_time)
##    data=X.T
##    conn_win_value=eeg.conn_win_value
##    M_list,config_matrix,bar_list=functional_connectivities(root,data,channels,fs,conn_win_value,method='coh',fmin=eeg.fmin_value,fmax=eeg.fmax_value)
##    like_brain_states(root,eeg,config_matrix,conn_win_value,method='kmeans')
    # options are 'correlation' or 'kmeans' or 'affinity_prop'


######## time sequence of mutual information and transfer entropy matrices, one for each time subwindow
##def _mutual_info_and_transfer_entropy():
##    if  set(eeg.fig.mne.info['bads'])==set(eeg.fig.mne.info["ch_names"]):
##        message_box(eeg.root,'Error: No channels selected!',type='error')
##        return
##    plot_selected(root,eeg,annotations=False)
##    X,fs,channels=get_data(eeg,eeg.raw,eeg.fig,eeg.start_time,eeg.end_time)
##    conn_win_value=eeg.conn_win_value 
##    MI_list,TE_list=mutual_info_and_transfer_entropy(root,eeg,conn_win_value)
##
##    spos,ax,fig=scrolling_stacked_matrix_sequence_viewer(root,MI_list,TE_list,channels,conn_win_value)


######### Spectrum weighted adapted direct, directed transfer function, one for each time subwindow, along with eigenvector centrality
##def _swadtf():
##    eeg.start_time = eeg.fig.mne.t_start - eeg.fig.mne.first_time
##    eeg.end_time = eeg.start_time + eeg.fig.mne.duration
##
##    if  set(eeg.fig.mne.info['bads'])==set(eeg.fig.mne.info["ch_names"]) or len(eeg.fig.mne.info['ch_names'])-len(eeg.fig.mne.info['bads'])<2:
##        message_box(eeg.root,'Error: At least two channels must be selected!',type='error')
##        return
##
##    if eeg.end_time-eeg.start_time<eeg.conn_win_value:
##        message_box(eeg.root,'Error: Selected interval is less than subwindow size from connectivity options! Expand interval!',type='error')
##        return
##
##    if eeg.end_time-eeg.start_time>120:
##        message_box(eeg.root,'Error: Connectivity animation limited to 120 seconds. Decrease interval length!',type='error')
##        return
##
##
##    plot_selected(root,eeg,annotations=False)
##    X,fs,channels=get_data(eeg,eeg.raw,eeg.fig,eeg.start_time,eeg.end_time)
##    data=X.T
##    start_time=eeg.start_time
##
##    conn_win_value= eeg.conn_win_value #max(5/(eeg.fmin_value*2),eeg.conn_win_value)
##
##    M_list,bar_list=effective_connectivity(root,data,channels,fs,conn_win_value,type='swdtf',fmin=eeg.fmin_value,fmax=eeg.fmax_value)
##    bar_list_max=1
##
##    heatplot_barplot_animation_combined(eeg,root,channels,M_list,bar_list,bar_list_max,start_time,conn_win_value,xlabel='Sending',ylabel='Receiving',barlabel='Driving Score',title='Directed Connectivity Values and Driving Scores')
##
##
############  SWADTF at mark #########################################

##def _swadtf_mark():
##    print(eeg.fig.mne.info['bads'])
##    if  set(eeg.fig.mne.info['bads'])==set(eeg.fig.mne.info['ch_names']) or len(eeg.fig.mne.info['ch_names'])-len(eeg.fig.mne.info['bads'])<2:
##        message_box(eeg.root,'Error: At least two channels must be selected!',type='error')
##        return
##    if  eeg.fig.mne.segment_loc==0:
##        message_box(eeg.root,'Error: No segment marked in raw data!',type='error')
##        return
##
##    swadtf_at_mark(eeg)

#########  connection reinforcements for above _swadtf
##def _connection_reinforcements():
##    eeg.start_time = eeg.fig.mne.t_start - eeg.fig.mne.first_time
##    eeg.end_time = eeg.start_time + eeg.fig.mne.duration
##
##    if  set(eeg.fig.mne.info['bads'])==set(eeg.fig.mne.info["ch_names"]) or len(eeg.fig.mne.info['ch_names'])-len(eeg.fig.mne.info['bads'])<2:
##        mmessage_box(eeg.root,'Error: At least two channels must be selected!',type='error')
##        return
##
##    if eeg.end_time-eeg.start_time<eeg.conn_win_value:
##        message_box(eeg.root,'Error: Selected interval is less than subwindow size from connectivity options! Expand interval!',type='error')
##        return
##
##    if eeg.end_time-eeg.start_time>120:
##        message_box(eeg.root,'Error: Connectivity animation limited to 120 seconds. Decrease interval length!',type='error')
##        return
##
##    plot_selected(root,eeg,annotations=False)
##    X,fs,channels=get_data(eeg,eeg.raw,eeg.fig,eeg.start_time,eeg.end_time)
##    data=X.T
##
##    conn_win_value= eeg.conn_win_value #max(5/(eeg.fmin_value*2),eeg.conn_win_value)
##
##    M_list,bar_list=effective_connectivity(root,data,channels,fs,conn_win_value,type='swdtf',fmin=eeg.fmin_value,fmax=eeg.fmax_value)
##    conn_matrix,conns=conn_count(M_list,channels,cutoff=eeg.conns_percentile)
##
##
##    bar_plot_connection_reinforcements(conns,channels,num_bars=8,title='Connection Reinforcements',sort=True)
##

############### HFO COMPUTATION FUNCTIONS ############################
#######################################################################


################ RMS METHOD ######################################
#####  Use eeg.plot_start and eeg.plot_duration below? Then cut out eeg.start_time,eeg.end_time
##def _hfos():
##
##    if  set(eeg.fig.mne.info['bads'])==set(eeg.fig.mne.info["ch_names"]):
##        message_box(eeg.root,'Error: At least two channels must be selected!',type='error')
##        return
##    if eeg.fs<250:
##        warning_box(eeg.root,'Warning: Sampling frequency of '+str(eeg.fs)+' Hz is small. Results may not be reliable',type='warning')
##
    ########## PUT THE ENTIRE WARNING FUNCTION CODE IN HERE; IF cancel button pressed, just return from this function; if proceed_button PRESSED
    ##############  JUST CLOSE THE WARNING WINDOW


##    plot_selected(root,eeg,annotations=False)
##
##    X,fs,channels=get_data(eeg,eeg.raw,eeg.fig,eeg.start_time,eeg.end_time)
##
##    hp = 80
##    lp = int(.9*fs/2)
##
##    counts=hfo_count(root,eeg,X,fs,channels,hp,lp, method='rms')
##
##    if len(counts)==0:
##        message_box(eeg.root,'Error: No HFOs found on chosen interval',type='error')
##        filemenu.entryconfig("Open",state="normal")
##        return
##
##    bar_plot_with_slider_rms(counts,channels,num_bars=35,title='HFO Counts')
##
##    filemenu.entryconfig("Open",state="normal")


################ BURNOS METHOD ####################################

##def _hfos_burnos():
##
##    if  set(eeg.fig.mne.info['bads'])==set(eeg.fig.mne.info["ch_names"]):
##        message_box(eeg.root,'Error: At least two channels must be selected!',type='error')
##        return

########## PUT THE ENTIRE WARNING FUNCTION CODE IN BELOW; IF cancel button pressed, just return from this function; if proceed_button PRESSED
##############  JUST CLOSE THE WARNING WINDOW; make sure to filemenu.entryconfig("Open",state="normal")


##    if eeg.fs<250:
##         decision=warning_box(eeg.root,'Warning: Sampling frequency of '+str(eeg.fs)+' Hz is small. Results may not be reliable',type='warning')
##         print('DECISION **********************')
##         print(decision)
##         if decision=='cancel':
##             return
##
##    plot_selected(root,eeg,annotations=False)
##    X,fs,channels=get_data(eeg,eeg.raw,eeg.fig,eeg.start_time,eeg.end_time)
##    start_time=eeg.start_time
##
##    hp = 80
##    lp = int(.9*fs/2)
##    df_combined_results, counts=HFOs_all_channels(root,eeg,X,channels,hp,lp,fs)
##
##
##    if df_combined_results.empty:
##        message_box(eeg.root,'Error: No HFOs found on chosen interval',type='error')
##        filemenu.entryconfig("Open",state="normal")
##        return
##
##    bar_plot_with_slider_burnos(df_combined_results,counts,channels,num_bars=35)
##
##    filemenu.entryconfig("Open",state="normal")


############# MENUS ##################
######################################

menubar = root.menuBar()
########## FILE MENU ######################
filemenu = QMenu("File", menubar)
openAction = QAction("Open", root)
openAction.triggered.connect(_open)

filemenu.addAction(openAction)
menubar.addMenu(filemenu)

########### PLOTS MENU #####################
view_menu = QMenu("Plot", menubar)

plotAction = QAction("Plot Selected", root)
plotAction.triggered.connect(lambda:  plot_selected(eeg))
view_menu.addAction(plotAction)

selectAllAction = QAction("Select All", root)
selectAllAction.triggered.connect(lambda:_select_all_channels(eeg))
view_menu.addAction(selectAllAction)

refreshAction = QAction("Refresh", root)
refreshAction.triggered.connect(lambda:_refresh(eeg))
view_menu.addAction(refreshAction)

for action in view_menu.actions(): action.setEnabled(False)
menubar.addMenu(view_menu)

############ CONNECTIVITY MEASURES MENU ###############
connectivity_menu = QMenu("Connectivity", menubar)
##connectivityOptionsAction = QAction("Connectivity Options", root)
##coherenceAction = QAction("Coherence", root)

coherenceAtMarkAction = QAction("Coherence at Mark", root)
coherenceAtMarkAction.triggered.connect(lambda:  _coh_mark(eeg) )
connectivity_menu.addAction(coherenceAtMarkAction)

coherenceAction = QAction("Coherence", root)
coherenceAction.triggered.connect(lambda:  _coh(eeg) )
connectivity_menu.addAction(coherenceAction)
                                                       
for action in connectivity_menu.actions(): action.setEnabled(False)
menubar.addMenu(connectivity_menu)


##preprocess_menu = QMenu("Data", menubar)
##removeLineNoiseAction = QAction("Remove Line Noise", root)
##plotIndependentSourcesAction = QAction("Plot Independent Sources", root)
##preprocess_menu.addActions([removeLineNoiseAction, plotIndependentSourcesAction])
##for action in preprocess_menu.actions(): action.setEnabled(False)
##menubar.addMenu(preprocess_menu)
##
##hfo_menu = QMenu("HFOs", menubar)
##rootMeanSquareAction = QAction("Root Mean Square", root)
##hilbertStockwellAction = QAction("Hilbert-Stockwell", root)
##hfo_menu.addActions([rootMeanSquareAction, hilbertStockwellAction])
##for action in hfo_menu.actions(): action.setEnabled(False)
##menubar.addMenu(hfo_menu)
##


##coherenceSimilarityStatesAction = QAction("Coherence Similarity States", root)
##directedConnectivityAction = QAction("Directed Connectivity", root)
##directedConnectivityAtMarkAction = QAction("Directed Connectivity at Mark", root)
##reinforcementConnectionsAction = QAction("Reinforcement Connections", root)
##conn_menu.addActions([connectivityOptionsAction, coherenceAction, coherenceAtMarkAction, coherenceSimilarityStatesAction, directedConnectivityAction, directedConnectivityAtMarkAction, reinforcementConnectionsAction])
##for action in conn_menu.actions(): action.setEnabled(False)
##menubar.addMenu(conn_menu)
##
##license_menu = QMenu("Help", menubar)
##videoTutorialAction = QAction("Video Tutorial", root)
##resourcesAction = QAction("Resources", root)
##license_menu.addActions([videoTutorialAction, resourcesAction])
##menubar.addMenu(license_menu)

# menubar = Menu(root)
# filemenu = Menu(menubar, tearoff=0)
# filemenu.add_command(label="Open", command=_open)
# # filemenu.add_command(label="Quit", command=quit)
# menubar.add_cascade(label="File", menu=filemenu)


# view_menu=Menu(menubar,tearoff=0)
# view_menu.add_command(label='Plot Selected', command=_plot_selected)
# view_menu.entryconfig('Plot Selected',state='disabled')
# view_menu.add_command(label='Select All',command=_select_all_channels)
# view_menu.entryconfig('Select All',state='disabled')
# view_menu.add_command(label="Refresh", command=_refresh)
# view_menu.entryconfig('Refresh',state='disabled')
# menubar.add_cascade(label='Plot',menu=view_menu)

# preprocess_menu=Menu(menubar,tearoff=0)

# preprocess_menu.add_command(label="Remove Line Noise", command=_remove_line_noise)
# preprocess_menu.entryconfig('Remove Line Noise',state='disabled')

# preprocess_menu.add_command(label='Plot Independent Sources',command=_plot_sources)
# preprocess_menu.entryconfig('Plot Independent Sources',state='disabled')

# menubar.add_cascade(label='Data',menu=preprocess_menu)


# hfo_menu=Menu(menubar,tearoff=0)
# hfo_menu.add_command(label='Root Mean Square', command=_hfos)
# hfo_menu.entryconfig('Root Mean Square',state='disabled')
# hfo_menu.add_command(label='Hilbert-Stockwell', command=_hfos_burnos)
# hfo_menu.entryconfig('Hilbert-Stockwell',state='disabled')
# menubar.add_cascade(label='HFOs', menu=hfo_menu)


# conn_menu=Menu(menubar,tearoff=0)
# conn_menu.add_command(label="Connectivity Options", command=_conn_options)
# conn_menu.entryconfig('Connectivity Options',state='disabled')

# conn_menu.add_command(label='Coherence', command=_coh)
# conn_menu.entryconfig('Coherence',state='disabled')

# conn_menu.add_command(labe='Coherence at Mark',command=_coh_mark)
# conn_menu.entryconfig('Coherence at Mark',state='disabled')

# conn_menu.add_command(label='Coherence Similarity States', command=_coh_time_communities)
# conn_menu.entryconfig('Coherence Similarity States',state='disabled')

# conn_menu.add_command(label='Directed Connectivity', command=_swadtf)
# conn_menu.entryconfig('Directed Connectivity',state='disabled')

# conn_menu.add_command(label='Directed Connectivity at Mark', command=_swadtf_mark)
# conn_menu.entryconfig('Directed Connectivity at Mark',state='disabled')

# conn_menu.add_command(label='Reinforcement Connections', command=_connection_reinforcements)
# conn_menu.entryconfig('Reinforcement Connections',state='disabled')


# menubar.add_cascade(label='Connectivity', menu=conn_menu)

# license_menu=Menu(menubar,tearoff=0)
# license_menu.add_command(label='Video Tutorial', command=start_video_tutorial)
# license_menu.add_command(label='Resources', command=program_info_window)
# menubar.add_cascade(label='Help', menu=license_menu)


##def disable_menu_pulldowns():
##    for menu in (filemenu,view_menu,preprocess_menu,conn_menu,hfo_menu):  # final version add hfo_menu here
##        for index in range(menu.index('end')+1):
##            menu.entryconfigure(index, state="disable")
##
##    return
##
def enable_menu_pulldowns():
    for action in view_menu.actions(): action.setEnabled(True)
    for action in connectivity_menu.actions(): action.setEnabled(True)
       #for index in range(menu.index('end')+1):
        #   menu.entryconfigure(index, state="normal")
    return


# root.config(menu=menubar)

##def message_box(root, message,type='error'):
##    disable_menu_pulldowns()
##    decision='proceed'
##    message_window = Toplevel(root)
##    message_window.geometry(str(14*len(message))+'x90')
##    message_window.attributes('-topmost',1)
##    message_window.wm_title('')
##
##    def button_press():
##        message_window.destroy()
##        enable_menu_pulldowns()
##    message_label=Label(message_window,fg='red',image = '::tk::icons::'+type)
##    message_label.pack()
##    message_text=Label(message_window,fg='red',text=message,font=("Arial Bold", 14))
##    message_text.pack()
##    message_button = Message_Button(message_window, text='OK', bg='blue',fg='yellow',command=button_press)
##    message_button.pack()
##
##def warning_box(root, message,type='warning'):
##    disable_menu_pulldowns()
##    decision='proceed'
##    warning_window = Toplevel(root)
##    warning_window.geometry(str(14*len(message))+'x120')
##    warning_window.attributes('-topmost',1)
##    warning_window.wm_title('')
##
##    print('Decision inside main warning_box: '+decision)
##
##    def proceed_button_press():
##        decision='proceed'
##        warning_window.destroy()
##        enable_menu_pulldowns()
##        print('Decision inside proceed: '+decision)
##        return decision
##    def cancel_button_press():
##        decision='cancel'
##        warning_window.destroy()
##        enable_menu_pulldowns()
##        print('Decision inside cancel: '+decision)
##        return decision
##
##    warning_label=Label(warning_window,fg='red',image = '::tk::icons::'+type)
##    warning_label.pack()
##    warning_text=Label(warning_window,fg='red',text=message,font=("Arial Bold", 14))
##    warning_text.pack()
##    proceed_button = Message_Button(warning_window, text='Continue', bg='blue',fg='yellow',command=proceed_button_press)
##    proceed_button.pack()
##    cancel_button = Message_Button(warning_window, text='Cancel', bg='blue',fg='yellow',command=cancel_button_press)
##    cancel_button.pack()
##
##    return
##



# root.mainloop()
root.show()
sys.exit(app.exec_())

