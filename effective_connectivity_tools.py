from tkinter import Tk, Frame,Label, Button, Entry,Label, Checkbutton,BooleanVar,filedialog,StringVar,messagebox,ttk,DoubleVar,Frame,LEFT,TOP,RIGHT,BOTTOM,BOTH,Menu,Toplevel,PhotoImage

from tkinter.ttk import Progressbar

import numpy as np
from matplotlib import pyplot as plt

from utils.centrality_scores import reverse_page_rank_score

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.general import font_size

import math
import numpy.ma as ma
from scipy import linalg, fftpack
from scipy.signal import hamming

import connectivipy as cp
#import sails

#from spectral_connectivity import Multitaper, Connectivity


############### MVAR CONSTRUCTION ##############
def construct_mvar(data_win,fs,channels):
    dt = cp.Data(data_win, fs=fs, chan_names=channels)
    best_order, crit=cp.Mvar.order_akaike(data_win,p_max=12,method='yw')  # 'yw ' 'ns' (nutallstrand), 'vm (Viera-Morf)
    #best_order, crit=cp.Mvar.order_schwartz(data,p_max=p_max,method='yw')
    #best_order=4
    dt.fit_mvar(best_order, 'yw')
    acoef, vcoef = dt.mvar_coefficients
    return acoef,vcoef


############## SAILS DDTF ###############################

'''
def ddtf(data_win,channels,fs,freqs,fmin,fmax):
# data is in form num_channels-by-num_times
    #p, crit=cp.Mvar.order_akaike(data_win,p_max=12,method='yw')
    p=3
    X = np.swapaxes(data_win.T, 0, 1)[:, :, np.newaxis]
    delay_vect = np.arange(p+1)

    model = sails.VieiraMorfLinearModel.fit_model(X, delay_vect)
    F = sails.FourierMvarMetrics.initialise(model, fs, freqs)

    DDTF=F.d_directed_transfer_function
    DDTF=np.squeeze(DDTF,axis=3)

    Nf=len(freqs)
    freq_indices=[i for i in range(Nf) if freqs[i]>fmin and freqs[i]<fmax]
    DDTF=np.real(1/len(freq_indices)*np.sum(DDTF[:,:,freq_indices],axis=2))
    return DDTF

'''

########### SPECTRAL CONNECTIVITY, NONPARAMETRIC APPROACH ##############
# data input must hav shape num_times-by-num_channels
'''
def ddtf(data_win,channels, fs, freqs,fmin,fmax):
    X=data_win.T
    m = Multitaper(time_series=X,sampling_frequency=fs)
    c = Connectivity.from_multitaper(m)
    freqs=c.frequencies
    Nf=len(freqs)
    freq_indices=[i for i in range(Nf) if freqs[i]>fmin and freqs[i]<fmax]
    DDTF = c.direct_directed_transfer_function()
    DDTF=np.squeeze(DDTF,axis=0)
    DDTF=np.real(1/len(freq_indices)*np.sum(DDTF[freq_indices,:,:],axis=0))
    return DDTF

'''
############### CONNECTIVIPY DDTF #####################
'''
# data_win is num_channels-by-num_times
def ddtf(data_win,channels, fs, fmin,fmax):
    dt=cp.Data(data_win,fs=fs,chan_names=channels)
    best_order, crit=cp.Mvar.order_akaike(data_win,p_max=20,method='yw')
    dt.fit_mvar(best_order, 'yw')
    ar, vr = dt.mvar_coefficients

    # Result if Num_freq-by-Num_channels-by-Num_channels
    DDTF=dt.conn('ddtf')
    print(np.round(DDTF,2))

    Nf=int(fs/10)

    freqs=[i*1/Nf*fs/2 for i in np.arange(1,Nf+1,1)]
    fmin=5
    fmax=30
    freq_indices=[i for i in range(Nf) if freqs[i]>fmin and freqs[i]<fmax]
    DDTF=np.real(1/len(freq_indices)*np.sum(DDTF[freq_indices,:,:],axis=0))
    return DDTF
'''

###############  SPECTRUM FUNCTION #####################
#######################################################

def spectrum(acoef, vcoef, fs=1, resolution=100):
    """
    Generating data point from matrix *A* with MVAR coefficients.
    Args:
      *acoef* : numpy.array
          array of shape (k, k, p) where *k* is number of channels and
          *p* is a model order.
      *vcoef* : numpy.array
          prediction error matrix (k, k)
      *fs* = 1 : int
          sampling rate
      *resolution* = 100 : int
          number of spectrum data points
    Returns:
      *A_z* : numpy.array
          z-transformed A(f) complex matrix in shape (*resolution*, k, k)
      *H_z* : numpy.array
          inversion of *A_z*
      *S_z* : numpy.array
          spectrum matrix (*resolution*, k, k)
    References:
    .. [1] K. J. Blinowska, R. Kus, M. Kaminski (2004) “Granger causality
           and information flow in multivariate processes”
           Physical Review E 70, 050902.
    """
    p, k, k = acoef.shape
    freqs = np.linspace(0, fs*0.5, resolution)
    A_z = np.zeros((len(freqs), k, k), complex)
    H_z = np.zeros((len(freqs), k, k), complex)
    S_z = np.zeros((len(freqs), k, k), complex)

    I = np.eye(k, dtype=complex)
    for e, f in enumerate(freqs):
        epot = np.zeros((p, 1), complex)
        ce = np.exp(-2.j*np.pi*f*(1./fs))
        epot[0] = ce
        for k in range(1, p):
            epot[k] = epot[k-1]*ce
        A_z[e] = I - np.sum([epot[x]*acoef[x] for x in range(p)], axis=0)
        H_z[e] = np.linalg.inv(A_z[e])
        S_z[e] = np.dot(np.dot(H_z[e], vcoef), H_z[e].T.conj())
    return A_z, H_z, S_z


#################### SWDTF  NEW ###################################
###############################################################


# column number is sending signal (column); incoming information for each channel is 1, so entries along each row are 1.

def swdtf(data_win,channels, fs, fmin,fmax):
    acoef,vcoef=construct_mvar(data_win,fs,channels)
    Az, Hz, Sz=spectrum(acoef, vcoef, fs=1,resolution=100)
    p, N, N = Az.shape
    H2=abs(np.multiply(Hz,Hz))

    f1=fmin
    f2=fmax

    SWDTF=np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            E1=H2[f1:f2,i,j]
            E2=np.sum(H2[f1:f2,j,:],axis=1)
            num=np.sum(E1*E2,axis=0)
            E3=np.expand_dims(E2,axis=0)
            E4=H2[f1:f2,i,:]
            denom=np.sum(np.dot(E3,E4),axis=1)[0]
            temp=num/denom
            SWDTF[i,j]=temp
    # Sum of incoming information for each receiving channel (row) is one
    #print(np.round(np.sum(SWDTF,axis=1),3))

    return SWDTF

'''
#################### SWDTF  OLD ###################################
###############################################################

def swdtf_old(acoef, vcoef, fs, fmin,fmax,resolution=100):

    # column number is sending signal

    Az,Hz,Sz=spectrum(acoef, vcoef, fs=fs, resolution=resolution)
    p, N, N = Az.shape
    H2=abs(np.multiply(Hz,Hz))

    f1=fmin
    f2=fmax

    SWDTF=np.zeros((N,N))

    for i in range(N):
        for j in range(N):
            E1=H2[f1:f2,i,j]
            E2=np.sum(H2[f1:f2,j,:],axis=1)
            num=np.sum(E1*E2,axis=0)
            E3=np.expand_dims(E2,axis=0)
            E4=H2[f1:f2,i,:]
            denom=np.sum(np.dot(E3,E4),axis=1)[0]
            temp=num/denom
            SWDTF[i,j]=temp

    return SWDTF

'''

################# SWDTF CONNECTIONS
def conn_count(SWDTF_list,channels,cutoff=.9):

    '''
    Determine number of reinforcements for each channel using SWDTF
    inputs: list of SWDTF matrices, channels, pct (90 is typical)

    outputs:

        inputs: List of SWDTF matrices, channels, percentile cutoff (0 t0 100)

        output: matrix, conn_matrix, of size num_channels by num_channels. The entry in row i
        column j represents the number of time windows for which the SWDTF from
        channel j to channel i exceeds the p-th percentile over all
        num_win by num_channels by num_channels such values; list of values, conns, whose
        j-th entry is the number of times channel j is a source, i.e. the column sums of conn_matrix

    '''
    num_channels=len(channels)
    num_win=len(SWDTF_list)

    for t in range(num_win):
        np.fill_diagonal(SWDTF_list[t], 0)

    p=np.quantile(SWDTF_list,cutoff)

    indices=np.argwhere(SWDTF_list>=p)

    conn_matrix=np.zeros((num_channels,num_channels))

    for ind in indices:
        conn_matrix[ind[1],ind[2]]+=1 #will need to add entries for signal with more channels

    conns=np.sum(conn_matrix,axis=0).tolist()
    return conn_matrix,conns



############ BARPLOT SHOWING CONNECTION COUNT FOR CHANNELS
def conns_barplot(conns,channels,title):
    conns_window = Toplevel()
    conns_window.configure(bg='lightgrey')
    conns_window.geometry('1000x1000')
    fig2=Figure()
    fig2.suptitle(title, fontsize=16)
    ax = fig2.add_subplot(111)
    ax.bar(channels,conns)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_xticklabels(channels,fontsize=8)

    canvas = FigureCanvasTkAgg(fig2, master=conns_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    return


def swadtf_at_mark(eeg):
    # Compute swadtf matrix using data from a window of with conn_win_value centered about time_mark.
    # data is num_channels-by-num_times

    eeg.segment_loc=eeg.fig.mne.segment_loc
    time_mark= eeg.segment_loc

    bads=eeg.fig.mne.info['bads'] # get current set of bad channels from figure
    channels=eeg.fig.mne.info["ch_names"] # get all channels from fig = eeg.raw.info['ch_names']
    channels=list(set(channels).difference(set(bads)))
    channels=[ch for ch in channels if ch in set(channels)-set(bads)]

    eeg.raw=eeg.raw.pick_channels(channels)

    fs=int(eeg.raw.info['sfreq'])
    start=int((time_mark-eeg.conn_win_value/2)*fs)
    stop=int((time_mark+eeg.conn_win_value/2)*fs)
    data,times=eeg.raw[:,start:stop]
    fmin=eeg.fmin_value
    fmax=eeg.fmax_value

    try:
        SWADTF=swdtf(data,channels, fs, fmin,fmax)
        Scores=reverse_page_rank_score(SWADTF)

    except:
         messagebox.showwarning('Warning',message='Connectivity calculation failed. Consider increasing length of interval or number of channels.')
         return


    fig, ax = plt.subplots(2)

    plot_window = Toplevel(bg="lightgray")
    plot_window.geometry('1400x900')
    plot_window.wm_title('')
    plot_window.attributes('-topmost', 'true')

    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP,fill=BOTH,expand=1)

    heatmap=ax[0].imshow(SWADTF,vmin=0, vmax=1, cmap='coolwarm', aspect='auto')

    ax[0].set_xticks(range(len(channels)))
    ax[0].set_xticklabels(channels,fontsize=font_size(channels))
    ax[0].set_yticks(range(len(channels)))
    ax[0].set_yticklabels(channels,fontsize=font_size(channels))

    ax[0].set_xlabel('sending',fontsize=14)
    ax[0].xaxis.set_label_position('top')
    ax[0].set_ylabel('receiving',fontsize=14)

    ax_divider = make_axes_locatable(ax[0])
    cax = ax_divider.append_axes('right', size='7%', pad='2%')
    cb = fig.colorbar(heatmap, cax=cax, orientation='vertical')

    ax[1].bar(channels,Scores)
    ax[1].set_ylim(0,  max(Scores) )
    ax[1].tick_params(axis='both', labelsize=font_size(channels))
    #ax[1].set_ylabel('Driving Score',fontsize=14)


    fig.suptitle('Directed Connectivity and Driving Scores in Small Window Centered at t= '+str(np.round(time_mark,2))+ ' sec', fontsize=12)



#################  WINDOWED CONNECTIVITY MEASURED #####################
#######################################################################

# Compute sequence of connectivity matrices using prescribed measure and window length.

def effective_connectivity(root,data,channels,fs,conn_win_value,type='swdtf',fmin=5,fmax=30):

    '''
    Computes list of connectivity matrices for signal using subwindows and
    multivariate autoregressive models. Within each window, signal is multiplied
    by hamming windowing function.

    inputs: data, signal of size num_channels-by-num_times; sampling frequency;
    connectivity type: coh,pdc,dtf,ddtf,ffdtf,swdtf (indicate as string);
    smallest and largestfrequencies (for purposes of swdtf), maximum MVAR model order.
    See connectivity_measures_10_4 for how to compute dtf, ddtf, ffdtf, where we
    average over frequencies

    outputs: List of connectivity matrices. The list length is determined by
    conn_win_value. For dtf, ddtf, and ffdtf, each matrix has three
    dimensions: frequency, channel, channel. For swdtf, there are two: channel-by-channel.

    For swdtf animate connectivity matrices and eigenvector centralities

    '''
    Nc=len(channels)

    # See https://www.fieldtriptoolbox.org/assets/pdf/workshop/goettingen2019/2_frequency_analysis.pdf for discussion about binning frequencies
    no=int(fs*conn_win_value)
    nfft=2*no
    win_size=nfft
    Nt=int(np.floor(data.shape[1]/no))-1*1 # must subtract one if using windows below.

    delta_f=1/(2*conn_win_value)
    freqs=[i*delta_f for i in range(no)]

    w=hamming(win_size)
    window = np.expand_dims(w, axis=1)

    M_list=[]
    bar_list=[]

    ######### PROGRESSBAR
    progress_window = Toplevel()
    progress_window.wm_title('Progress')
    progress_window.geometry('300x100')
    progress_window.attributes('-topmost', 'true')
    progress_window.configure(bg='lightgrey')
    #progress_window_label = Label(root, text="Progress")
    #progress_window_label.pack()


    win_progress_var=DoubleVar()
    win_progressbar=Progressbar(master=progress_window,variable=win_progress_var,length=Nt,maximum=1)
    win_progressbar.pack(side=TOP,ipady=5,fill=BOTH,expand=True)
    win_progress_var.set(0)
    win_progressbar.update()
    win_progressbar_label=Label(master=progress_window,text='',bg='lightgray')
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

    singular_matrix_warning=False
    ##########  MAIN TIME LOOP ###########

    for t in range(Nt):
        singular_matrix_warning=False

        # TERMINATE BUTTON AND PROGRESSBAR
        if killed:
            break

        win_progress_var.set((win_count+0)/Nt)
        win_progressbar_label.config(text=str(round(win_count/Nt*100))+'%')
        win_progressbar.update()
        win_count+=1

        data_win=data[:,t*no:t*no+win_size]
        data_win=np.multiply(window,data_win.T).T

        if type=='ddtf':
            M_conn=ddtf(data_win,channels, fs,freqs,fmin,fmax)
            Scores=reverse_page_rank_score(M_conn)
        else:
            try:
                M_conn=swdtf(data_win,channels, fs, fmin,fmax)
                Scores=reverse_page_rank_score(M_conn)

            except:
                 singular_matrix_warning=True
                 M_conn=np.zeros((Nc,Nc))
                 Scores=list(np.zeros(Nc))
                 continue

        M_list.append(M_conn)
        bar_list.append(Scores)


    if singular_matrix_warning:
         messagebox.showwarning('Warning',message='Connectivity calculation failed for at least one time window. Consider increasing length of interval or number of channels. ')

    progress_window.destroy()

    return M_list,bar_list


#################  END WINDOWED CONNECTIVITY MEASURE #####################
#######################################################################



'''
################# SWDTF CONNECTIONS
def conn_count(SWDTF_list,channels,cutoff=.9):

    Determine number of reinforcements for each channel using SWDTF
    inputs: list of SWDTF matrices, channels, pct (90 is typical)

    outputs:

        inputs: List of SWDTF matrices, channels, percentile cutoff (0 t0 100)

        output: matrix, conn_matrix, of size num_channels by num_channels. The entry in row i
        column j represents the number of time windows for which the SWDTF from
        channel j to channel i exceeds the p-th percentile over all
        num_win by num_channels by num_channels such values; list of values, conns, whose
        j-th entry is the number of times channel j is a source, i.e. the column sums of conn_matrix


    num_channels=len(channels)
    num_win=len(SWDTF_list)

    for t in range(num_win):
        np.fill_diagonal(SWDTF_list[t], 0)

    p=np.quantile(SWDTF_list,.95)

    indices=np.argwhere(SWDTF_list>=p)
    #print('indices: '+str(indices))
    conn_matrix=np.zeros((num_channels,num_channels))

    for ind in indices:
        conn_matrix[ind[1],ind[2]]+=1 #will need to add entries for signal with more channels

    conns=np.sum(conn_matrix,axis=0).tolist()
    return conn_matrix,conns

'''
############ BARPLOT SHOWING CONNECTION COUNT FOR CHANNELS
def conns_barplot(conns,channels,title):
    conns_window = Toplevel()
    conns_window.configure(bg='lightgrey')
    conns_window.geometry('1000x1000')
    fig2=Figure()
    fig2.suptitle(title, fontsize=16)
    ax = fig2.add_subplot(111)
    ax.bar(channels,conns)
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_xticklabels(channels,fontsize=8)

    canvas = FigureCanvasTkAgg(fig2, master=conns_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
    return

'''

def swadtf(root,eeg):

    if  set(eeg.fig.mne.info['bads'])==set(eeg.raw.info["ch_names"]):
        messagebox.showerror('Error','No channels selected!')
        return

    X,fs,channels=get_data(eeg.raw,eeg.fig,eeg.start_time,eeg.end_time)
    #X,fs,channels=get_data(eeg.raw_orig,eeg.fig,eeg.start_time,eeg.end_time)

    conn_win_value=eeg.conn_win_value
    fmin=eeg.fmin_value
    fmax=eeg.fmax_value

    M_list=connectivity(root,X,fs,channels,type='swdtf',conn_win_value=conn_win_value,fmin=fmin,fmax=fmax,p_max=20)
    #print(np.sum((M_list[1])[:,:],axis=1))

    cutoff=eeg.conns_percentile

    conn_matrix,conns=conn_count(M_list,channels,cutoff=cutoff)

    #conns_barplot(conns,channels,'Reinforcement Connections')
    bar_plot_with_slider(conns,channels,num_bars=8,title='Reinforcement Connections')

'''
