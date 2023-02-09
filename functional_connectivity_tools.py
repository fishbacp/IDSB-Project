import mne
from mne_connectivity import spectral_connectivity_epochs
import networkx as nx
import network_tools as nt
from netgraph import Graph

import warnings
warnings.filterwarnings("ignore")

import connectivipy as cp

from mne import create_info,Annotations
from mne.io import RawArray

from tkinter import Tk,Toplevel,Label,DoubleVar,TOP,BOTH,Frame,Button,messagebox
from tkinter.ttk import Progressbar
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams["figure.figsize"] = (10,8)


from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans,AffinityPropagation
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

from scipy.signal import hamming

#from network_tools import louvain, leiden

from sklearn.metrics.cluster import adjusted_rand_score

from utils.general import font_size, PSD
#from utils.animation_tools import heatplot_centrality_animation_combined
from utils.centrality_scores import generalized_degree_centrality

def symmeterize(A):
    #Return a symmetrized version of NumPy lower-triangular array A with 1's along diagonal.
    np.fill_diagonal(A, 1)
    return A + A.T - np.diag(A.diagonal())



def coherence_at_mark(eeg):
    #if eeg.fig.mne.segment_loc==0: ### ENSURE SEGMENT SELECTED
    #        messagebox.showerror("Error","Select time point in recording!")
    #        error=True

    eeg.segment_loc=eeg.fig.mne.segment_loc
    time_mark= eeg.segment_loc

    bads=eeg.fig.mne.info['bads'] # get current set of bad channels from figure
    channels=eeg.fig.mne.info["ch_names"] # get all channels from fig = eeg.raw.info['ch_names']
    #channels=list(set(channels).difference(set(bads)))
    channels=[ch for ch in channels if ch in set(channels)-set(bads)]

    eeg.raw=eeg.raw.pick_channels(channels)

    fs=int(eeg.raw.info['sfreq'])
    start=int((time_mark-eeg.conn_win_value/2)*fs)
    stop=int((time_mark+eeg.conn_win_value/2)*fs)
    data,times=eeg.raw[:,start:stop]
    fmin=eeg.fmin_value
    fmax=eeg.fmax_value


    Nc=len(channels)
    data_win=np.expand_dims(data, axis=0)

    Coh=spectral_connectivity_epochs(data_win,sfreq=fs, method='coh', mode='multitaper', fmin=fmin, fmax=fmax, faverage=True )
    S=symmeterize(np.reshape(Coh.get_data(), (Nc,Nc)))

    fig, ax = plt.subplots(2)

    plot_window = Toplevel(bg="lightgray")
    plot_window.geometry('1400x900')
    plot_window.wm_title('')
    plot_window.attributes('-topmost', 'true')

    canvas = FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP,fill=BOTH,expand=1)

    heatmap=ax[0].imshow(S,vmin=0, vmax=1, cmap='coolwarm', aspect='auto')

    ax[0].set_xticks(range(Nc))
    ax[0].set_xticklabels(channels,fontsize=font_size(channels))
    ax[0].set_yticks(range(Nc))
    ax[0].set_yticklabels(channels,fontsize=font_size(channels))

    ax_divider = make_axes_locatable(ax[0])
    cax = ax_divider.append_axes('right', size='7%', pad='2%')
    cb = fig.colorbar(heatmap, cax=cax, orientation='vertical')

    G,community_labels,node_to_community,labels_dict,node_color=nt.Graph_communities_params(S,channels)

    # See https://github.com/paulbrodersen/netgraph/blob/master/netgraph/_main.py for options
    Graph(G,ax=ax[1],node_size=5,node_label_offset=.1,node_color=node_color, node_labels=labels_dict,node_edge_width=0, edge_alpha=0.1, node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
    edge_layout='bundled', edge_layout_kwargs=dict(k=2000))

    fig.suptitle('Coherence in Small Window Centered at t= '+str(np.round(time_mark,2))+ ' sec \n Coherence Communities: '+str(community_labels), fontsize=12)


############ EXAMPLE
'''
raw = mne.io.read_raw_edf('11.edf',preload=True)
channels=raw.info["ch_names"]
Nc=10
fs=int(raw.info['sfreq'])
data,times=raw[0:Nc,:]
channels=channels[0:Nc]

time_mark=80
conn_win_value=.5

fmin=5
fmax=30

root=Tk()
root.geometry('900x900')
root.wm_title("Main Window")

coherence_at_mark(root,data,fs,channels,time_mark,conn_win_value,fmin=5,fmax=50)

root.mainloop()
'''

#### CREATE LIST OF FUNCTIONAL CONNECTIVITY MATRICES AVERAGED OVER FREQUENCY
####  fmin TO fmax. ALSO CREATE CORRESPONDING CONFIGURATION MATRIX
##### data has size num_channels -by -num_times

### CAN ALSO USE FREQUENCY BANDS:
###  fmin=(2.5,4,8,12,30) 100,250,500)
### fmax=(4,8,12,30,100) 250,500,1000)  frequency ranges are  delta, theta, alpha, beta, gamma,
### IN THIS CASE S WILL HAVE SIZE NUM_CHANNELS -BY - NUM_CHANNELS -BY -NUM FREQUENCY BANDS

def functional_connectivities(root,data,channels,fs,conn_win_value,method='coh',fmin=5,fmax=30):

    Nc=len(channels)
    no=int(fs*conn_win_value)
    nfft=2*no
    win_size=nfft
    Nt=int(np.floor(data.shape[1]/no))-1*1 # must subtract one if using windows below.

    # using multitaper method below, no need to window. See https://www.osti.gov/servlets/purl/1560107?
    w=hamming(win_size)
    window = np.expand_dims(w, axis=1)

    M_list=[]
    bar_list=[]
    config_matrix=np.zeros((int(Nc*(Nc-1)/2),Nt))

    ######### PROGRESSBAR
    progress_window = Toplevel(root)
    progress_window.geometry('300x100')
    progress_window.wm_title('Progress')
    progress_window.attributes('-topmost', 'true')
    progress_window.configure(bg='lightgrey')
    #progress_window_label = Label(root, text="")
   # progress_window_label.pack()


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

    ##########  MAIN TIME LOOP ###########

    # available methods include: 'coh', 'plv', 'pli', 'wpli' . See https://mne.tools/0.13/generated/mne.connectivity.spectral_connectivity.html
    for t in range(Nt):

        # TERMINATE BUTTON AND PROGRESSBAR
         if killed:
            break

         win_progress_var.set((win_count+0)/Nt)
         win_progressbar_label.config(text=str(round(win_count/Nt*100))+'%')
         win_progressbar.update()
         win_count+=1


         data_win=data[:,t*no:t*no+win_size]
         data_win=np.multiply(window,data_win.T).T

         data_win=np.expand_dims(data_win, axis=0)
         #S,freqs,times,n_epochs,n_tapers=spectral_connectivity(data_win,sfreq=fs, method='coh', mode='multitaper', fmin=fmin, fmax=fmax, faverage=True )
         #M_list.append(symmetrize(S[:,:,0]))

         Coh=spectral_connectivity_epochs(data_win,sfreq=fs, method='coh', mode='multitaper', fmin=fmin, fmax=fmax, faverage=True )
         S=symmeterize(np.reshape(Coh.get_data(), (Nc,Nc)))
         M_list.append(S)
         config_matrix[:,t]=S[np.triu_indices(Nc,k=1)]

         #GDC=generalized_degree_centrality
         alpha=.8 #alpha=1 means centrality is just weighted degree; alpha=0 means we are just using adjacency matrix
         GDC=generalized_degree_centrality(S,alpha)
         #PSD_list=PSD(data[:,t*no:t*no+win_size],channels,fs,fmin,fmax)
         bar_list.append(GDC)

    progress_window.destroy()
    # print(np.round(M_list[0:3],3))
    # print(np.round(config_matrix[:,0:3],3))

    return M_list,config_matrix,bar_list



############# CLUSTER TIME WINDOWS BASED UPON CONFIGURATION MATRIX************
############### PLOT LIKE WINDOWS AS ANNOTATIONS

def like_brain_states(root,eeg,config_matrix,conn_win_value,method='affinity_prop'):

    #progress_window = Toplevel(root)
    #progress_window.wm_title('')
    #progress_window.attributes('-topmost', 'true')
    #progress_window.configure(bg='lightgrey')
    #progress_window_label = Label(root, text="")
    #progress_window.geometry('300x100')
    #progress_window_label.pack()

    #win_progressbar_label=Label(master=progress_window,text='Computing...be patient!',bg='lightgray')
    #win_progressbar_label.pack(side=TOP,pady=5,fill=BOTH, expand=True)

    #config_matrix = preprocessing.scale(config_matrix.T).T

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(config_matrix.T)

    # scaled features has shape num_time_windows -by -num_channels

    Nt=config_matrix.shape[1]

# CLUSTERING VIA AFFINITY PROPAGATION ON COLUMNS OF STANDARDIZED CONFIGURATION MATRIX


    if method=='affinity_prop':
        clustering = AffinityPropagation(random_state=5).fit(scaled_features)
        assignments=clustering.labels_

# CLUSTERING VIA LOUVAIN METHOD APPLIED TO ADJACENCY MATRIX, WHOSE ENTRIES ARE MAGNITUDES OF CORRELATIONS OF
# CORRESPONDING PEARSON CORRELATION MATRIX
# SINCE SCALED FEATURES HAS SIZE NUM-TIME-WINDOWS BY NUM-CHANNELS, WE USE TRANSPOSE WHEN COMPUTING CORRELATION MATRIX


    if method=='correlation':
        corr_matrix=abs(np.corrcoef(config_matrix.T))
        G=nx.from_numpy_matrix(corr_matrix)
        partition=community_louvain(G)
        assignments=list(partition.values())


# CLUSTERING VIA K-MEANS ON COLUMNS OF CONFIGURATION MATRIX

    if method=='kmeans':
        sse = []
        num_cluster_values=int(Nt/4)
        for k in range(1, num_cluster_values):
            kmeans = KMeans(init="random",n_clusters=k,n_init=10,max_iter=300,random_state=42)
            kmeans.fit(scaled_features)
            sse.append(kmeans.inertia_)

        # elbow method
        try:
            kl = KneeLocator(range(1, num_cluster_values), sse, curve="concave", direction="decreasing")
            k_optimal=kl.elbow

            #print('k optimal: '+str(k_optimal))

            kmeans = KMeans(init="random",n_clusters=k_optimal,n_init=10,max_iter=300,random_state=42)
            kmeans.fit(scaled_features)
            assignments=kmeans.labels_

        except:
            clustering = AffinityPropagation(random_state=5).fit(scaled_features)
            assignments=clustering.labels_


    descriptions=[str(comm) for comm in assignments]

    onsets=[eeg.start_time+t*conn_win_value for t in range(Nt)]
    durations=[conn_win_value for t in range(Nt)]

    community_markings=Annotations(onset=onsets,duration=durations,description=descriptions)
    raw_temp=eeg.raw.copy()
    raw_temp.set_annotations(community_markings)

    #progress_window.destroy()

    ##### CLOSE ANY OPEN WIDGETS AND PLACE RAW ON TOP
    '''
    def all_children (window) :
        _list = window.winfo_children()
        for item in _list :
            if item.winfo_children() :
                _list.extend(item.winfo_children())
        return _list

    widget_list = all_children(root)
    for item in widget_list:
        item.pack_forget()

    '''

    time_communities_window = Toplevel()
    time_communities_window.configure(bg='lightgrey')
    time_communities_window.geometry('1400x900')

    time_communities_window.wm_title(str(Nt)+' windows; '+str(max(assignments))+ ' coherence similarity states')
    time_communities_window.attributes('-topmost', 'true')
    mne.set_config('MNE_BROWSE_RAW_SIZE','16,4')
    plt.rcParams["figure.figsize"] = [18,10]
    fig=raw_temp.plot(show=False,block=True,scalings=.000050,n_channels=10,bad_color='gray',start=eeg.start_time,duration=10) #(eeg.end_time-eeg.start_time))

    canvas = FigureCanvasTkAgg(fig, master=time_communities_window)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP,fill=BOTH,expand=1)

    #title = Label(canvas_frame, text=str(Nt)+' windows; '+str(max(assignments))+ ' states',bg='white')
    #title.pack(side=TOP,fill=BOTH,expand=1)


##### FROM THE CONFIGURATION MATRIX COMPUTE COSINE SIMILARITY BETWEEN ALL POSSIBLE PAIRS
##### OF COLUMNS IN ORDER TO CREATE NETWORK ADJACENCY MATRIX, WHERE NODES ARE TIME WINDOWS

def time_windows_adjacency_matrix(raw,method='coh',conn_win_value=1,fmin=5,fmax=30):
    M_list,config_matrix=functional_connectivities(raw,method=method,conn_win_value=conn_win_value,fmin=fmin,fmax=fmax)
    Nt=config_matrix.shape[1]
    # WARNING: need Nt>=2 !!!
    Adj=np.zeros((Nt,Nt))
    for t in range(Nt):
        for s in range(Nt):
            u=config_matrix[:,t].reshape(1,config_matrix.shape[0])
            v=config_matrix[:,s].reshape(1,config_matrix.shape[0])
            #Adj[t,s]=cos_sim(u,v)[0][0]
            Adj[t,s]=euclid_dist(u,v)[0][0]
    return Adj

##### COMPUTE TIME WINDOWS COMMUNITIES USING MODULARITY

def time_windows_communities(raw,method='coh',conn_win_value=2,fmin=5,fmax=30):
    Adj=time_windows_adjacency_matrix(raw,method=method,conn_win_value=conn_win_value,fmin=fmin,fmax=fmax)
    Nt=Adj.shape[0]
    G=nx.from_numpy_matrix(Adj)
    labels_dict={n:str(n) for n in range(Nt)}
    community_assignments,Q,community_labels=louvain(G,labels_dict,node_size=1000,font_size=17,graph=False)
    return community_assignments,Q,community_labels


'''
Adj=time_windows_adjacency_matrix(raw,method='coh',conn_win_value=conn_win_value,fmin=fmin,fmax=fmax)
print(np.round(Adj,pl))

G=nx.from_numpy_matrix(Adj)
labels={n:str(n) for n in range(Adj.shape[0])}
#community_assignments,Q,community_labels=leiden(G,labels,node_size=1000,font_size=17,resolution_parameter=1,graph=False)
#print(community_labels)

community_assignments,Q,community_labels=louvain(G,labels,graph=False)

# within louvain, community_assignments is a list where entry i tells us, by number, which community i belongs to.
print(community_assignments)
print(Q)

'''

############# SAMPLE IMPLEMENTATIONS ###########################

############### USING SIMULATED SIGNALS

'''
from utils.simulated_signals import signal_1,signal_2
X=signal_1(show=False) # X signal 2 has size 5-by-10K

X=np.array([X[:,0],X[:,0],X[:,1],X[:,1],X[:,2],X[:,3],X[:,2]]).T
print('Shape of X:')
print(X.shape)

fs=1000
channels=['A','B','C','D','E','F','G']
info = create_info(sfreq=fs, ch_names=channels,ch_types='seeg')
raw = RawArray(data=X.T, info=info)

############ USING ACTUAL DATA

raw = mne.io.read_raw_edf('11.edf',preload=True)
channels=raw.info["ch_names"]
Nc=10
fs=int(raw.info['sfreq'])
data,times=raw[0:Nc,:]
channels=channels[0:Nc]
info = create_info(sfreq=fs, ch_names=channels,ch_types='seeg')
raw = RawArray(data=data, info=info)


# Note: conn_win_value*2*fmin must be at least 5 cycles.
#conn_win_value=10
fmin=5

#fmin=5/(conn_win_value*2)
conn_win_value=5/(fmin*2)
fmax=30 # must be larger than fmin

root=Tk()
root.geometry('1000x1000')
root.wm_title("Main Window")

M_list,config_matrix,bar_list=functional_connectivities(root,data,channels,fs,method='coh',conn_win_value=conn_win_value,fmin=fmin,fmax=fmax)

root.mainloop()


### EXAMPLE ANIMATION

from tkinter import Tk,TOP,BOTH,Toplevel
from utils.animation_tools import heatplot_animation

root=Tk()
root.geometry('1000x1000')
root.wm_title("Main Window")

#M_list=[np.random.rand(4,4) for i in range(50)]
#channels=['a','b','c','d']

heatplot_animation(root,channels,M_list,conn_win_value)

root.mainloop()

'''

######## COMPUTE MATRIX OF CLUSTERING COEFFICIENTS BASED UPON MATRICES IN M_list
######## OUTPUT MATRIX HAS SIZE NUM_CHANNELS-BY-NUM_TIMES. EACH ENTRY (C,T) RECORDS
####### THE CLUSTERING COEFFICIENT FOR CHANNEL C AT TIME T WINDOW
def clustering_coefficient_matrix(M_list,channels):
    Nc=len(channels)
    Nt=len(M_list)
    cluster_coeff_matrix=np.zeros((Nc,Nt))
    for t in range(Nt):
        M=M_list[t]
        G=nx.from_numpy_matrix(M)
        C=nx.clustering(G,weight='weight')
        cluster_coeff_matrix[:,t]=np.fromiter(C.values(), dtype=float)
    return(cluster_coeff_matrix)

'''Example
M_list=[np.random.rand(5,5) for i in range(300)]
channels=['a','b','c','d','e']
cluster_coeff_matrix=clustering_coefficient_matrix(M_list,channels)
'''
