import mne
from mne.viz import set_browser_backend

import pandas as pd
 import plotly.express as px

import matplotlib
from matplotlib import pyplot as plt
mne.set_config('MNE_BROWSE_RAW_SIZE','20,10')

# Basic tkinter
from tkinter import Tk, Frame,TOP, BOTH

# read in sample data file, which must be in same directory as this script. It's stored in an mne class named raw

file='sample_data.edf'
raw=mne.io.read_raw_edf(file,preload=True)

### Three plotting options, below:

# choice=0: Creates plot using qt backend. The plot quality is good, but using it in my software will require converting everything
# from tkinter to qt

# choice=1: Standard data plot using the plot method from MNE, which uses matplotlib backend.

# choice=2: Standard data plot from choice 1, but figure is placed on tkinter figure canvas. This is what my software currently does.

# choice =3: Data is converted to pandas dataframe, which is then plotted. (Only first four channels are selected here.) This is
# a bit slow, but I'm sure this could be placed on a tkinter figure canvas.

# choice=4: Data is converted to pandas data frame and then plotted in a local browser window using plotly module with scrolling.
# I'm wondering if it might be possible to create an html widget of sorts containing an interactive plotly browser?

choice=0

if choice ==0:
    set_browser_backend("qt")
    fig=raw.plot(show=True,block=True, duration=20,bgcolor='w',color='b', bad_color='r')
    plt.show()

if  choice==1:
    set_browser_backend('matplotlib')
    fig=raw.plot(show=True,block=True, duration=20,bgcolor='w',color='b', bad_color='r')
    plt.show()

if choice==2:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    set_browser_backend('matplotlib')
    matplotlib.use('Qt5Agg',force=True)
    
    root = Tk()
    root.wm_title("Plot figure on canvas")
    root.geometry('800x600')

    fig=raw.plot(show=False,block=True, duration=20,bgcolor='w',color='b', bad_color='r')
    
    canvas_frame=Frame(root)
    canvas_frame.pack(side=TOP,expand=True)

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=TOP,fill=BOTH,expand=1)

    root.mainloop() 

if choice==3 or choice ==4:
    df=raw.to_data_frame()
    df=df.iloc[:, [1,2,3,4]]
    
    if choice==3:
        df.plot()
        plt.show()

    if choice==4:
        df['x'] = df.index
        df_melt = pd.melt(df, id_vars="x", value_vars=df.columns[:-1])
        fig=px.line(df_melt, x="x", y="value",color="variable")

        # Add range slider
        fig.update_layout(xaxis=dict(rangeslider=dict(visible=True),
                             type="linear"))
        fig.show()


