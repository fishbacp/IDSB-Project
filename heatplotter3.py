from PyQt5.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget,QLineEdit,QLabel,QSlider
#from PyQt5.QtWidgets import QSlider, QLabel,QTextEdit


#  Import PyQt5 first before pyqtgraph
import PyQt5
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator,QFont

from qtpy.QtCore import (QEvent, QThread, Qt, Signal, QRectF, QLineF,QPointF, QPoint, QSettings)

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, mkQApp,QtGui, QtCore

import numpy as np
import matplotlib

pg.setConfigOption('background', 'lightgray')
pg.setConfigOption('foreground','black')


font = QFont("Arial", 18)

def make_matrices(n,N):
  global matrices
  matrices=[np.random.rand(n,n) for i in range(N)]
  return matrices

columns = ["A", "B", "C",'D','E','F','G','H','I','J','K','L','M']
n=len(columns)
N=50
matrices=make_matrices(n,N)

class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.central = QWidget()                          # create a QWidget
        #slider = QSlider(orientation=QtCore.Qt.Vertical)  # Vertical Slider

        graph_widget = pg.GraphicsLayoutWidget(show=True)

        plotItem = graph_widget.addPlot()
        plotItem.resize(200,200)
        
        self.setWindowTitle('Coherence Animation')
        self.setStyleSheet("background-color: lightgray;")
        self.resize(800,800)

        self.setCentralWidget(self.central)  # set the QWidget as centralWidget

        outer_layout = QVBoxLayout(self.central)


        num_frames_widget=QLineEdit()
        num_frames_widget.setFixedWidth(40)
        num_frames_widget.setFont(font)
        num_frames_widget.setText('50')

        num_frames_widget.setStyleSheet("QLineEdit"
                           "{"
                           "background-color : white;"
                           "}")
        
        validator=QIntValidator()
        validator.setRange(5,500)
        num_frames_widget.setValidator(validator)
        num_frames_widget.setMaxLength(3)        
        num_frames_widget.setAlignment(Qt.AlignRight)

        matrices=make_matrices(n,N)

        
        outer_layout.addWidget(num_frames_widget)
        #num_frames_widget.setAlignment(Qt.AlignHCenter)

        
        counter_widget=QLabel("Time value = 0")
        counter_widget.setFont(font)
        outer_layout.addWidget(counter_widget)     
        counter_widget.setAlignment(Qt.AlignHCenter)

        layout1 = QHBoxLayout(self.central)  # assign layout to central widget
        layout1.addWidget(graph_widget) # No More error

        outer_layout.addLayout(layout1)

        
        def plotter():
            heatmap.setImage(matrices[time_value.value()-1])
            counter_widget.setText('Time value = '+str(time_value.value()))
            #print(2*int(num_frames_widget.text()))
                        
        time_value=QSlider()
        time_value.setRange(0,N)
        time_value.setValue(0)
        sliderWidget=layout1.addWidget(time_value) #add slider to layout 1 (top row)
        time_value.valueChanged.connect(plotter)

        self.show()

      
        pg.setConfigOption('imageAxisOrder', 'row-major')
        heatmap = pg.ImageItem()
        tr = QtGui.QTransform().translate(-0.5, -0.5)
        heatmap.setTransform(tr)
        
        corr_matrix=matrices[0] #np.random.rand(5,5)
        heatmap.setImage(corr_matrix)

        plotItem.invertY(True)
        plotItem.setDefaultPadding(0.0)
        plotItem.addItem(heatmap)

        plotItem.showAxes( True, showValues=(True, True, False, False), size=40 )

        ticks = [ (idx, label) for idx, label in enumerate( columns ) ]
        for side in ('left','top','right','bottom'):
            plotItem.getAxis(side).setTicks( (ticks, []) )
            plotItem.getAxis(side).setTickFont(font)
        plotItem.getAxis('bottom').setHeight(10)

        colorMap=pg.colormap.getFromMatplotlib('jet')
        bar = pg.ColorBarItem( interactive=False,values=(0,1) , colorMap=colorMap)
        bar.setImageItem(heatmap, insert_in=plotItem)


mkQApp("Matrix Animation")

main_window = MainWindow()

if __name__ == '__main__':
    pg.exec()

