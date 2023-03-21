XKCD_COLORS = {
        'cloudy blue': '#acc2d9',
        'dark pastel green': '#56ae57',
        'dust': '#b2996e',
        'electric lime': '#a8ff04',
        'fresh green': '#69d84f',
        'light eggplant': '#894585',
        'nasty green': '#70b23f',
        'really light blue': '#d4ffff',
        'tea': '#65ab7c',
        'warm purple': '#952e8f',
        'yellowish tan': '#fcfc81',
        'cement': '#a5a391',
        'dark grass green': '#388004',
        'dusty teal': '#4c9085',
        'grey teal': '#5e9b8a',
        'macaroni and cheese': '#efb435',
        'pinkish tan': '#d99b82',
        'spruce': '#0a5f38',
        'strong blue': '#0c06f7',
        'toxic green': '#61de2a',
        'windows blue': '#3778bf',
        'blue blue': '#2242c7',
        'blue with a hint of purple': '#533cc6',
        'booger': '#9bb53c',
        'bright sea green': '#05ffa6'
        }
color_names=list(XKCD_COLORS.values())

import numpy as np
import sys
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

# netgraph is a matplotlib-based network plotting tool. 
from netgraph import Graph 

import networkx as nx
from networkx.algorithms.community.louvain import louvain_communities


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=20, height=20, dpi=100):
        super(MplCanvas, self).__init__(Figure(figsize=(width, height), dpi=dpi))
        self.setParent(parent)
        self.ax = self.figure.add_subplot(111)

 ######## INPUT: list of channel labels and matrix S, where  S will correspond to coherence_at_mark(eeg) for our purposes.
######### OUTPUT: PyQt plot of network communities, where community memberships are represented by color.
######### See https://www.pythonguis.com/tutorials/plotting-matplotlib/ which discusses creating PyQt plots using matplotlib.

        # SAMPLE MATRIX S AND LIST OF CHANNEL LABELS:

        S=np.array([[0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
          0, 0],
         [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0],
         [1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
          0, 0],
         [1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
          0, 0],
         [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
          0, 0],
         [1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0],
         [1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0],
         [0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
          0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
          0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
          0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
          0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
          0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
          0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
          0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0,
          0, 0],
         [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
          0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
          1, 1],
         [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1,
          1, 1],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1,
          1, 1],
         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1,
          1, 1],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
          1, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
          0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1,
          1, 0]])

        channels=['A'+str(i) for i in range(S.shape[0])]

##### CREATE NETWORK HAVING ADJACENCY MATRIX S
        G=nx.from_numpy_matrix(S)
        Nc=len(channels)
        channels_dict={i:channels[i] for i in range(Nc)}
        

###### DETERMINE COMMUNITIES USING LOUVAIN METHOD

        clusters=louvain_communities(G)

###### PARTITION OF LABELS, E.G. {{A1, A4, A5}, {A8, A10, A22}, ...}
        community_labels=[{channels[member] for member in comm} for comm in clusters ]
        node_to_community = dict()
        nodes=range(Nc)
        for i in range(Nc):
            for j in range(len(clusters)):
                if nodes[i] in clusters[j]:
                    node_to_community.update({i:j})
                    
####### ASSIGN A COLOR TO EACH COMMUNITY
            
        community_to_color={i:color_names[i] for i in range(8)}        
        node_color = {node: community_to_color[community_id] for node, community_id in node_to_community.items()}

######  USE NETGRAPH TO PLOT NETWORK AND HIGHLIGHT COMMUNITIES
        
        self_plot_instance=Graph(G,ax=self.ax,node_labels=channels_dict,node_label_fontdict=dict(size=12), node_label_offset=0.075,
        node_color=node_color, node_edge_width=0, edge_alpha=0.1  ,
        node_layout='community', node_layout_kwargs=dict(node_to_community=node_to_community),
        edge_layout='bundled', edge_layout_kwargs=dict(k=2000),
        )

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.canvas = MplCanvas(self, width=10, height=10, dpi=100)
        
        widget = QtWidgets.QWidget()
        self.setCentralWidget(widget)

        layout = QtWidgets.QVBoxLayout(widget)
        layout.addWidget(self.canvas)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
