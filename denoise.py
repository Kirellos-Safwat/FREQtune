import numpy as np
import sys
import os
from pyqtgraph.exporters import ImageExporter
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel, QHBoxLayout, QFileDialog, QMessageBox
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from scipy import signal as sg


class Denoise(QWidget):
    def __init__(self, signal):
        super().__init__()
        self.signal = signal
        #storing starting and ending points of selection
        self.start_pos = None
        self.end_pos = None
        self.sub_signal = None

        #disable mouse panning when performing selection
        self.mouse_move_connected = False


        self.initUI()

    def initUI(self):
        self.setWindowTitle("Noise Reduction") # Window Title
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        #disable mouse panning when performing selection
        self.plot_widget.scene().sigMouseClicked.connect(self.on_mouse_clicked)

        self.plot_widget.setLimits(xMin=0, yMin=min(self.signal.data), yMax=max(self.signal.data))
        self.plot_widget.plot(self.signal.time, self.signal.data, pen={'color': '#3D8262'})
        #create region item to highlight selected area
        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)  
        self.region.hide()  # default is hidden till used
        self.plot_widget.addItem(self.region)
        self.plot_widget.plotItem.vb.sigRangeChanged.connect(self.on_range_changed)

        # Get the viewbox and connect the range change event
        self.viewbox = self.plot_widget.getViewBox()
        self.viewbox.setMouseEnabled(x=True, y=True)  # Enable mouse interaction only in X-direction
        self.viewbox.sigRangeChanged.connect(self.on_range_changed)

        # Store initial y-range to keep it constant
        self.initial_y_range = self.viewbox.viewRange()[1]
        
    def on_range_changed(self):
        """Custom zoom behavior: Restrict zoom to X-axis only."""
        # Get the current range for x and y axes
        x_range, y_range = self.viewbox.viewRange()

        # Keep the Y range constant (store initial range when the window is created)
        y_range_new = self.initial_y_range
        
        # Update the viewbox with only the X range adjusted
        self.viewbox.setRange(xRange=x_range, yRange=y_range_new, padding=0)

    def wiener_filter(self, data , selected_range):
        start, end = selected_range
        start_idx = int(start)
        end_idx = int(end)
        sub_signal = data[start_idx:end_idx + 1]
        # apply wiener filter to the signal
        return sg.wiener(sub_signal)

    def on_mouse_clicked(self, event):
        #mouse click
        if event.button() == Qt.LeftButton:
            pos = event.scenePos()
            mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)

            if self.start_pos is None:  #1st click
                self.start_pos = mouse_point.x()
                self.region.setRegion([self.start_pos, self.start_pos])  #start ur region
                self.region.show()

                #temporarily disable panning and zooming
                if not self.mouse_move_connected:
                    self.plot_widget.scene().sigMouseMoved.connect(self.on_mouse_moved)
                    self.mouse_move_connected = True

            else:  #2nd click
                self.end_pos = mouse_point.x()
                self.region.setRegion([self.start_pos, self.end_pos])  #end ur region

                selected_range = (self.start_pos, self.end_pos)
                self.create_sub_signal(selected_range)

                #reset after selection
                self.plot_widget.setMouseEnabled(x=True, y=True)
                self.start_pos = None 

    def on_mouse_moved(self, event):
        #mouse movement
        pos = event
        mouse_point = self.plot_widget.plotItem.vb.mapSceneToView(pos)

        if self.start_pos is not None:
            self.end_pos = mouse_point.x()
            self.region.setRegion([self.start_pos, self.end_pos])  #update region to follow mouse

    def create_sub_signal(self, selected_range):
        #get start, end positions from selected range
        start, end = selected_range
        start_idx = int(start)
        end_idx = int(end)

        start_idx = max(0, min(start_idx, len(self.signal.data) - 1))
        end_idx = max(0, min(end_idx, len(self.signal.data) - 1))

        if start_idx > end_idx:  
            start_idx, end_idx = end_idx, start_idx
        #extraction of sub-signal
        self.sub_signal  = self.signal.data[start_idx:end_idx + 1]
        # plot
        # self.plot_widget.plot(self.signal.time, self.signal.data, pen={'color': '#3D8262'})
        #hide region after selection
        filtered_signal = self.wiener_filter(self.signal.data, selected_range)
        self.plot_widget.clear()
        self.plot_widget.plot(filtered_signal, pen={'color': 'r'})
        self.region.hide()

    def reset_graph(self):
        self.plot_widget.clear()
        self.start_pos = None
        self.end_pos = None
        self.region.hide()  

        self.region = pg.LinearRegionItem()
        self.region.setZValue(10)  
        self.region.hide()  #default is hidden till used
        self.plot_widget.addItem(self.region)

    

    # def keyPressEvent(self, event):
    #     if event.key() == Qt.Key_Left:

    #     elif event.key() == Qt.Key_Right:

