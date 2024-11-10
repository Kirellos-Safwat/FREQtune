from PyQt5.QtWidgets import QSlider
import matplotlib as plt
from PyQt5 import QtCore
plt.use('Qt5Agg')

class Slider:
    def __init__(self , index ):
        # Create a slider
        self.index= index
        self.slider = QSlider()
        #sets the orientation of the slider to be vertical.
        self.slider.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(10)
        self.slider.setSingleStep(1)
        self.slider.setMinimum(0)
        self.slider.setMaximum(20)
        self.slider.setValue(10)
        # self.sliderlabel.setText()
    def get_slider(self):
        return self.slider