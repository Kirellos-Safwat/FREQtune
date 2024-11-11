from scipy.fft import fft #four fourier transform
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import pandas as pd
import copy
from PyQt5.QtWidgets import QSlider,QHBoxLayout , QLabel
import matplotlib as plt
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, uic 
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QTimer, QEvent
import os
import sys
plt.use('Qt5Agg')
import librosa #for audio processing
import bisect
import pyqtgraph as pg
from scipy import signal as sg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sounddevice as sd #for handling audio feedback
import numpy as np
from Signal import SignalGenerator
from Slider import Slider
from PyQt5.QtCore import Qt


    
class EqualizerApp(QtWidgets.QMainWindow):    
    def __init__(self, *args, **kwargs):
        super(EqualizerApp, self).__init__(*args, **kwargs)
        # Load the UI Page
        uic.loadUi(r'task3.ui', self)
        self.is_playing = False  # To keep track of whether the audio is playing or paused
        self.playback_speed = 1.0  # Default playback speed (1.0 means normal speed)
        self.original_graph.setBackground("#ffffff")
        self.equalized_graph.setBackground("#ffffff")
        self.frequency_graph.setBackground("#ffffff")
        self.selected_mode = None
        self.syncing = True
        self.selected_window = None
        self.frame_layout = QHBoxLayout(self.sliders_frame)
        self.current_signal=None
        self.player = QMediaPlayer(None,QMediaPlayer.StreamPlayback) #instance for audio playback
        self.player.setVolume(50)
        #self.timer = QTimer(self) 
        self.timer = QtCore.QTimer(self) #updates position during audio playback
        self.elapsed_timer = QtCore.QElapsedTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.updatepos)
        self.line = pg.InfiniteLine(pos=0, angle=90, pen=None, movable=False)
        self.changed_orig = False
        self.changed_eq = False
        self.player.positionChanged.connect(self.updatepos)
        self.current_speed = 1
        self.slider_gain = {}
        self.equalized_bool = False
        self.time_eq_signal = SignalGenerator('EqSignalInTime')
        self.eqsignal = None
        self.sampling_rate = None
        self.line = pg.InfiniteLine(pos=0.1, angle=90, pen=None, movable=False)
        self.type = 'orig' 

        # Initialize your graphs and connect mouse events
        self.original_graph.setMouseTracking(True)
        self.equalized_graph.setMouseTracking(True)

        # Connect the mouse events to their handlers
        self.original_graph.mousePressEvent = self.mousePressEvent
        self.original_graph.mouseMoveEvent = self.mouseMoveEvent
        self.original_graph.mouseReleaseEvent = self.mouseReleaseEvent
        
        self.equalized_graph.mousePressEvent = self.mousePressEvent
        self.equalized_graph.mouseMoveEvent = self.mouseMoveEvent
        self.equalized_graph.mouseReleaseEvent = self.mouseReleaseEvent

        self.is_panning = False
        self.last_mouse_pos = None
        self.user_interacting = False

        #freq & spectrogram setup 
        self.available_palettes = ['twilight', 'Blues', 'Greys', 'ocean', 'nipy_spectral']
        self.current_color_palette = self.available_palettes[0]
        self.spectrogram_widget = {
            'before': self.spectrogram_before,
            'after': self.spectrogram_after
        }

        #UI conections
        self.modes_combobox.activated.connect(lambda: self.combobox_activated())
        self.load_btn.clicked.connect(lambda: self.load())
        self.hear_orig_btn.clicked.connect(lambda:self.playMusic('orig'))
        self.hear_eq_btn.clicked.connect(lambda:self.playMusic('equalized'))
        self.play_pause_btn.clicked.connect(lambda: self.play_pause()) 
        self.replay_btn.clicked.connect(lambda: self.replay())
        self.zoom_in_btn.clicked.connect(lambda: self.zoom_in())
        self.zoom_out_btn.clicked.connect(lambda: self.zoom_out())
        self.speed_up_btn.clicked.connect(lambda: self.speed_up()) 
        self.speed_down_btn.clicked.connect(lambda: self.speed_down())  
        self.checkBox.stateChanged.connect(lambda : self.hide())
        self.dictionary = {
            'Uniform Range':{},
            'Musical Instruments': {"Guitar": [40,400],
                                "Flute": [400, 800],
                                "Violin ": [950, 4000],
                                "Xylophone": [5000, 14000]
                                },
            "Animal Sounds":{"Dog": [0, 450],
                                "Wolf": [450, 1100],
                                "Crow": [1100, 3000],
                                "Bat": [3000, 9000]
                                },
            'ECG Abnormalities': {"Normal" : [0,35],
                                "Arrythmia_1 ": [48, 52],
                                "Arrythmia_2": [55, 94],
                                "Arrythmia_3": [95, 155]
                                }
        }

    def sync_range(self):
        range_ = self.original_graph.getViewBox().viewRange()
        self.equalized_graph.getViewBox().setXRange(*range_[0], padding=0)
        self.equalized_graph.getViewBox().setYRange(*range_[1], padding=0)


    def event(self, event):
        return super().event(event)

            
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_panning = True
            self.last_mouse_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.is_panning and self.last_mouse_pos is not None:
            delta = event.pos() - self.last_mouse_pos
            self.pan(delta.x(), delta.y())
            self.last_mouse_pos = event.pos()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_panning = False
            self.last_mouse_pos = None

    def load(self):
        path_info = QtWidgets.QFileDialog.getOpenFileName(
            None, "Select a signal...",os.getenv('HOME'), filter="Raw Data (*.csv *.wav *.mp3)")
        path = path_info[0] #actual file path is 1st element of tuple
        time = []
        self.equalized_bool = False #signal isn't equalized yet
        sample_rate = 0
        data = [] #empty list where signal data is to be stored later 

        signal_name = path.split('/')[-1].split('.')[0]   #get signal name from file path
        type = path.split('.')[-1] #get extension
        #check file type and load data accordingly

        #if it is an audio
        if type in ["wav", "mp3"]:
            data, sample_rate = librosa.load(path)
            Duration = librosa.get_duration(y=data, sr=sample_rate)
            self.duration = Duration
            time = np.linspace(0, Duration, len(data))
            self.audio_data = path

        #if it is a signal
        elif type == "csv":
            data_of_signal = pd.read_csv(path)  
            time = np.array(data_of_signal.iloc[:,0].astype(float).tolist()) #get time values from 1st column of CSV and put them into float array
            data = np.array(data_of_signal.iloc[:,1].astype(float).tolist()) #get signal data from 2nd column os CSV and put them into float array
            if len(time) > 1:
                sample_rate = 1 /( time[1]-time[0]) #calc sample rate
                #sample_rate = 1 /(time[1]-time[0])
            else:
                sample_rate=1

        #create SignalGenerator instance and set its attributes
        self.current_signal = SignalGenerator(signal_name)
        self.current_signal.data = data
        self.current_signal.time = time
        self.current_signal.sample_rate = sample_rate 
        self.sampling_rate = sample_rate


        #calc & set the FT of signal
        T = 1 / self.current_signal.sample_rate  #calc period
        x_data, y_data = self.get_Fourier(T, self.current_signal.data)  #x_data: freq , y_data: amp
        self.current_signal.freq_data = [x_data, y_data]

        #UNIFORM MODE:
        for i in range(10): #divide freq into 10 equal ranges
            self.batch_size = len(self.current_signal.freq_data[0])//10  #batch_sz = len(freqdata)/10
            self.dictionary['Uniform Range'][i] = [i*self.batch_size,(i+1)*self.batch_size]   #store ranges in dictionary

        self.frequency_graph.clear()
        if self.spectrogram_after.count() > 0:
            self.spectrogram_after.itemAt(0).widget().setParent(None) #remove canvas by setting parent -> None

        self.Plot("original")
        self.plot_spectrogram(data, sample_rate , self.spectrogram_before)
        self.frequency_graph.plot(self.current_signal.freq_data[0],
                    self.current_signal.freq_data[1],pen={'color': 'b'})
        self.eqsignal = copy.deepcopy(self.current_signal) #makes deep copy of current_signal and store it in eqsignal to preserve original signal for later processing


    def get_Fourier(self, T, data):
        N=len(data)  #bec FFT depends on #data_points in signal
        freq_amp= np.fft.fft(data) #freq_amp will contain real and img parts which will be used to get magnitude & phase
        self.current_signal.phase = np.angle(freq_amp[:N//2])  #store phase info for +ve freq

        #N -> #data_points in signal , T -> time interval between samples
        Freq= np.fft.fftfreq(N, T)[:N//2] #generate freq pin for each freq component val

        Amp = (2/N)*(np.abs(freq_amp[:N//2])) #store magnitude info for +ve freq
        return Freq, Amp
    

    def Range_spliting(self):
        freq = self.current_signal.freq_data[0]  #zero index for freq val
        if self.selected_mode == 'Uniform Range':
            self.current_signal.Ranges = [(i*self.batch_size,(i+1)*self.batch_size) for i in range(10)]  #divide range into 10 ranges
        else: 
            dict_ = self.dictionary[self.selected_mode] #get freq range for selected mode
            #calculate frequency indices for specified ranges
            for  _, (start,end) in dict_.items(): #key:_ , val : (s,e)
                start_ind = bisect.bisect_left(freq, start) #get index of 1st freq >= start_val
                end_ind = bisect.bisect_right(freq, end) - 1  #get index of 1st freq =< end_val
                self.current_signal.Ranges.append((start_ind, end_ind)) #append calculated range corresponding to specific freq. range in signal
        self.eqsignal.Ranges = copy.deepcopy(self.current_signal.Ranges) #get indpenedent copy of range and store it in processed signal


    def Plot(self, graph):
            signal= self.time_eq_signal if self.equalized_bool else self.current_signal #which signal is to be plotted
            if signal:
                #time domain 
                self.equalized_graph.clear()
                graphs = [self.original_graph, self.equalized_graph] #list containing two graphs (original - equalized) 
                graph = graphs[0] if graph == "original" else graphs[1]                 
                graph.clear()
                graph.setLabel('left', "Amplitude")
                graph.setLabel('bottom', "Time")
                plot_item = graph.plot(
                    signal.time, signal.data, name=f"{signal.name}") #plot signal using time & data values
                # Add legend to the graph
                if graph.plotItem.legend is not None:
                    graph.plotItem.legend.clear()
                legend = graph.addLegend()
                legend.addItem(plot_item, name=f"{signal.name}")

    def plot_freq(self):
        signal = self.eqsignal if self.equalized_bool else self.current_signal

        if signal and signal.Ranges: 
            _, end_last_ind = signal.Ranges[-1] #get end index of last frequency range to know when to stop plotting

            self.frequency_graph.clear()
            
            #plot original frequency data
            self.frequency_graph.plot(signal.freq_data[0][:end_last_ind],                   # array of freqs
                                    signal.freq_data[1][:end_last_ind], pen={'color': 'b'}) # array of corresponding magnitudes
            
            for i in range(len(signal.Ranges)):
                start_ind, end_ind = signal.Ranges[i]
                #add vertical lines for start & end of each range
                start_line = signal.freq_data[0][start_ind]
                end_line = signal.freq_data[0][end_ind - 1]
                v_line_start = pg.InfiniteLine(pos=start_line, angle=90, movable=False, pen=pg.mkPen('r', width=2))
                self.frequency_graph.addItem(v_line_start)
                v_line_end = pg.InfiniteLine(pos=end_line, angle=90, movable=False, pen=pg.mkPen('r', width=2))
                self.frequency_graph.addItem(v_line_end)

    def plot_spectrogram(self, samples, sampling_rate , widget):
        if widget.count() > 0:
            widget.itemAt(0).widget().setParent(None)  #if widget contains any items, clear existing spectrogram
        data = samples.astype('float32')
        n_fft=500 #size of fft = window length
        hop_length=320 #number of samples between fft windows
        #compute short-time FT magnitude squared
        frequency_magnitude = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)) ** 2
        #compute mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(S=frequency_magnitude, y=data, sr=sampling_rate, n_fft=n_fft,
                    hop_length=hop_length, win_length=n_fft, n_mels =128)
        #convert power spectrogram to dB
        decibel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)  #(Max power -> 0dB)
        time_axis = np.linspace(0, len(data) / sampling_rate)
        fig = Figure()
        fig = Figure(figsize=(3,3))
        ax = fig.add_subplot(111)
        # x-axis -> entire time, y-axis -> freq: 0 - Ts/2
        ax.imshow(decibel_spectrogram, aspect='auto', cmap='viridis',extent=[time_axis[0], time_axis[-1], 0, sampling_rate / 2])
        ax.axes.plot()
        canvas = FigureCanvas(fig)
        widget.addWidget(canvas)

    def playMusic(self, type):
        self.current_speed = 1
        self.line_position = 0
        self.player.setPlaybackRate(self.current_speed)
        media = QMediaContent(QUrl.fromLocalFile(self.audio_data))
        # Set the media content for the player and start playing
        self.player.setMedia(media)
        self.type = type
        if type == 'orig':
            sd.stop()
            self.timer.stop()
            self.changed_orig = True
            self.changed_eq = False
            # Create a QMediaContent object from the local audio file
            self.player.play()
            self.player.setVolume(100)
            # Add a vertical line to the original graph
            self.equalized_graph.removeItem(self.line)
            self.original_graph.addItem(self.line)
            self.timer.start()
        else:
            self.changed_eq = True
            self.changed_orig = False
            self.timer.start()
            self.player.play()
            self.player.setVolume(0)
            self.original_graph.removeItem(self.line)
            self.equalized_graph.addItem(self.line)
            sd.play(self.time_eq_signal.data, self.current_signal.sample_rate, blocking=False)
            self.player.play()
                        


    def updatepos(self):
            max_x = self.original_graph.getViewBox().viewRange()[0][1]
            graphs = [self.original_graph, self.equalized_graph]
            graph = graphs[0] if self.changed_orig  else graphs[1]
        # Get the current position in milliseconds
            position = self.player.position()/1000
            # Update the line position based on the current position
            self.line_position = position 
            max_x = graph.getViewBox().viewRange()[0][1]
            #print(position)
            if self.line_position > max_x:
                self.line_position = max_x
            #self.line_position = position
            self.line.setPos(self.line_position)
        
    def speed_up(self):
        # Increase the playback speed
        self.current_speed = self.current_speed * 2  # You can adjust the increment as needed
        self.player.setPlaybackRate(self.current_speed)
        if self.changed_eq :
            sd.play(self.time_eq_signal.data, self.current_signal.sample_rate, speed = self.current_speed, volume = 1.0 )
        #print(self.current_speed)

    def speed_down(self):
        # Decrease the playback speed
        self.current_speed = self.current_speed / 2  # You can adjust the increment as needed
        new_speed = max(0.1, self.current_speed / 2)  # Ensure speed doesn't go below 0.1
        self.player.setPlaybackRate(new_speed)
        if self.changed_eq :
            sd.play(self.time_eq_signal.data, self.current_signal.sample_rate, speed = self.current_speed, volume = 1.0 )
        #print(new_speed)

    def replay (self):
        self.playMusic('orig' if self.type == 'orig' else 'equalized')

    def play_pause(self):
        if self.is_playing:
            # Pause the audio
            self.audio_player.pause()  # Assuming you have an audio player object
            self.is_playing = False
            self.play_pause_button.setText("Play")  # Change the button text to "Play"
        else:
            # Start playing the audio
            self.audio_player.play()  # Assuming you have an audio player object
            self.is_playing = True
            self.play_pause_button.setText("Pause")  # Change the button text to "Pause"

    def combobox_activated(self):
        # Get the selected item's text and display it in the label
        selected_index = self.modes_combobox.currentIndex()
        self.selected_mode = self.modes_combobox.currentText()
        # store the mode in a global variable 
        self.add_slider()
        self.Range_spliting()

    def clear_layout(self ,layout):
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater() 

    def add_slider(self):          
        self.clear_layout(self.frame_layout) 
        dictionary = self.dictionary[self.selected_mode]
        for i,(key,_ )in enumerate(dictionary.items()):
            # print(f"Index: {i}, Key: {key}")
            label = QLabel(str(key))  # Create a label with a unique identifier
            slider_creator = Slider(i)
            slider = slider_creator.get_slider()
            self.slider_gain[i] = 10
            slider.valueChanged.connect(lambda value, i=i: self.update_slider_value(i, value/10))
            self.frame_layout.addWidget(slider)
            self.frame_layout.addWidget(label)
        
    def update_slider_value(self, slider_index, value):
        # This method will be called whenever a slider is moved
        self.slider_gain[slider_index] = value
        #print (self.slider_gain)
        self.equalized(slider_index, value)
        self.Plot('equalized')

    def zoom_in(self):
        for graph in [self.original_graph, self.equalized_graph]:
            graph.getViewBox().scaleBy((0.9, 0.9))
        self.sync_range()  # Sync ranges after zooming
        print('zoomed in')

    def zoom_out(self):
        for graph in [self.original_graph, self.equalized_graph]:
            graph.getViewBox().scaleBy((1.1, 1.1))
        self.sync_range()  # Sync ranges after zooming
        print('zoomed out')

    def pan(self, delta_x, delta_y):
        # Define a scaling factor for panning
        pan_scale = 0.005  # Adjust this value to change the sensitivity of panning

        # Apply the panning to both graphs using translateBy
        for graph in [self.original_graph, self.equalized_graph]:
            # Reverse the x direction
            graph.getViewBox().translateBy((-delta_x * pan_scale, delta_y * pan_scale))
        
        self.sync_range()  # Sync ranges after panning

    def on_user_interaction_start(self):
        self.user_interacting = True


    def equalized(self, slider_index, value):
        self.equalized_bool = True
        self.time_eq_signal.time = self.current_signal.time
        
        start, end = self.current_signal.Ranges[slider_index]
        
        # Get the original amplitude data
        Amp = np.array(self.current_signal.freq_data[1][start:end])
        
        # Scale the amplitude directly with the given value
        new_amp = Amp * value
        
        # Update the equalized signal's frequency data
        self.eqsignal.freq_data[1][start:end] = new_amp
        
        # Plot the frequency graph without smoothing
        self.plot_freq()  # Ensure this method does not involve any smoothing logic
        
        # Update the time equalized signal
        self.time_eq_signal.data = self.recovered_signal(self.eqsignal.freq_data[1], self.current_signal.phase)
        
        # Adjust the time signal length
        excess = len(self.time_eq_signal.time) - len(self.time_eq_signal.data)
        self.time_eq_signal.time = self.time_eq_signal.time[:-excess]
        
        # Plot the equalized signal
        self.Plot("equalized")
        
        # Plot the spectrogram
        self.plot_spectrogram(self.time_eq_signal.data, self.current_signal.sample_rate, self.spectrogram_after)
        
    def recovered_signal(self,Amp, phase):
        #complex array from amp and phase combination
        Amp = Amp * len(self.current_signal.data)/2 #N/2 as we get amp from fourier by multiplying it with fraction 2/N 
        complex_value = Amp * np.exp(1j*phase)
        # taking inverse fft to get recover signal
        recovered_signal = np.fft.irfft(complex_value)
        # taking only the real part of the signal
        return (recovered_signal)
    
    def hide(self):
        if (self.checkBox.isChecked()):
            self.specto_frame_before.hide()
            self.label_3.setVisible(False)
            self.specto_frame_after.hide()
            self.label_4.setVisible(False)
        else:
            self.specto_frame_before.show()
            self.label_3.setVisible(True)
            self.specto_frame_after.show()
            self.label_4.setVisible(True)
    
def main():
    app = QtWidgets.QApplication(sys.argv)
    main = EqualizerApp()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()