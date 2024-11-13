import sounddevice as sd  # for handling audio feedback
import librosa  # for audio processing
from PyQt5.QtCore import Qt
from Slider import Slider
from Signal import SignalGenerator
from matplotlib.figure import Figure
from scipy import signal as sg
import bisect
from scipy.fft import fft  # four fourier transform
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import pandas as pd
import copy
from PyQt5.QtWidgets import QHBoxLayout, QLabel
import matplotlib as plt
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, uic
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import os
import sys
plt.use('Qt5Agg')


class EqualizerApp(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(EqualizerApp, self).__init__(*args, **kwargs)
        uic.loadUi(r'task3.ui', self)
        self.is_playing = False  #keeps track of whether audio is playing/paused
        self.playback_speed = 1.0 #normal playback speed is default
        self.original_graph.setBackground("black")
        self.equalized_graph.setBackground("black")
        self.frequency_graph.setBackground("black")

        self.original_graph.getPlotItem().showGrid(x=True, y=True)
        self.equalized_graph.getPlotItem().showGrid(x=True, y=True)
        self.frequency_graph.getPlotItem().showGrid(x=True, y=True)

        self.selected_mode = None
        self.syncing = True  #for syncing graphs together
        self.selected_window = None
        self.frame_layout = QHBoxLayout(self.sliders_frame)
        self.current_signal = None
        self.linear_frequency_scale = True #toggle between linear freq scale & audiogram
        #instance for audio playback
        self.player = QMediaPlayer(None, QMediaPlayer.StreamPlayback)
        self.player.setVolume(50)
        self.timer = QtCore.QTimer(self)
        self.elapsed_timer = QtCore.QElapsedTimer()
        self.timer.setInterval(100)  #0.1 sec
        self.timer.timeout.connect(self.updatepos)  #calling update_pos each 0.1sec
        self.line = pg.InfiniteLine(pos=0, angle=90, pen=None, movable=False) #to show playback position in original and equalized graphs
        self.changed_orig = False
        self.changed_eq = False
        self.player.positionChanged.connect(self.updatepos)
        self.current_speed = 1
        self.slider_gain = {} #store gain adjustment for each slider
        self.equalized_bool = False
        self.time_eq_signal = SignalGenerator('EqSignalInTime')
        self.eqsignal = None
        self.line = pg.InfiniteLine(pos=0.1, angle=90, pen=None, movable=False)
        self.type = 'orig'
        self.is_panning = False
        self.last_mouse_pos = None
        self.user_interacting = False

        #initializing graphs & connecting mouse events
        self.original_graph.setMouseTracking(True)
        self.equalized_graph.setMouseTracking(True)

        #overriding mouse wheel events
        self.original_graph.wheelEvent = self.zoom_graph
        self.equalized_graph.wheelEvent = self.zoom_graph

        #connecting mouse events to handlers
        self.original_graph.mousePressEvent = self.mousePressEvent
        self.original_graph.mouseMoveEvent = self.mouseMoveEvent
        self.original_graph.mouseReleaseEvent = self.mouseReleaseEvent

        self.equalized_graph.mousePressEvent = self.mousePressEvent
        self.equalized_graph.mouseMoveEvent = self.mouseMoveEvent
        self.equalized_graph.mouseReleaseEvent = self.mouseReleaseEvent

        # freq & spectrogram setup
        self.available_palettes = ['twilight',
                                   'Blues', 'Greys', 'ocean', 'nipy_spectral']
        self.current_color_palette = self.available_palettes[2]
        self.spectrogram_widget = {
            'before': self.spectrogram_before,
            'after': self.spectrogram_after
        }

        self.freq_radio.toggled.connect(self.toggle_freq)
        self.audio_radio.toggled.connect(self.toggle_freq)

        #UI conectionsss:
        self.modes_combobox.activated.connect(
            lambda: self.combobox_activated())
        self.load_btn.clicked.connect(lambda: self.load())
        self.hear_orig_btn.clicked.connect(lambda: self.playMusic('orig'))
        self.hear_eq_btn.clicked.connect(lambda: self.playMusic('equalized'))
        self.play_pause_btn.clicked.connect(lambda: self.play_pause())
        self.replay_btn.clicked.connect(lambda: self.replay())
        self.zoom_in_btn.clicked.connect(lambda: self.zoom_in())
        self.zoom_out_btn.clicked.connect(lambda: self.zoom_out())
        self.speed_slider.valueChanged.connect(
            lambda: self.update_speed(self.speed_slider.value()))
        self.checkBox.stateChanged.connect(lambda: self.hide())
        self.dictionary = {
            'Uniform Range': {},
            'Musical Instruments': {"Guitar": [(40, 400)],
                                    "Flute": [(400, 800)],
                                    "Violin ": [(950, 4000)],
                                    "Xylophone": [(5000, 14000)]
                                    },
            "Animal Sounds": {"Dog": [(0, 450)],
                              "Wolf": [(450, 1100)],
                              "Crow": [(1100, 3000)],
                              "Bat": [(3000, 9000)]
                              },
            'ECG Abnormalities': {"Normal": [(0, 35)],
                                  "Arrythmia_1 ": [(48, 52)],
                                  "Arrythmia_2": [(55, 94)],
                                  "Arrythmia_3": [(95, 155)]
                                  }
        }

    def zoom_graph(self, event):
        if self.current_signal is None:
            return
        zoom_factor = 1.1
        delta = event.angleDelta().y()  #get wheel movement direction

        #determine whether to zoom in or out
        if delta > 0:
            scale_factor = 1 / zoom_factor  #zoom in
        else:
            scale_factor = 0.5 #zoom out
            original_view_range = self.original_graph.getViewBox().viewRange() #return current view range of x,y limits as follows:
            current_x_min, current_x_max = original_view_range[0]
            current_y_min, current_y_max = original_view_range[1]

            if len(self.current_signal.time) == 0:
                return  #if signal doesn't have any time data, do nothing

            #set signal length to last time point
            signal_length = self.current_signal.time[-1]

            signal_y_min = np.min(self.current_signal.data)
            signal_y_max = np.max(self.current_signal.data)

            new_x_min = current_x_min - (current_x_min * scale_factor)
            new_x_max = current_x_max - (current_x_max * scale_factor)

            new_y_min = current_y_min + (current_y_min * scale_factor)
            new_y_max = current_y_max + (current_y_max * scale_factor)

            if new_x_min < 0 or new_x_max > signal_length:
                return

            if new_y_min < signal_y_min:
                new_y_min = signal_y_min
            if new_y_max > signal_y_max:
                new_y_max = signal_y_max

            self.original_graph.getViewBox().setYRange(new_y_min, new_y_max)

            scale_factor = zoom_factor  

        #apply scaling with zoom_factor to both graphs
        self.original_graph.getViewBox().scaleBy((scale_factor, scale_factor))
        self.equalized_graph.getViewBox().scaleBy((scale_factor, scale_factor))

        self.sync_range()

    def sync_range(self): #syncs any changes for both original and equalized graphs
        range_ = self.original_graph.getViewBox().viewRange()
        self.equalized_graph.getViewBox().setXRange(*range_[0], padding=0)
        self.equalized_graph.getViewBox().setYRange(*range_[1], padding=0)

    def event(self, event): # to detect mouse press / release / move for panning and zooming
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
            None, "Select a signal...", os.getenv('HOME'), filter="Raw Data (*.csv *.wav *.mp3)")
        path = path_info[0]  # actual file path is 1st element of tuple

        self.equalized_bool = False  # signal isn't equalized yet
        sample_rate = 0
        data = []  # empty list where signal data is to be stored later

        # get signal name from file path
        signal_name = path.split('/')[-1].split('.')[0]
        type_ = path.split('.')[-1]  # get extension
        # check file type and load data accordingly

        # if it is an audio
        if type_ in ["wav", "mp3"]:
            data, sample_rate = librosa.load(path)
            Duration = librosa.get_duration(y=data, sr=sample_rate)
            time = np.linspace(0, Duration, len(data))
            self.audio_path = path

        # elif it is a signal (ECG)    
        elif type_ == "csv":
            signal_data = pd.read_csv(path)
            time = np.array(signal_data.iloc[:, 0].astype(float).tolist())
            data = np.array(signal_data.iloc[:, 1].astype(float).tolist())
            if len(time) > 1:
                sample_rate = 1 / (time[1]-time[0])
            else:
                sample_rate = 1

        # create "Signal" instance and set its attributes
        self.current_signal = SignalGenerator(signal_name, data=data,
                                              time=time, sample_rate=sample_rate)

        # calc & set the FT of signal
        T = 1 / sample_rate  # calc period
        frequency_axis, amplitude_axis = self.get_Fourier(T, data)
        self.current_signal.freq_data = [frequency_axis, amplitude_axis]

        # UNIFORM MODE:
        self.batch_size = len(frequency_axis)//10
        for i in range(10):  # divide freq into 10 equal ranges
            self.dictionary['Uniform Range'][i] = [
                i*self.batch_size, (i+1)*self.batch_size]  # store ranges in dictionary

        self.frequency_graph.clear()
        if self.spectrogram_after.count() > 0:
            self.spectrogram_after.itemAt(0).widget().setParent(
                None)  # remove canvas by setting parent -> None

        self.Plot("original")
        self.plot_spectrogram(data, sample_rate, self.spectrogram_before)
        self.frequency_graph.plot(
            frequency_axis, amplitude_axis, pen={'color': 'b'})

        # makes deep copy of current_signal and store it in eqsignal to preserve original signal for later processing
        self.eqsignal = copy.deepcopy(self.current_signal)

        self.combobox_activated()

    def get_Fourier(self, T, data):
        N = len(data)  # bec FFT depends on #data_points in signal
        # freq_amp will contain real and img parts which will be used to get magnitude & phase
        freq_amp = np.fft.fft(data)
        self.current_signal.phase = np.angle(
            freq_amp[:N//2])  # store phase info for +ve freq

        # N -> #data_points in signal , T -> time interval between samples
        # generate freq pin for each freq component val
        Freq = np.fft.fftfreq(N, T)[:N//2]

        # 2/N to normalize the amplitude of the FFT
        Amp = (2/N)*(np.abs(freq_amp[:N//2]))
        return Freq, Amp

    def Range_spliting(self):
        if self.current_signal is None:
            return
        freq = self.current_signal.freq_data[0]  #zero index for freq val
        #IF UNIFORM:
        if self.selected_mode == 'Uniform Range':
            self.current_signal.Ranges = [(i*self.batch_size,(i+1)*self.batch_size) for i in range(10)]  #divide range into 10 ranges
        #IF NOT UNIFORM
        else: 
            dict_ = self.dictionary[self.selected_mode] #get freq range for selected mode
            self.current_signal.Ranges={}
            #calculate frequency indices for specified ranges
            new_dict = {i: value for i, (key, value) in enumerate(dict_.items())}
            for _, range_ in new_dict.items(): #_ : key, range_ : val
                #Loop through subrange in range for non-cont ranges
                for subrange in range_:
                    start, end = subrange
                    # get index of 1st freq >= start_val
                    start_ind = bisect.bisect_left(freq, start)
                    # get index of 1st freq <= end_val
                    end_ind = bisect.bisect_right(freq, end) - 1
                    if _ not in self.current_signal.Ranges:
                        self.current_signal.Ranges[_] = [(start_ind, end_ind)] #if key is not in Ranges, initialize list containing tuple (start_ind, end_ind) for that key
                    else:
                        self.current_signal.Ranges[_].append((start_ind,end_ind)) #if key is in Ranges, append(start_ind, end_ind) tuple to existing list for that key
                    
        print(self.current_signal.Ranges)
        self.eqsignal.Ranges = copy.deepcopy(self.current_signal.Ranges)

    def Plot(self, graph):
        signal = self.time_eq_signal if self.equalized_bool else self.current_signal#determine which signal to plot
        if signal:
            # time domain
            self.equalized_graph.clear()
            graph = self.original_graph if graph == "original" else self.equalized_graph
            graph.clear()
            graph.setLabel('left', "Amplitude")
            graph.setLabel('bottom', "Time")
            plot_item = graph.plot(
                signal.time, signal.data, name=f"{signal.name}", pen={'color': '#3D8262'})
            # add legend to graph
            if graph.plotItem.legend is not None:
                graph.plotItem.legend.clear()
            legend = graph.addLegend()
            legend.addItem(plot_item, name=f"{signal.name}")

    def plot_freq(self):
        signal = self.eqsignal if self.equalized_bool else self.current_signal
        if signal and signal.Ranges:  # Check if signal is not None and signal.Ranges is not empty
            # get end index of last frequency range to know when to stop plotting
            if self.selected_mode != 'Uniform Range':
                _, end_last_ind = signal.Ranges[0][0][0], signal.Ranges[3][-1][1]
            else:
                _, end_last_ind = signal.Ranges[-1]


            self.frequency_graph.setLabel('bottom', 'Log(Frequency)', units='Hz')
            self.frequency_graph.setLabel('left', 'Magnitude', units='dB')

            if not self.linear_frequency_scale:  
                self.frequency_graph.clear()
                self.frequency_graph.setLogMode(x=True, y=False)

                ticks = [[(np.log10(250), '250'), (np.log10(500), '500'), (np.log10(1000), '1k'), (np.log10(2000), '2k'), (np.log10(4000), '4k')]]
                self.frequency_graph.getAxis('bottom').setTicks(ticks)

                # plot original frequency data
                self.frequency_graph.plot(signal.freq_data[0][:end_last_ind],                   # array of freqs
                                          signal.freq_data[1][:end_last_ind], pen={'color': 'r'})  # array of corresponding magnitudes

            else:  # freq domain
                self.frequency_graph.clear()
                self.frequency_graph.getAxis('bottom').setTicks(None)
                self.frequency_graph.setLogMode(x=False, y=False)
                self.frequency_graph.plot(signal.freq_data[0][:end_last_ind],              # array of freqs
                                          signal.freq_data[1][:end_last_ind], pen={'color': 'b'})
        
        def plot_ranges(start, end, color):
            # Add vertical lines for the start and end of the range
            start_line = np.log10(signal.freq_data[0][start] + 1e-10) if not self.linear_frequency_scale else signal.freq_data[0][start]
            end_line = np.log10(signal.freq_data[0][end - 1] + 1e-10) if not self.linear_frequency_scale else signal.freq_data[0][end - 1]
            v_line_start = pg.InfiniteLine(
                pos=start_line, angle=90, movable=False, pen=pg.mkPen(color, width=2, style=Qt.DashLine))
            self.frequency_graph.addItem(v_line_start)
            v_line_end = pg.InfiniteLine(
                pos=end_line, angle=90, movable=False, pen=pg.mkPen(color, width=2, style=Qt.DashLine))
            self.frequency_graph.addItem(v_line_end)
            
        for i in range(len(signal.Ranges)):
            if self.selected_mode != 'Uniform Range':
                for range_ in signal.Ranges[i]:
                    plot_ranges(*range_, color='y')
            else:
                plot_ranges(*signal.Ranges[i], color='y')



    def toggle_freq(self):
        if self.freq_radio.isChecked():
            self.linear_frequency_scale = False
        elif self.audio_radio.isChecked():
            self.linear_frequency_scale = True
        self.plot_freq()


    def plot_spectrogram(self, samples, sampling_rate, widget):
        if widget.count() > 0:
            # if widget contains any items, clear existing spectrogram
            widget.itemAt(0).widget().setParent(None)
        data = samples.astype('float32')
        n_fft = 500  # size of fft = window length
        hop_length = 320  # number of samples between fft windows
        # compute short-time FT magnitude squared
        frequency_magnitude = np.abs(librosa.stft(
            data, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)) ** 2
        # compute mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(S=frequency_magnitude, y=data, sr=sampling_rate, n_fft=n_fft,
                                                         hop_length=hop_length, win_length=n_fft, n_mels=128)
        # convert power spectrogram to dB
        decibel_spectrogram = librosa.power_to_db(
            mel_spectrogram, ref=np.max)  # (Max power -> 0dB)
        time_axis = np.linspace(0, len(data) / sampling_rate)
        fig = Figure()
        fig = Figure(figsize=(3, 3))
        fig.patch.set_alpha(0)

        ax = fig.add_subplot(111)
        ax.set_facecolor("none")
        ax.tick_params(colors='gray')
        # x-axis -> entire time, y-axis -> freq: 0 - Ts/2
        ax.imshow(decibel_spectrogram, aspect='auto', cmap='viridis',
                  extent=[time_axis[0], time_axis[-1], 0, sampling_rate / 2])
        ax.axes.plot()

        canvas = FigureCanvas(fig)
        widget.addWidget(canvas)


    def playMusic(self, type_):
        self.current_speed = self.speed_slider.value() #get speed value from slide
        print(f"speed from slider: {self.current_speed}")

        self.update_speed(self.current_speed)
        print(f"updated speeed: {self.current_speed}")

        if self.current_speed == 0:
            self.current_speed = 1 # to avoid silent playback
            
        print(f"current speeed: {self.current_speed}")
        self.line_position = 0 #vertical line for tracking playbcak position at grpahs
        self.player.setPlaybackRate(self.current_speed)
        print(f"{self.current_speed} for playmusic")

        #set playback properties
        self.type = type_
        self.is_playing = True
        self.play_pause_btn.setText("Pause")

        if type_ == 'orig': #if original graph
            sd.stop()

            #set media content for  player and start playing
            media = QMediaContent(QUrl.fromLocalFile(self.audio_path))
            self.player.setMedia(media)
            self.player.play()
            self.player.setVolume(100)

            self.changed_orig = True
            self.changed_eq = False

            #add a vertical line to the original graph and remove it from eq graph
            self.equalized_graph.removeItem(self.line)
            self.original_graph.addItem(self.line)

        else: #if equalized graph
            #stop original audio if it's playing by setting volume = 0
            self.player.play()
            self.player.setVolume(0)

            #update the graph and timer for the equalized sound
            self.changed_eq = True
            self.changed_orig = False

            #add vertical line to equalized graph and remove it from original one
            self.original_graph.removeItem(self.line)
            self.equalized_graph.addItem(self.line)

            sd.play(self.time_eq_signal.data,
                    self.current_signal.sample_rate, blocking=False)

        self.timer.start()

    def updatepos(self):
        max_x = self.original_graph.getViewBox().viewRange()[0][1] #get end of visible playback time
        graphs = [self.original_graph, self.equalized_graph]
        graph = graphs[0] if self.changed_orig else graphs[1]
        #get current position in ms
        position = self.player.position()/1000
        #update line position based on current position
        self.line_position = position
        max_x = graph.getViewBox().viewRange()[0][1]
        if self.line_position > max_x:
            self.line_position = max_x
        self.line_position = position
        self.line.setPos(self.line_position)

    def update_speed(self, direction):
        #adjust playback speed, ensuring it remains above 0.1x
        print(f" speed before update_speed {self.current_speed}")
        if direction == 0:
            self.current_speed = 1
        else:
            self.current_speed =  direction+1 if direction >0 else np.abs(direction)*0.1
            print(f" test speed after update_speed {self.current_speed}")
        #stop current playback to apply new speed
        if self.changed_eq:
            sd.stop()
            #calc new sampling rate based on current speed
            adjusted_samplerate = int(
                self.current_signal.sample_rate * self.current_speed)

            #calc starting sample based on current playback position
            start_sample = int(self.line_position *
                               self.current_signal.sample_rate)

            #play equalized audio at adjusted sample rate for speed control
            sd.play(
                self.time_eq_signal.data[start_sample:], samplerate=adjusted_samplerate)
        else:
            #for original audio, apply speed adjustment if necessary
            self.player.setPlaybackRate(self.current_speed)

    def replay(self):
        #restart playback according to current type
        self.playMusic(self.type)

    def play_pause(self):
        if self.is_playing:
            #pause currently playing sound based on type
            if self.type == 'orig':
                #pause original audio
                self.player.pause()
            else:
                #pause equalized audio 
                self.player.pause()
                #(stop and store position)
                sd.stop()
                self.equalized_position = self.line_position

            #update play/pause state
            self.is_playing = False
            self.play_pause_btn.setText("Play")
        else:
            #resume currently paused sound based on type
            if self.type == 'orig':
                #resume original audio
                self.player.play()
                self.player.setPlaybackRate(
                    self.current_speed)  #apply speed setting

            else:
                #resume equalized audio from stored position
                adjusted_samplerate = int(
                    self.current_signal.sample_rate * self.current_speed)

                start_sample = int(
                    self.equalized_position * self.current_signal.sample_rate / self.current_speed)
                sd.stop()
                sd.play(self.time_eq_signal.data[start_sample:],
                        samplerate=adjusted_samplerate, blocking=False)
                self.player.play()

            #update play/pause state
            self.is_playing = True
            self.play_pause_btn.setText("Pause")

    def on_media_finished(self):
        #reset button when media finishes
        if self.player.mediaStatus() == QMediaPlayer.EndOfMedia:
            self.is_playing = False
            self.play_pause_btn.setText("Play")
            self.timer.stop()  #stop updating position

    def combobox_activated(self):
        #get the selected item's text and display it in the label
        self.selected_mode = self.modes_combobox.currentText()
        # store the mode in a global variable
        self.add_slider()
        self.Range_spliting()

    def clear_layout(self, layout):
        for i in reversed(range(layout.count())):
            item = layout.itemAt(i)
            if item.widget():
                item.widget().deleteLater()

    def add_slider(self):
        self.clear_layout(self.frame_layout)
        dictionary = self.dictionary[self.selected_mode]
        for i, (key, _) in enumerate(dictionary.items()):
            label = QLabel(str(key))  #create a label with a unique identifier
            slider_creator = Slider(i)
            slider = slider_creator.get_slider()
            self.slider_gain[i] = 10
            slider.valueChanged.connect(
                lambda value, i=i: self.update_slider_value(i, value/10))
            self.frame_layout.addWidget(slider)
            self.frame_layout.addWidget(label)

    def update_slider_value(self, slider_index, value):
        self.slider_gain[slider_index] = value
        self.equalized(slider_index, value)
        self.Plot('equalized')
        self.plot_freq()

    def zoom_in(self):
        for graph in [self.original_graph, self.equalized_graph]:
            graph.getViewBox().scaleBy((0.9, 0.9))
        self.sync_range() 

    def zoom_out(self):
        if self.current_signal is None:
            return
        scale_factor = 0.5

        #needs more coding to handle not zooming out outside range case
        original_view_range = self.original_graph.getViewBox().viewRange()

        #current x-axis range (min and max)
        current_x_min, current_x_max = original_view_range[0]
        current_y_min, current_y_max = original_view_range[1]

        if len(self.current_signal.time) == 0:
            return  #if signal doesn't have any time data, do nothing

        #end of the signal in seconds
        signal_length = self.current_signal.time[-1]

        signal_y_min = np.min(self.current_signal.data)
        signal_y_max = np.max(self.current_signal.data)

        new_x_min = current_x_min - (current_x_min * scale_factor)
        new_x_max = current_x_max - (current_x_max * scale_factor)

        new_y_min = current_y_min + (current_y_min * scale_factor)
        new_y_max = current_y_max + (current_y_max * scale_factor)

        if new_x_min < 0 or new_x_max > signal_length:
            return

        if new_y_min < signal_y_min:
            new_y_min = signal_y_min
        if new_y_max > signal_y_max:
            new_y_max = signal_y_max

        self.original_graph.getViewBox().setYRange(new_y_min, new_y_max)

        for graph in [self.original_graph, self.equalized_graph]:
            graph.getViewBox().scaleBy((1.1, 1.1))
        self.sync_range() 

    def pan(self, delta_x, delta_y):
        if self.current_signal is None:
            return  #don't do anything if current_signal is not set

        #define a scaling factor for panning sensitivity
        pan_scale = 0.005  

        #get current visible range of original graph
        original_view_range = self.original_graph.getViewBox().viewRange()

        #current x-axis range (min and max)
        current_x_min, current_x_max = original_view_range[0]
        current_y_min, current_y_max = original_view_range[1]

        if len(self.current_signal.time) == 0:
            return  #if signal doesn't have any time data, do nothing

        #end of the signal in seconds
        signal_length = self.current_signal.time[-1]

        signal_y_min = np.min(self.current_signal.data)
        signal_y_max = np.max(self.current_signal.data)

        #calc new x-axis range after applying panning (delta_x)
        new_x_min = current_x_min - (delta_x * pan_scale)
        new_x_max = current_x_max - (delta_x * pan_scale)

        #calc new y-axis range after applying panning (delta_y)
        new_y_min = current_y_min - (delta_y * pan_scale)
        new_y_max = current_y_max - (delta_y * pan_scale)

        if new_x_min < 0 or new_x_max > signal_length:
            return  # don't pan

        if new_y_min < signal_y_min:
            new_y_min = signal_y_min
        if new_y_max > signal_y_max:
            new_y_max = signal_y_max

        self.original_graph.getViewBox().setYRange(new_y_min, new_y_max)

        #apply panning to both graphs in synced manner using translateBy
        for graph in [self.original_graph, self.equalized_graph]:
            #reverse x direction
            graph.getViewBox().translateBy((-delta_x * pan_scale, delta_y * pan_scale))
        self.sync_range()  

    def on_user_interaction_start(self):
        self.user_interacting = True

    def equalized(self, slider_index, value):
        if self.current_signal is None:
            return
        self.equalized_bool = True
        self.time_eq_signal.time = self.current_signal.time
        if self.selected_mode != 'Uniform Range':
            for subrange in self.current_signal.Ranges[slider_index]:
                start, end = subrange

                #get original amplitude data
                Amp = np.array(self.current_signal.freq_data[1][start:end])

                #scale amplitude directly with given value
                new_amp = Amp * value

                #update equalized signal's freq data
                self.eqsignal.freq_data[1][start:end] = new_amp
        else:
            start, end = self.current_signal.Ranges[slider_index]

            #get original amplitude data
            Amp = np.array(self.current_signal.freq_data[1][start:end])

            #scale the amplitude directly with the given value
            new_amp = Amp * value

            #update the equalized signal's frequency data
            self.eqsignal.freq_data[1][start:end] = new_amp

        
        self.time_eq_signal.data = self.recovered_signal(   #update time equalized signal
            self.eqsignal.freq_data[1], self.current_signal.phase)

        
        excess = len(self.time_eq_signal.time) - len(self.time_eq_signal.data) #adjust time signal length
        self.time_eq_signal.time = self.time_eq_signal.time[:-excess]

        self.Plot("equalized")  #plot equalized signal


        self.plot_spectrogram(
            self.time_eq_signal.data, self.current_signal.sample_rate, self.spectrogram_after)

    def recovered_signal(self, Amp, phase):
        # complex array from amp and phase combination
        # N/2 as we get amp from fourier by multiplying it with fraction 2/N
        Amp = Amp * len(self.current_signal.data)/2
        complex_value = Amp * np.exp(1j*phase)
        # taking inverse fft to get recover signal
        recovered_signal = np.fft.irfft(complex_value)
        return (recovered_signal)

    def hide(self): # show/hide spectograms
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
