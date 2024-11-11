class SignalGenerator:
    def __init__(self, name, data=[], time=[], sample_rate=None, freq_data=None, Ranges=[], phase=None):
        self.name = name
        self.data = data
        self.time = time
        self.sample_rate = sample_rate
        self.freq_data = freq_data
        self.Ranges = Ranges
        self.phase = phase