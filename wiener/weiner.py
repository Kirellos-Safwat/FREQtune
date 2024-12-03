import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d

# Generate a noisy signal
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)
signal = np.sin(2 * np.pi * 50 * t)  # Clean signal (50 Hz sine wave)
noise = np.random.normal(0, 0.5, fs)  # Gaussian noise
observed = signal + noise  # Noisy observation

# Estimate power spectral densities using Welch's method
freqs_signal, psd_signal = welch(signal, fs=fs, nperseg=256)
freqs_noise, psd_noise = welch(noise, fs=fs, nperseg=256)

# Compute Wiener filter transfer function
H = psd_signal / (psd_signal + psd_noise)

# Interpolate H to match the frequencies of observed_fft
observed_fft = np.fft.rfft(observed)
freqs_observed = np.fft.rfftfreq(len(observed), 1 / fs)
H_interp = interp1d(freqs_signal, H, bounds_error=False,
                    fill_value="extrapolate")
H_matched = H_interp(freqs_observed)

# Apply Wiener filter in the frequency domain
filtered_fft = observed_fft * H_matched
filtered_signal = np.fft.irfft(filtered_fft)

# Plot results
plt.figure()
plt.plot(t, observed, label="Noisy Signal")
plt.plot(t, filtered_signal, label="Filtered Signal", alpha=0.7)
plt.legend()
plt.title("Wiener Filtered Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()
