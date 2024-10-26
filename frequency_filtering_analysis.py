import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, ifft, fftfreq

# Parameters
fc = 500   # Frequency of the cosine (500 Hz)
fs = 10000 # Sampling rate (10 kHz to capture all details)
t = np.arange(-0.01, 0.01, 1/fs) # Time vector

# Signal generation
x_t = (np.sinc(1000 * t)) ** 2
z_t = x_t * np.cos(2 * np.pi * fc * t)

# Fourier Transform and Frequency Vector
Z_f = fftshift(fft(z_t))
freqs = fftshift(fftfreq(len(t), 1/fs))

# Low-pass filter (Passes frequencies below 0.5 kHz)
W = 500
H_f = np.abs(freqs) < W
Y_f = Z_f * H_f

# Inverse Fourier Transform to get the filtered signal in time domain
y_t = np.real(ifft(fftshift(Y_f)))

# Plotting results

# Spectrum of X(f) and Z(f)
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(freqs, np.abs(fftshift(fft(x_t))), label="|X(f)|")
plt.title("Spectrum of X(f)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(freqs, np.abs(Z_f), label="|Z(f)|", color="orange")
plt.title("Spectrum of Z(f)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(freqs, np.abs(Y_f), label="|Y(f)|", color="green")
plt.title("Spectrum of Y(f)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.tight_layout()
plt.show()

# Time domain signal y(t)
plt.figure(figsize=(10, 4))
plt.plot(t, y_t, label="y(t)", color="purple")
plt.title("Filtered Signal in Time Domain y(t)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()
