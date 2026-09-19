"""
radar_preprocessing.py
----------------------
Implements radar signal preprocessing: chirp generation, matched filtering,
and range-Doppler map formation.

Paper reference:
  Section III-A: Radar Signal Model
  Eq. (1): s(t) = exp(j2π(f_c t + 0.5 α t²))
  Eq. (2): y(t) = β s(t-τ) exp(j2π f_d t) + w(t)
"""

import numpy as np
import torch

class RadarPreprocessor:
    def __init__(self, fc=77e9, B=4e9, Tc=50e-6, fs=10e6, Nr=256, Nd=128):
        """
        Args:
            fc: carrier frequency (Hz)
            B: bandwidth (Hz)
            Tc: chirp duration (s)
            fs: sampling frequency (Hz)
            Nr: number of range bins
            Nd: number of Doppler bins
        """
        self.fc = fc
        self.B = B
        self.Tc = Tc
        self.fs = fs
        self.Nr = Nr
        self.Nd = Nd
        self.c = 3e8

    def generate_chirp(self, t=None):
        """Generate the transmitted chirp waveform (Eq. 1)."""
        if t is None:
            t = np.arange(0, self.Tc, 1/self.fs)
        alpha = self.B / self.Tc
        return np.exp(1j * 2 * np.pi * (self.fc * t + 0.5 * alpha * t**2))

    def simulate_received_signal(self, ranges, velocities, gains=None, noise_power=0.1):
        """
        Simulate received signal from multiple targets (Eq. 2).
        Args:
            ranges: list of target ranges (m)
            velocities: list of radial velocities (m/s)
            gains: complex path gains β for each target
            noise_power: variance of AWGN
        Returns:
            y: received signal (complex 1D array)
        """
        t = np.arange(0, self.Tc, 1/self.fs)
        s = self.generate_chirp(t)
        y = np.zeros_like(s, dtype=complex)
        if gains is None:
            gains = [1.0] * len(ranges)
        for R, v, beta in zip(ranges, velocities, gains):
            tau = 2 * R / self.c
            fd = 2 * v * self.fc / self.c
            # delayed and Doppler-shifted chirp
            s_delayed = self.generate_chirp(t - tau)
            y += beta * s_delayed * np.exp(1j * 2 * np.pi * fd * t)
        # add noise
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(t)) + 1j * np.random.randn(len(t)))
        y += noise
        return y

    def range_doppler_map(self, y):
        """
        Compute range-Doppler map from received signal.
        Uses 2D FFT after deramping (stretch processing) – simplified here.
        Returns:
            Y: complex range-Doppler map of shape (Nr, Nd)
        """
        # In practice: deramp, low-pass filter, sample, then 2D FFT.
        # Here we simulate a simple 2D FFT on reshaped data.
        # For demonstration, we assume y is already a 2D matrix of shape (Nr, Nd).
        if y.ndim == 1:
            # reshape into (Nr, Nd) – this is a placeholder.
            y = y[:self.Nr * self.Nd].reshape(self.Nr, self.Nd)
        Y = np.fft.fft2(y)
        return Y

    def __call__(self, raw_data):
        """Convert raw data to range-Doppler map."""
        if isinstance(raw_data, np.ndarray) and raw_data.ndim == 2:
            # already a range-Doppler map
            return raw_data
        else:
            y = self.simulate_received_signal(**raw_data)
            return self.range_doppler_map(y)
