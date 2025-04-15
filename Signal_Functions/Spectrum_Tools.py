'''
These functions provide multiple methods for cross-power specturm analysis.
Includes:
1. Simple FFT analysis
2. Slide window FFT analysis
3. Cross-FFT-power spectrum
4. Cross Wavelet power spectrum
5. Hilbert-Transform-Based Phase Locking Value(PLV) score connectome (DOI:10.3389/fnins.2018.01037) (New method, a little advance.)
'''

#%%
# FUNCTION 1: Direct FFT
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  


def FFT_Spectrum(series, fps,ticks = 0.01,plot = False,normalize = True):
    """
    Compute Power spectrum of given series, note that it will return power density and raw power at the same time.

    Parameters:
        series (array-like): Input time series data.
        fps (float): Sampling frequency (frames per second).
        ticks (float): Ticks of power spectrum.
        plot (Bool): Whether Plot power spectrum result.

    Returns:
        freq_ticks (ndarray): Array of frequencies.
        binned_power (ndarray): Power at each frequency band.
        freqs_raw (ndarray): Raw freq tick, this will be affected by series length.
        power_raw (ndarray): Raw power of FFT results, not normalized.
        total_power (float): Full power of current 
    """
    n = len(series)  # Number of data points

    # Compute the FFT
    fft_result = np.fft.fft(series)
    freqs_raw = np.fft.fftfreq(n, d=1/fps)  # Frequency bins

    # Compute the power spectral density (PSD)
    power_raw = np.abs(fft_result) ** 2 / (fps * n)  # Normalized PSD
    power_raw = power_raw[:n // 2]  # Keep only positive frequencies
    freqs_raw = freqs_raw[:n // 2]  # Keep only positive frequencies

    # Normalize the PSD so that the total power sums to 1
    total_power = np.sum(power_raw)
    if normalize == True:
        power_density = power_raw / total_power
    else:
        power_density = power_raw


    # Bin the power density
    bin_edges = np.arange(0, freqs_raw[-1] + ticks, ticks)
    freq_ticks = (bin_edges[:-1] + bin_edges[1:]) / 2  # Center of each bin
    binned_power = np.zeros_like(freq_ticks)
    # sum power inside given bin band.
    for i in range(len(bin_edges) - 1):
        # Find frequencies within the current bin
        mask = (freqs_raw >= bin_edges[i]) & (freqs_raw < bin_edges[i + 1])
        binned_power[i] = np.sum(power_density[mask])

    # plot if required.
    if plot == True:
        plt.figure(figsize=(6,6))
        # plt.plot(freqs, power_density, label="Power Spectrum Raw")
        plt.bar(freq_ticks, binned_power, width=ticks, align='center', alpha=0.7, label="Binned PSD")
        # plt.plot(freq_ticks, binned_power, alpha=0.7, label="Binned PSD")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Density")
        plt.title(f"Binned Power Spectral Density (Bin Width = {ticks} Hz)")
        # plt.grid(True)
        plt.legend()
        plt.show()

    return freq_ticks,binned_power,freqs_raw,power_raw,total_power


#%% FUNCTION 2, slide window FFT
def FFT_Spectrum_Slide(series,win_size,win_step,fps,ticks=0.01,normalize = True):

    winnum = (len(series)-win_size)//win_step+1
    for i in tqdm(range(winnum)):
        c_series = series[i*win_step:i*win_step+win_size]
        freq_ticks,c_spectrum,_,_,c_power = FFT_Spectrum(c_series,fps,ticks)
        # initialize pd frame.
        if i == 0:
            power_spectrum = pd.DataFrame(0.0,columns = range(winnum),index = freq_ticks.round(3))
        if normalize == True:
            power_spectrum.loc[:,i] = c_spectrum
        else:
            power_spectrum.loc[:,i] = c_spectrum*c_power
    return power_spectrum

#%% FUNCTION 3, simple wavelet analysis
def Wavelet_Coherent(signal,fps,freqs):
    """
    Perform wavelet analysis on a signal and return a DataFrame with frequency bands and time.
    kernel uses cmor1.5-1.0 wavelet, 

    Parameters:
    - fps: int, frames per second (sampling rate) of the signal.
    - signal: array-like, the input signal.
    - freqs: nd array, The frequency you want. Wavelet will be done only on these freqs.

    Returns:
    - df: pandas.DataFrame, DataFrame with frequency bands as index and time as columns.
    """
    # Calculate the scales for the wavelet transform
    scales = fps/freqs
    
    # Perform the Continuous Wavelet Transform (CWT)
    coefficients, frequencies = pywt.cwt(signal, scales, "cmor1.5-1.0", sampling_period=1.0/fps)
    


    # Create a DataFrame with frequency bands as index and time as columns
    df = pd.DataFrame(np.abs(coefficients)**2, index=frequencies, columns=range(len(signal)))
    
    return df