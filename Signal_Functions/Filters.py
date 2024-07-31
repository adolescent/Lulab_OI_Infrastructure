'''
Include multiple functions of signal processing.

'''
from scipy import signal

def Signal_Filter_1D(series,HP_freq,LP_freq,fps,keep_DC = True,order = 5):
    DC_power = float(series.mean())
    nyquist = 0.5 * fps
    low = LP_freq / nyquist
    high = HP_freq / nyquist
    filtedData = series
    # do low pass first.
    if LP_freq != False:
        b, a = signal.butter(order, low, 'lowpass')
        filtedData = signal.filtfilt(b, a,filtedData,method = 'pad',padtype ='odd')
    if HP_freq != False:
        b, a = signal.butter(order, high, 'highpass')
        filtedData = signal.filtfilt(b, a,filtedData,method = 'pad',padtype ='odd')

    # b, a = signal.butter(order, [low, high], btype='bandpass')
    # filtered_data = signal.filtfilt(b, a, series,method = 'pad',padtype='odd')
    if keep_DC == True:
        filtedData += DC_power

    return filtedData


