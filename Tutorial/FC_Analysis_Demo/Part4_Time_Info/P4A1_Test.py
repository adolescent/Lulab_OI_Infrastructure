'''
This script will try to do basic time information analysis for spontaneous response.
Including:
1. Event find,repeat time and half-peak-length
2. FFT
3. Interval analysis (fit curve included)

This works for both raw activation and PCA weight both.

'''
#%%
import Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm
import pandas as pd
from Brain_Atlas.Atlas_Mask import Mask_Generator
import seaborn as sns
  
wp = r'D:\ZR\_Data_Temp\Ois200_Data\Full_Demo\Wild_Type\Preprocessed'
MG = Mask_Generator(bin=4)

series = np.load(cf.join(wp,'z_series.npy'))
# join chamber mask with brain area mask, getting only mask with values.
mask = cv2.imread(cf.join(wp,'Chamber_mask.png'),0)>0
joint_mask = (series.std(0)>0)*mask
# mask and clip input graph.
# NOTE this part is important for getting correct results.
series = np.clip(series,-3,3)*joint_mask

#%% Getting visual area.
'''
Part 1, easy peak find analysis of example series
docs here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html
'''
LV_Mask = joint_mask*(MG.Get_Func_Mask('VI','L'))
LV_curve = series[:,LV_Mask].mean(1)
plt.plot(LV_curve)
del series # save memory
#%% Find peaks and FWHM of test series, to determin parameter.
from scipy.signal import find_peaks,peak_widths
# x = LV_curve[10000:11000]
x = LV_curve
peaks, properties = find_peaks(x, prominence=0.75,distance=10,height=-0.5) # these parameters is tuned manually. prominence: peak must be how higher than local max. distance: dist of 2 peak(avoid noist), height: min height of peak.

plt.plot(x)
plt.plot(peaks, x[peaks], "x")
# plt.vlines(x=peaks, ymin=x[peaks] - properties["prominences"],
#            ymax = x[peaks], color = "C1")
# plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
#            xmax=properties["right_ips"], color = "C1")
plt.show()
#%% half height
results_half = peak_widths(x, peaks, rel_height=0.5) # bigger rel_height will cause bugs, as peak maynot always decay to base line.
# results_full = peak_widths(x, peaks, rel_height=0.8)
# higher peak have bigger FWHM is normal.
plt.plot(x)
plt.plot(peaks, x[peaks], "x")
plt.hlines(*results_half[1:], color="C2")
# plt.hlines(*results_full[1:], color="C3")
# results_half in shape(width,height,Left,Right)
#%%
plt.scatter(x = results_half[0],y = properties["prominences"],s = 3,c='r')
plt.scatter(x = results_half[0],y = results_half[1],s = 3,c ='b')

#%% Now calculate the interval of nearest peak.
def calculate_waiting_times(event_times):
    # Sort the event times chronologically
    sorted_times = sorted(event_times)
    
    # Calculate the waiting time between consecutive events
    waiting_times = []
    for i in range(len(sorted_times) - 1):
        waiting_time = sorted_times[i+1] - sorted_times[i]
        waiting_times.append(waiting_time)
    return waiting_times

duration = calculate_waiting_times(peaks)
# fit duration with exponential 

#%%
'''
Below shows waittime duration analysis
'''
from scipy.stats import expon,weibull_min
params = expon.fit(duration)
# params_w = weibull_min.fit(duration,floc=0)

# Create a range of x values for the fitted line
x = np.linspace(0, max(duration), 100)
# Exponential PDF
pdf_fitted = expon.pdf(x, *params)
# pdf_fitted = weibull_min.pdf(x, *params_w)
sns.histplot(duration, bins=30, kde=False, stat='density', color='blue', alpha=0.6, label='Data Histogram')
# Plot the fitted exponential distribution
plt.plot(x, pdf_fitted, 'r-', label='Fitted Exponential Distribution', lw=2)
plt.title('Histogram and Fitted Exponential Distribution')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
# plt.grid()
plt.show()
#%% then QQ plot and R
from scipy.stats import expon, probplot, linregress

fig, ax = plt.subplots(figsize=(10, 6))
res = probplot(duration, dist="expon", sparams=params, plot=ax)

# 3. Calculate linear regression to find R-squared
theoretical_quantiles = res[0][0]  # Theoretical quantiles
sample_quantiles = res[0][1]*params[1]        # Sample quantiles
slope, intercept, r_value, p_value, std_err = linregress(theoretical_quantiles, sample_quantiles)
r_squared = r_value**2

# 4. Add title, labels, and R-squared value to the plot
plt.title('Q-Q Plot: Data vs. Fitted Exponential Distribution')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Sample Quantiles')
plt.grid()
plt.text(0.1, 0.9, f'R²: {r_squared:.4f}', transform=ax.transAxes, fontsize=12, color='red')
plt.show()

#%% 
'''
Below shows an FFT analysis, including fixed-time FFT and slide-window FFT.
Function is already packed.
'''
from Signal_Functions.Spectrum_Tools import *

freq,power,freq_raw,power_raw = FFT_Spectrum(series=LV_curve,fps=5,ticks=0.01,plot=True)

#%% slide window fft
slide_spectrum = FFT_Spectrum_Slide(LV_curve,1500,300,5,0.01,True)

sns.heatmap(slide_spectrum.loc[:0.5,:],center = 0)

