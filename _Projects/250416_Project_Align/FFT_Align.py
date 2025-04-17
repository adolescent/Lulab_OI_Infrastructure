'''
This script is designed for remove motion correction noise. 
Denoise as an motion correction.

'''



#%% 
import numpy as np
import matplotlib.pyplot as plt
import Common_Functions as cf
from scipy import fftpack
from scipy.ndimage import shift
import skimage.io
from skimage import img_as_float
import seaborn as sns
from tqdm import tqdm
import cv2
from Signal_Functions.Filters import*

wp = r'D:\WSL\data\ois\250409\16#\analysis\0411_for16_test\Preprocessed'
series_raw = np.load(cf.join(wp,'binned_r_series.npy'))

# there are motions between 8000 to 12000. Let's try to correct it.
#%%
test_series = series_raw[9000:11000,:,:]
template = test_series.mean(0)
# test_series = series_raw

target = test_series[0,50:,:]
#%% Function is designed for Alignment only, API not easy to use directly.
import numpy as np
from scipy.signal import correlate,convolve
from scipy.ndimage import shift

def detect_boulders_canny(gray, canny_low=3, canny_high=50):

    # Reduce noise with Gaussian blur
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blurred = gray
    # Detect edges using Canny
    edges = cv2.Canny(blurred, canny_low, canny_high)

    return edges.astype('f8')

def detect_boulders_sobel(gray,ksize=3):
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    # Step 2: Compute gradient magnitude
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_mag = np.uint8(255 * gradient_mag / np.max(gradient_mag))  
    # Normalize to 0-255
    # Step 3: Threshold to retain strong edges
    # _, thresh = cv2.threshold(gradient_mag, thres, 255, cv2.THRESH_BINARY)
    return gradient_mag.astype('f8')



def motion_correction(target, template,motion_lim = 5,ksize = 5):

    ## Preprocess for boulder tetection.
    # template_mean_sub = template - np.mean(template)
    # target_mean_sub = target - np.mean(target)
    target_u1 = (target/256).astype('u1')
    template_u1 = (template/256).astype('u1')
    template_mean_sub = detect_boulders_sobel(template_u1,ksize)
    target_mean_sub = detect_boulders_sobel(target_u1,ksize)
    # plt.imshow(target_mean_sub)

    # Compute cross-correlation 
    corr = correlate(template_mean_sub, target_mean_sub, mode='same')
    # corr = convolve(template_mean_sub, target_mean_sub, mode='same')
    
    # Determine the center of the correlation matrix
    center_y, center_x = np.array(corr.shape) // 2
    
    # Extract a 11x11 region around the center (5 pixels in each direction)
    corr_region = corr[center_y - motion_lim : center_y + motion_lim+1, center_x - motion_lim : center_x + motion_lim+1]
    
    # Find the peak within the region
    peak_y, peak_x = np.unravel_index(np.argmax(corr_region), corr_region.shape)
    
    # Calculate the shift from the target to the template
    dx = peak_x - motion_lim  # x shift (columns)
    dy = peak_y - motion_lim  # y shift (rows)
    
    # Ensure shifts are within the ±5 pixel limit (handles edge cases)
    dx = np.clip(dx, -motion_lim, motion_lim)
    dy = np.clip(dy, -motion_lim, motion_lim)
    
    # Shift the target image by (-dx, -dy) to align with the template
    aligned = shift(target, (dy, dx), order=0, mode='constant', cval=0.0)
    
    return dx, dy, aligned

x,y,aligned = motion_correction(target,template,motion_lim=50)
# plt.imshow(aligned)


#%% calculate motion correction.
# x,y,aligned = Alignment(template,target)
xs = np.zeros(len(test_series))
ys = np.zeros(len(test_series))
aligned_frames = np.zeros(shape = test_series.shape,dtype='u2')
for i in tqdm(range(len(test_series))):
    xs[i],ys[i],aligned_frames[i,:,:] = motion_correction(test_series[i,:,:],template)

plt.plot(xs,alpha = 0.7)
plt.plot(ys,alpha = 0.7)


#%% test before and after align.

import skvideo.io

def Video_Plotter(series,savepath,filename,fps=15):
    # series here only accept uint 8 data type!
    if series.dtype != 'uint8':
        raise ValueError('Only uint 8 data type allowed.')
    
    fullpath = cf.join(savepath,filename)+'.avi'
    _,height,width = series.shape
    
    writer = skvideo.io.FFmpegWriter(fullpath, outputdict={
        '-vcodec': 'rawvideo',  #use the h.264 codec
        #  '-vcodec': 'libx264',
        '-crf': '0',           #set the constant rate factor to 0, which is lossless
        '-preset':'veryslow',   #the slower the better compression, in princple, 
        # '-r':str(fps), # this only control the output frame rate.
        '-pix_fmt': 'yuv420p',
        '-vf': "setpts=PTS*{},fps={}".format(25/fps,fps) , # I don't know why, but fps can only be set this way.
        '-s':'{}x{}'.format(width,height)
    }) 

    for frame in tqdm(series):
        # cv2.imshow('display',frame)
        writer.writeFrame(frame)  #write the frame as RGB not BGR
        # time.sleep(1/fps)

    writer.close() #close the writer
    cv2.destroyAllWindows()
    return True


series_A = (test_series.astype('f8')/256).astype('u1')
series_B = (aligned_frames.astype('f8')/256).astype('u1')

# norm_series_A = series_A/max(series_A.max(),abs(series_A.min()))
# plotable_A = (norm_series_A*127+127).astype('u1')
# norm_series_B = series_B/max(series_B.max(),abs(series_B.min()))
# plotable_B = (norm_series_B*127+127).astype('u1')
concat_series = np.concatenate((series_A,series_B),axis=2)

Video_Plotter(series = concat_series,savepath=wp,filename='drr_Z_compare',fps = 10) # 10 fps is 2x speed.

