'''
计算各个ROI之间的相关系数
1、ROI像素平均值
2、ROI内各个像素点
'''

from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm


def pearsonr_pixel_drr(test_dataframe,frames_affine,roi_num,frame_num):
    for a in range(roi_num[0], roi_num[1]):
        if a == roi_num[0]:
            seed = test_dataframe.columns[a]
            seed_roi= test_dataframe[seed].to_numpy().reshape(frames_affine.shape[1],frames_affine.shape[2])
            drr_red_nan_allseed = frames_affine[frame_num[0]:frame_num[1],seed_roi]
        else:
            seed = test_dataframe.columns[a]
            seed_roi= test_dataframe[seed].to_numpy().reshape(frames_affine.shape[1],frames_affine.shape[2])
            seed_series = frames_affine[frame_num[0]:frame_num[1],seed_roi]
            drr_red_nan_allseed = np.concatenate((drr_red_nan_allseed,seed_series),axis=1)


    roi_cor_pixel = np.zeros(shape =(drr_red_nan_allseed.shape[1],drr_red_nan_allseed.shape[1]),dtype='f8')
    for i in range(drr_red_nan_allseed.shape[1]):
        for j in tqdm(range(drr_red_nan_allseed.shape[1])):
            roi_cor_pixel[i,j] ,_ = pearsonr(drr_red_nan_allseed[:,i],drr_red_nan_allseed[:,j])
    return roi_cor_pixel,drr_red_nan_allseed

def pearsonr_roi_drr(test_dataframe,frames_affine,roi_num,frame_num):
    for a in range(roi_num[0], roi_num[1]):
        if a == roi_num[0]:
            seed = test_dataframe.columns[a]
            seed_roi= test_dataframe[seed].to_numpy().reshape(frames_affine.shape[1],frames_affine.shape[2])
            drr_red_nan_allseed_mean = frames_affine[frame_num[0]:frame_num[1],seed_roi].mean(1).reshape(-1,1)

        else:
            seed = test_dataframe.columns[a]
            seed_roi= test_dataframe[seed].to_numpy().reshape(frames_affine.shape[1],frames_affine.shape[2])
            seed_series = frames_affine[frame_num[0]:frame_num[1],seed_roi].mean(1).reshape(-1,1)
            drr_red_nan_allseed_mean = np.concatenate((drr_red_nan_allseed_mean,seed_series),axis=1)
    roi_cor_brainarea = np.zeros(shape =(drr_red_nan_allseed_mean.shape[1],drr_red_nan_allseed_mean.shape[1]),dtype='f8')
    for i in tqdm(range(drr_red_nan_allseed_mean.shape[1])):
        for j in range(drr_red_nan_allseed_mean.shape[1]):
            roi_cor_brainarea[i,j] ,_ = pearsonr(drr_red_nan_allseed_mean[:,i],drr_red_nan_allseed_mean[:,j])
    return roi_cor_brainarea,drr_red_nan_allseed_mean

def pearsonr_atlas_roi_drr(test_dataframe,frames_affine,roi_num,frame_num):
    for a in range(roi_num[0], roi_num[1]):
        if a == roi_num[0]:
            seed_roi= test_dataframe[a]
            drr_red_nan_allseed_mean = frames_affine[frame_num[0]:frame_num[1],seed_roi].mean(1).reshape(-1,1)

        else:
            seed_roi= test_dataframe[a]
            seed_series = frames_affine[frame_num[0]:frame_num[1],seed_roi].mean(1).reshape(-1,1)
            drr_red_nan_allseed_mean = np.concatenate((drr_red_nan_allseed_mean,seed_series),axis=1)
    roi_cor_brainarea = np.zeros(shape =(drr_red_nan_allseed_mean.shape[1],drr_red_nan_allseed_mean.shape[1]),dtype='f8')
    for i in tqdm(range(drr_red_nan_allseed_mean.shape[1])):
        for j in range(drr_red_nan_allseed_mean.shape[1]):
            roi_cor_brainarea[i,j] ,_ = pearsonr(drr_red_nan_allseed_mean[:,i],drr_red_nan_allseed_mean[:,j])
    return roi_cor_brainarea,drr_red_nan_allseed_mean

#roi_num = 10 #10,17,15
#frame_num = [0,frames_affine.shape[0]]
#roi_cor_pixel,drr_red_nan_allseed = pearsonr_roi_drr(test_dataframe,frames_affine,roi_num,frame_num)
#cf.Save_Variable(file_processed,'roi_cor_pixel',roi_cor_pixel)