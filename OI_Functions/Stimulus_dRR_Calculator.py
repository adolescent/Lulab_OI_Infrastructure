'''

These functions will calculate dR/R on Ois 200 stim runs.
Method of dR/R can be desined and only Stimulus runs can use these functions.

############################ Logs ####################
240425 v 0.0.1, method established, by ZR


'''
#%%  Import and core functions.
import numpy as np
import Common_Functions as cf
import matplotlib.pyplot as plt

def Find_Condition_IDs(series,id):

    # get conditions first
    result_series = []
    current_series = []
    for i, num in enumerate(series):
        if num == id:
            current_series.append(i)
        else:
            if len(current_series) > 0:
                result_series.append(current_series)
            current_series = []
    if len(current_series) > 0:
        result_series.append(current_series)
    # cut conditions of the same length
    condition_len = min(len(sublist) for sublist in result_series)
    result_series = [sublist[:condition_len] for sublist in result_series]

    return result_series,condition_len



#%% Core functions.
def dRR_Generator(frames,stim_frame_align,base_method = 'previous',base = [0,1]):
    '''
    This function will get dR/R condition series of the given data, aligned to stim frame align, and methods optional.
    
    Parameters
    ----------
    frames : (3D np array)
        Array of all captured data frame. The first dim shall be frame num.
    stim_frame_align : (series)
        Series of stimulus ids. Need to be generated before this function.
    base_method : ('isi' or 'previous')
        if choose isi, we will use global isi as base line of dR/R, elif we choose previous, we will use the previous several frames of each condition.
    base : (list), optional
        base frame of dR/R calculation. Will be ignored if base_method is set to 'isi'
        
    Returns
    -------
    dRR_dics : (dict)
        Dictionary of dR/R series, each key is a possible ID and followed by a 4D series.(e.g. (25,16,512,512) meaning that we have 25 repeats, each repeat have 16 frame, in pix 512x512.

    '''


    dRR_dics = {}
    all_ids = list(set(stim_frame_align))
    all_ids.remove(-1)
    # drr method notion.
    if base_method == 'isi':
        print(f'Use Global ISI as R0 to calculate dR/R.')
    elif base_method == 'previous':
        print(f'Use Frames {base} as R0 to calculate dR/R')

    # Get all ISI conditions, this have no condition problems.
    isi_locs = np.where(stim_frame_align==-1)[0]
    isi_frames = frames[isi_locs,:,:]
    isi_mean = isi_frames.mean(0)
    dRR_dics[-1] = (isi_frames/isi_mean)-1
    

    # Cut frame align into different conditions.
    for i,c_id in enumerate(all_ids):
        c_condition_ids,c_len = Find_Condition_IDs(stim_frame_align,c_id)
        c_condition_response = np.zeros(shape = (len(c_condition_ids),c_len,frames.shape[1],frames.shape[2]),dtype = 'f8')
        # for each repeat, we calculate dR/R.
        for j,c_cond in enumerate(c_condition_ids):
            c_cond_R_frames = frames[c_cond,:,:]
            if base_method == 'isi':
                base_frame = isi_mean
            elif base_method == 'previous':
                base_frame = c_cond_R_frames[base,:,:].mean(0)
            c_drr = c_cond_R_frames/base_frame
            c_condition_response[j,:,:,:] = c_drr-1
        dRR_dics[c_id] = c_condition_response
    
    
    # then stim on parts, these parts need to be the same length.

    return dRR_dics







if __name__ == '__main__':
    wp = r'D:\ZR\_Data_Temp\Ois200_Data\240417_M3_Bars\test_exposetime0.1ms_Vbar_Run01\Preprocessed'
    frames = np.load(cf.join(wp,'Red.npy'))
    stim_frame_align = cf.Load_Variable(wp,'Stim_Frame_Align.sfa')
    drr_dics = dRR_Generator(frames,stim_frame_align)
    cf.Save_Variable(wp,'dRR_Dictionaries',drr_dics) # dr/r data are in format f8, so usually 4 times bigger.
    

