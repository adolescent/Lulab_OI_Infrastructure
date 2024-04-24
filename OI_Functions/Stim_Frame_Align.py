'''
These functions are useful to match stim with spon 

'''
#%% Imports
import numpy as np
import Common_Functions as cf

#%% Core functions, used below.
def Series_Clean(input_series,min_length):
    '''
    Clean stim triggers, make sure we have actuall stim on signals.
    '''
    arr = np.array(input_series)
    # Find the indices where the value changes
    indices = np.where(np.diff(arr) != 0)[0] + 1
    # Split the array into segments based on the change indices
    segments = np.split(arr, indices)
    # Replace the segments with length less than or equal to 2 with 0s
    result = []
    for seg in segments:
        if len(seg) <= min_length:
            result.extend([0] * len(seg))
        else:
            result.extend(seg.tolist())

    return result


def Pulse_Timer(input_series,skip_step):
    '''
    Get time of all pulse start time, AKA camera time.
    '''
    # Convert the boolean series to a numpy array
    arr = np.array(input_series)
    # Find the indices where the value changes from 0 to 1
    pulse_start_indices = np.where(np.diff(arr) == 1)[0] + 1
    # Filter out the pulse start indices where the distance to the previous is less than 5
    pulse_times = []
    for i, idx in enumerate(pulse_start_indices):
        if i == 0 or idx - pulse_start_indices[i-1] >= skip_step:
            pulse_times.append(idx)
    pulse_times = np.array(pulse_times)

    return pulse_times


def Stim_Align(cleaned_series,stim_series,stim_check = True):
    '''
    Align Stim id to stim on times, get the id of all current time.
    '''
    # cut all stim series.
    arr = np.array(cleaned_series)
    indices = np.where(np.diff(arr) != 0)[0] + 1
    segments = np.split(arr, indices)

    # check stim on ids
    count = sum(1 for arr in segments if (lambda x: np.all(x))(arr))
    if stim_check == False:
        print('No Stimulus check, make sure you know what you are doing.')
    elif (stim_check == True) and (count != len(stim_series)):
        raise ValueError('Stim trigger and Stim list Not Match! Check your input.')
    
    # fill stim id values into series.
    stim_ids = []
    stim_id = 0
    for seg in segments:
        if seg.sum() == 0:
            stim_ids.extend((seg-1).tolist())
        else:
            stim_ids.extend((seg*float(stim_series[stim_id])).tolist())
            stim_id += 1
    return stim_ids

def Stim_Extend(input_series,head_extend,tail_extend):
    '''
    Extend On frames with values given, get desired on series.
    '''
    indices = np.where(np.diff(input_series) != 0)[0] + 1
    # Split the array into segments based on the change indices
    segments = np.split(input_series, indices)
    adjusted_frame_list = []
    # start from first, we must have -1 at the beginning.
    adjusted_frame_list.append(cf.List_Extend(segments[0],0,-head_extend))
    for i in range(1,len(segments)-1):# First and last frame use differently.
        if (i%2) != 0:# odd id means stim on.
            adjusted_frame_list.append(cf.List_Extend(segments[i],head_extend,tail_extend))
        else:# even id means ISI.
            adjusted_frame_list.append(cf.List_Extend(segments[i],-tail_extend,-head_extend))
    # Process last part then.
    adjusted_frame_list.append(cf.List_Extend(segments[-1],-tail_extend,0))
    # After adjustion, we need to combine the list.
    frame_stim_list = []
    for i in range(len(adjusted_frame_list)-1):# Ignore last ISI, this might be harmful.
        frame_stim_list.extend(adjusted_frame_list[i])
    frame_stim_list = np.array(frame_stim_list)
    return frame_stim_list

#%% main function actually we can pack them into a class, but I'm lazy.

def Stim_Camera_Align(camera_trigger,stim_trigger,stim_series,
                           stim_check = True,
                           skip_frame = 1000,stim_on_min = 3000,
                           camera_level = 2.5,stim_level = 2.5,
                           head_extend = 2,tail_extend = 4,
                           ):
    """
    Align each stims time to frames, getting stim id series.
    ONLY SINGLE CHANNEL CAPTURE CAN USE THIS FUNCTION.

    Parameters
    ----------
    ##### Vital parameters
    camera_trigger : (1D array)
        Series of camera trigger, we expect pulse trigger here.
    stim_trigger : (1D array)  
        Series of stimulus voltage trigger, we expect same fps as camera trigger.
    stim_series : (1D array)
        Series of stim sequence.

    ##### Check parameters
    stim_check : (bool), optional
        Whether we check the sum of all stim num, to avoid error.

    ##### thresholds for find
    skip_frame : (int),optional
        skip count for each frame, to avoid mis-count.
    stim_on_min : (int),optional
        Min time of stim-on, to avoid mis-count.
    camera_level : (int),optional
        Voltage level to find camera trigger, usually 2.5 for 5V
    stim_level : (int),optional
        Voltage level to find stim trigger, usually 2.5 for 5V

    ##### Extend parameters
    head_extend : (int),optional
        Frames used before stim onset, usually use previous 2 frame
    tail_extend : (int),optional
        Frames used after stim offset, usually record 4 more frames.


    Returns
    -------
    Name_Lists : (list)
       Return a list, all file names contained.

    """
    # get frame times and stimulus timelines.
    camera_bool = camera_trigger>camera_level
    camera_time = Pulse_Timer(camera_bool,skip_frame)
    stim_trigger_filted = Series_Clean(stim_trigger>stim_level,stim_on_min)
    stim_timelines = Stim_Align(stim_trigger_filted,stim_series,stim_check)
    frame_stim_ids = np.zeros(len(camera_time))
    # align frame time with stims.
    for i,c_time in enumerate(camera_time):
        frame_stim_ids[i] = stim_timelines[c_time]
    # extend frames.
    frame_stim_ids = Stim_Extend(frame_stim_ids,head_extend,tail_extend)

    return frame_stim_ids


#%% Test run here.
if __name__ == '__main__':
    
    import OIS_Tools as Ois_Tools
    import matplotlib.pyplot as plt
    import seaborn as sns
    import re


    test_path = r'D:\ZR\_Data_Temp\Ois200_Data\240417_M3_Bars\test_exposetime0.1ms_Vbar_Run01\Preprocessed'
    stim_ids = np.load(cf.join(test_path,'ai_series.npy'))
    # and read stim txt here.
    txt_path = r'D:\ZR\_Data_Temp\Ois200_Data\240417_M3_Bars\stimid_folder_0240419\Run01_V\stim_id_17_14_44.txt'
    txt_file = open(txt_path, "r")
    stim_data = txt_file.read()
    stim_series = re.split('\n|,',stim_data)[:25]

    camera_trigger = stim_ids[:,0]
    stim_trigger = stim_ids[:,10]
    stim_frame_align = Stim_Camera_Align(camera_trigger,stim_trigger,stim_series,head_extend=2,tail_extend=4)
    
    cf.Save_Variable(test_path,'Stim_Frame_Align',stim_frame_align,'.sfa')
    plt.plot(stim_frame_align[:])


