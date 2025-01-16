'''
This script will turn .bin data into readable ones in python or matlab.




########################## LOGS ###############################
(Actually you can do this on git = =)

ver 0.0.1 by ZR, function created. 2024/04/19


'''
#%%

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import OI_Functions.Common_Functions as cf
from OI_Functions.OIS_Tools import *
import numpy as np
from scipy.io import savemat
import time





def Single_Folder_Processor(path,save_format = 'python',subfolder = 'Preprocessed',keepna = False):

    """
    Get all file names of specific type.

    Parameters
    ----------
    path : (str)
        Single Folder Processor Will save .
    save_format : ('python' or 'matlab')
        Determine which type will we save data. 
    subfolder : (str),optional
        Subfolder to save data.
    keepna : (bool),optional
        Whether keep na, or fill na with nearby values.
    Returns
    -------
    Name_Lists : (list)
       Return a list, all file names contained.

    """

    # get important file paths
    try:
        info_path = cf.Get_File_Name(path,'.txt','info')[0]
    except IndexError:
        path_rel = path.split('\\')[-1]
        print(f'Folder [{path_rel}] seems is not a run folder, we will skip it.')
        return False
    
    save_path = cf.join(path,subfolder)
    cf.mkdir(save_path)
    analog_file_names = cf.Get_File_Name(path,'.bin','ai')
    img_file_names = cf.Get_File_Name(path,'.bin','img')

    # get all channel datas
    channel_names = Info_Reader(info_path)['Channel_Names']
    _,ai_signals = Analog_Reader_All(analog_file_names)
    _,graphs_all = Graph_Reader_All(img_file_names,channel_names,keepna = keepna)

    # save datas in format we want.
    if save_format == 'python':# save with pyton npy.
        np.save(cf.join(save_path,'ai_series.npy'),ai_signals)
        # and save graphs.
        for i,c_name in enumerate(channel_names):
            c_graphs = graphs_all[c_name]
            avr_graph = c_graphs.mean(0).astype('u2')
            plt.imsave(cf.join(save_path,f'{c_name}.png'), avr_graph, cmap='gray', vmin=avr_graph.min(), vmax=avr_graph.mean()+5*avr_graph.std(), format='png')
            np.save(cf.join(save_path,f'{c_name}.npy'),c_graphs)
    elif save_format == 'matlab': # save mat seems not working, avoid using it.
        savemat(cf.join(save_path,'ai_series.mat'),{'ai_signals':ai_signals})
        for i,c_name in enumerate(channel_names):
            c_graphs = graphs_all[c_name]
            avr_graph = c_graphs.mean(0).astype('u2')
            plt.imsave(cf.join(save_path,f'{c_name}.png'), avr_graph, cmap='gray', vmin=avr_graph.min(), vmax=avr_graph.mean()+5*avr_graph.std(), format='png')
            savemat(cf.join(save_path,f'{c_name}.mat'),{f'{c_name}':c_graphs})

    return True

#%% All Dayfolder Preprocessing.
def One_Key_OIS_Preprocessor(root_folder,save_format = 'python'):
    """
    Process One days run in a single key.

    Parameters
    ----------
    root_folder : (str)
        Root folder of all days data.
    save_format : ('python' or 'matlab')
        Determine which type will we save data. 

    Returns
    -------
    True

    """
    start_time = time.time()
    all_run_folder = cf.Get_Subfolders(root_folder)

    for i,c_run in tqdm(enumerate(all_run_folder)):
        c_run_name = c_run.split('\\')[-1]
        print(f'Processing run [{c_run_name}]....')
        Single_Folder_Processor(c_run,save_format=save_format)

    end_time = time.time()
    print(f'Process Done. \nTime cost : {(end_time-start_time):.1f} s')

#%% Recently we found HD have problem frequently. So I added some functions, use which can you save seperately.
# these functions will cost more memory, but will not lose middle vars if HD drops.
def One_Key_Graph_Reader(path,keepna = False):
    info_path = cf.Get_File_Name(path,'.txt','info')[0]
    img_file_names = cf.Get_File_Name(path,'.bin','img')
    channel_names = Info_Reader(info_path)['Channel_Names']
    _,graphs_all = Graph_Reader_All(img_file_names,channel_names,keepna = keepna)
    return graphs_all
    


def One_Key_AI_Reader(path):
    # info_path = cf.Get_File_Name(path,'.txt','info')[0]
    analog_file_names = cf.Get_File_Name(path,'.bin','ai')
    _,ai_signals = Analog_Reader_All(analog_file_names)
    return ai_signals


if __name__ == '__main__':
    
    path = r'D:\ZR\_Data_Temp\Ois200_Data\240417_M3_Bars'
    One_Key_OIS_Preprocessor(path)