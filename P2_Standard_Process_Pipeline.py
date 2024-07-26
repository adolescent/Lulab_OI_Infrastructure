'''
This is the process pipe line of Ois 200 data.
Before do this, make sure preprocessing is done. (Demo P1) 


Actually you can pack them into a class, but I'm eazy...
It 
Just 
Works

########################## Logs #########################
ver 0.0.1  2024-04-25  Demo Generated


'''
#%% Imports
import OI_Functions.Common_Functions as cf
from OI_Functions.Stim_Frame_Align import Stim_Camera_Align
from OI_Functions.Stimulus_dRR_Calculator import dRR_Generator
from OI_Functions.Map_Subtractor import Sub_Map_Generator
import numpy as np
import re
import matplotlib.pyplot as plt

data_folder = r'F:\20240428_RF\M6_RF_Hbar_run01\Preprocessed'
stim_txt_path = r'G:\RF_20240428\Run01_H\stim_id_14_52_26.txt'
# save_path = r'D:\ZR\_Data_Temp\Ois200_Data\240417_M3_Bars\Run01_Results'
stim_trigger_id = 7



#%% 1.Stim Frame Align 
stim_ids = np.load(cf.join(data_folder,'ai_series.npy'))
camera_trigger = stim_ids[:,0]
stim_trigger = stim_ids[:,stim_trigger_id+3]
txt_file = open(stim_txt_path, "r")
stim_data = txt_file.read()
stim_series = re.split('\n| ',stim_data)[:]
stim_frame_align = Stim_Camera_Align(camera_trigger,stim_trigger,stim_series,head_extend=2,tail_extend=4,skip_frame=1000)
cf.Save_Variable(data_folder,'Stim_Frame_Align',stim_frame_align,'.sfa')

#%% 2. Get dF/F Frames
frame = np.load(cf.join(data_folder,'Red.npy'))
dRR_dics = dRR_Generator(frame,stim_frame_align)
cf.Save_Variable(data_folder,'dRR_Dictionaries',dRR_dics)
#%% 3. Generate dR/R Graphs.
all_conditions = list(dRR_dics.keys())
calculator = Sub_Map_Generator(dRR_dics)
for i in range(1,5):
    _,_,_ = calculator.Get_Map([i],[i+1],clip_value = 5,clip_method = 'fix',
                           savepath = cf.join(data_folder,'T_Graphs'),map = 'ttest',LP_sigma = 3,
                           graph_name = f'{i}_T_Test_Graph_Example')
#%% 4. You can plot response curve, and add mask to it.
calculator.Condition_Response_Curve()
curves = calculator.Response_Curves
import seaborn as sns
sns.lineplot(data = curves,hue = 'ID',x = 'Frame',y = 'Response')