

#%% Imports
import OI_Functions.Common_Functions as cf
from OI_Functions.Stim_Frame_Align import Stim_Camera_Align
from OI_Functions.Stimulus_dRR_Calculator import dRR_Generator
from OI_Functions.Map_Subtractor import Sub_Map_Generator
import numpy as np
import re
import matplotlib.pyplot as plt
from P1_OIS_Preprocessing import One_Key_OIS_Preprocessor

raw_folder = r'D:\YJX\spon_data\240719_RF_OIS_TEST'
# preprocess first
One_Key_OIS_Preprocessor(raw_folder)
# wp = r'D:\YJX\spon_data\240719_RF_OIS_TEST\Run01_RF_VBar_20Hz_2ms'

#%%
data_folder = r'D:\YJX\spon_data\240719_RF_OIS_TEST\Run01_RF_VBar_20Hz_2ms\Preprocessed'
stim_trigger_id = 7
stim_ids = np.load(cf.join(data_folder,'ai_series.npy'))
camera_trigger = stim_ids[:,0]
stim_trigger = stim_ids[:,stim_trigger_id+3]
# txt_file = open(stim_txt_path, "r")
# stim_data = txt_file.read()
# stim_series = re.split('\n| ',stim_data)[:]
stim_series = list(map(lambda x: x, [1, 2, 3, 4, 5, 6] * 21))
stim_series.extend([7])
stim_frame_align = Stim_Camera_Align(camera_trigger,stim_trigger,stim_series,head_extend=10,tail_extend=50,skip_frame=30,stim_level=1.45,stim_check = True,stim_on_min=15000)
cf.Save_Variable(data_folder,'Stim_Frame_Align',stim_frame_align,'.sfa')
#%% Generate dFF Graphs.
frame = np.load(cf.join(data_folder,'Red.npy'))
dRR_dics = dRR_Generator(frame,stim_frame_align[:-136],base_method='previous',base = np.arange(12,16))
# Plot curves
all_conditions = list(dRR_dics.keys())
calculator = Sub_Map_Generator(dRR_dics)
calculator.Condition_Response_Curve()
curves = calculator.Response_Curves
#%%
import seaborn as sns
# plotable = curves[curves['ID']==2]
plotable = curves[curves['ID']!=7]
plotable = plotable[plotable['ID']!=6]
sns.lineplot(data = plotable,hue = 'ID',x = 'Frame',y = 'Response',errorbar=None)
#%% Plot Maps

for i in range(1,6):
    graph,raw_drr,p_values = calculator.Get_Map([i],[6],clip_value =1,clip_method = 'std',
    savepath = cf.join(data_folder,'T_Graphs'),map = 'ttest',LP_sigma = 0.7,save_flag = True,HP_sigma=100,used_frame = np.arange(20,45),
    graph_name = f'{i}_T_Test_Graph_Example')
