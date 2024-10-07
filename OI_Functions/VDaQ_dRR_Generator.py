'''
These functions will transfer VDaQ BLK files into dR/R format for further calculation.


##################################Logs################################
ver 0.0.1 2024-4-26 Function Established by ZR.


'''
#%%
import numpy as np
import struct
import matplotlib.pyplot as plt
import OI_Functions.Common_Functions as cf


class BLK2DRR(object):
    name = r'Transfer VDaQ BLK File into dR/R file.'
    def __init__(self,blk_path):
        self.blk_path = blk_path
        self.blk_lists = cf.Get_File_Name(blk_path,'.BLK')
        temp_data = np.fromfile(self.blk_lists[0],dtype = 'u4')[0:429]#前429为头文件
        self.BLK_Property = {}
        self.BLK_Property['Data_type'] = temp_data[7]#存储数据类型，12为uint16；13为uint32；14为float32.一般都是13
        self.BLK_Property['Width'] = temp_data[9]
        self.BLK_Property['Height'] = temp_data[10]
        self.BLK_Property['N_Frame_Per_Stim'] = temp_data[11]
        self.BLK_Property['N_Stim'] = temp_data[12]
        del temp_data

    def Read_All_Frames(self):
        self.all_graphs = np.zeros(shape = (len(self.blk_lists),self.BLK_Property['N_Stim'],self.BLK_Property['N_Frame_Per_Stim'],self.BLK_Property['Height'] ,self.BLK_Property['Width']))
        for i,c_blk in enumerate(self.blk_lists):
            data = np.fromfile(c_blk, dtype='<u4')[429:]
            c_graphs = np.reshape(data,(self.BLK_Property['N_Stim'],self.BLK_Property['N_Frame_Per_Stim'],self.BLK_Property['Height'] ,self.BLK_Property['Width']))
            self.all_graphs[i,:,:,:,:] = c_graphs
        
    
    def dR_R_Calculator(self,base_frame = [0,1],save = True): 
        try:
            self.all_graphs
        except AttributeError:
            print('BLK not read yet.')
            self.Read_All_Frames()
        # dRR_Dictionaries
        self.dRR_dic = {}
        self.dRR_dic[-1] = None # VDaq have no ISI method.

        id_lists = np.arange(self.all_graphs.shape[1])
        for i in range(len(id_lists)):
            c_id_frames = self.all_graphs[:,i,:,:,:]
            all_id_drr = np.zeros(shape = (c_id_frames.shape))
            for j in range(len(self.blk_lists)):
                c_blk = c_id_frames[j,:,:,:]
                c_base = c_blk[base_frame,:,:].mean(0)
                if c_base.sum() == 0:
                    raise ValueError(f'BLK{j} seems to be broken. remove before process.')
                c_drr = (c_blk/c_base)-1
                all_id_drr[j,:,:,:] = c_drr
            self.dRR_dic[i+1] = all_id_drr
        if save == True:
            cf.Save_Variable(cf.join(self.blk_path,'Processed'),'dRR_Dictionaries',self.dRR_dic)




if __name__ == '__main__':

    wp = r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\Run01_G8\Processed'
    # transfomer = BLK2DRR(wp)
    # transfomer.dR_R_Calculator()
    # dRR_dics = transfomer.dRR_dic
    dRR_dics = cf.Load_Variable(wp,'dRR_Dictionaries.pkl')
    #%% try to calculate some graphs.
    from OI_Functions.Map_Subtractor import Sub_Map_Generator
    
    calculator = Sub_Map_Generator(dRR_dics)
    graph,raw_drr,p_values = calculator.Get_Map([1,5],[3,7],clip_value = 1.5,
                           savepath = cf.join(wp,'T_Graphs'),
                           graph_name = 'H-V',map = 'ttest',HP_sigma = 300,LP_sigma = 0.75)


