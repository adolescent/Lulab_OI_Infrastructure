#%%

import OI_Functions.Common_Functions as of
import OI_Functions.Ois_Tools as Ois_Tools










#%%
if __name__ == '__main__':
    wp = r'D:\ZR\_Data_Temp\Ois200_Data\240417_M3_Bars\test_exposetime0.1ms_Vbar_Run01'
    analog_file_names = of.Get_File_Name(wp,'.bin','ai')
    # graph_file_name = of.Get_File_Name(r'D:\ZR\_Data_Temp\Ois200_Data\Test_Base_1color_2bins','.bin','img')
    graph_file_name = of.Get_File_Name(r'D:\ZR\_Data_Temp\Ois200_Data\Test_Base_2color\Spon_base_float','.bin','img')

