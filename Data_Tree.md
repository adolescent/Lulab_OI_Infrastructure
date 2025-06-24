# 函数索引    

本文档是截止到2025-06-20的代码说明书，包含了每个函数的功能说明。    
文档本身较长，请使用搜索功能。

---
## _ZR_Codes_Archieved    
主要是本函数开发过程中的一些测试代码，已经归档，不做说明

---
## Brain_Atlas    

Boundaries.png:脑区边界的示意图    
Brain_Areas_A&B:两种不同绘制方法的脑区标注    

---
### Atlas_Mask.py 
生成标准脑区模板的函数类，调用方式：  
from Brain_Atlas.Atlas_Mask import Mask_Generator    
MG = Mask_Generator(bin=4)
- bin是缩放系数，决定生成的Mask分辨率。bin=1对应1320x1140，常用bin=4对应分辨率330x285    

本脑区模板是预生成的，基于Allen的鼠脑数据库。参考的工具包：    
https://ccf-streamlines.readthedocs.io/en/latest/

#### 内置变量
MG.breg:bregma 点的坐标    
MG.idmap:每个id代表一个脑区，0是mask外的区域。可对照MG.masks得到id对应的脑区    
MG.masks:一个列表，包含每个脑区的mask信息
#### MG.Pix_Label(y,x)
输入一点坐标，返回这一点归属的脑区
#### MG.Get_Mask(area,LR='L')
area为需要的脑区名，**需要全名**。LR为左/右半脑
#### MG.IDName(c_id)
c_id属于当前的id数字，返回这个数字对应的脑区全名。辅助Lite
#### MG.Avr_By_Area(graph,min_pix=100)
按脑区平均当前功能图。graph为当前需要平均的功能图，min_pix是平均时所需要的最小像素数。小于这个值的脑区会被忽略。
#### MG.Get_Weight_Map(area_names,weight_frame,spliter = '_')
辅助Lite，给定脑区名和每个脑区对应的权重，返回权重图。用于进行分脑区的一些计算和可视化。一般结合Atlas_Data_Tools功能使用。    
area_names：全名，比如VISrl_L，分隔符为spliter，默认为'_'，生成的heatmap一般自带脑区全名。    
weight_frame：每个脑区的可视化权重，需要手动提供。heatmap一般自带这个。
#### MG.Get_Func_Mask(area='VI',LR='L')
得到特定功能的全脑区mask。area为想到得到的功能区，支持VI/SS/MO/RSP四个选项，LR为左右半脑，返回一个mask，即是具有特定功能的脑区。
#### MG.Area_Counters()
返回所有脑区的边界图，1-0的脑区边界，不需要提供参数。

---
## OI_Functions
本文件夹下包含了基本成像图的处理函数，功能较多，建议参照Tutorial的demo使用。   

---
### Align_Tools.py
用于将成像功能图对齐到标准模板    
本函数只进行平移、缩放和旋转，不做切变，因此需要尽量确保成像时左右水平。    
调用方式为:    
import cv2    
from OI_Functions.Align_Tools import Match_Pattern    
MP = Match_Pattern(avr=avr_graph,bin=4,lbd=4.2)    
avr是对齐的平均图，参照这张平均图选择中线和参考点    
bin同Mask_Generator,根据采样分辨率选择bin值，256x256的采集推荐用bin=4    
lbd：lambda-bregma-distance,预估的两点间距离，参考值为4.2mm    
**注意，本函数的使用只能在远程桌面环境下运行，cv2工具包不然会没办法弹出选点窗口**
#### MP.Select_Anchor()
#### MP.Fit_Align_Matrix()
定义完MP后，运行这两个函数。Select Anchor用于选点，会弹出一个对话框，请依次选择bregma,lambda,中线上任意三点。Fit_Align_Matrix则根据选择的点拟合变换矩阵,将返回一个对齐的示意图，手动比较对齐的效果。如果效果满意可以进行下一步，不满意则可重复运行这两节。
#### MP.Transform_Series(stacks,intra = cv2.INTER_NEAREST)
如拟合和变换效果满意，则可使用上述变化进行下一步操作，对序列进行变换。   
stacks是全部待变换的图像帧，形状必须为（N_frame x Height x Width），一般预处理得到的形状就是合适的。    
intra是插值方法，参见https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html。默认使用的为最近邻插值，可以避免出现莫列波纹。    
这一步的操作需要一些时间。

---
### Atlas_Corr_Tools.py
用于对标准模板的活动进行分脑区计算，可以得到相关矩阵    
调用方式为:    
from OI_Functions.Atlas_Corr_Tools import Atlas_Data_Tools    
ADT = Atlas_Data_Tools(series=series,bin=4,min_pix=30)    
bin同上，min_pix为计算中所接受的最小脑区像素数。小于该数目的脑区会被忽略。    
series是**对齐至标准模板后**的活动序列。    
在使用该方法前，请先施加chamber mask，将成像皮层外的像素置零。

#### ADT.Get_All_Area_Response(keep_unilateral = False)
计算当前序列的全部脑区活动，并将活动储存在 *ADT.Area_Response* 变量中，这一变量包含每个脑区活动的全部信息。    
keep_unilateral决定是否保留不对称的脑区。    
#### ADT.Combine_Response_Heatmap()    
计算完成后，使用该方法得到heatmap，返回值为*ADT.Area_Response_Heatmap*，是每个脑区的热图，用于进行相关矩阵的计算。
#### ADT.Get_Corr_Matrix(win_size = 600,win_step = 150)
计算得到滑窗的相关矩阵，返回值为*ADT.Corr_Matrix*，为一个字典型，每个元素是一个时间窗的相关矩阵。    
可以设置win_size为全序列长度得到非滑窗的结果。    
相关矩阵中脑区的顺序已经被重排过，现在是左右对称的，并按照功能重排过。    

---
#### Contra_Similar(series,bin=4)
这一函数用来计算一个序列的左右对称性，即比较每个pix和镜面对称的pix的相关系数。    
series为原始的图片序列，顺序为(N_Frame x N_height x N_width)    
调用方式为:    
from OI_Functions.Atlas_Corr_Tools import Contra_Similar    
返回变量为每个pix的左右相似度图。

---
#### Paiwise_Calculator(matrix,mask)
计算mask内全部pixel之间的两两相关，返回一个pandas Frame，包含相关的两个像素ID，两个像素之间的距离，以及两个像素之间的相关系数，使用皮尔逊相关。
matrix于上面的series使用相同，mask为要做两两相关的mask范围，可以自定义，形状必须于matrix相同。

#### Pairwise_ID_Loc(mask,pixel_id)
是*Paiwise_Calculator*函数的Lite，用于还原回每个pixel id中，当前pixel所在的位置。需要提供mask和id。

---
### Common_Functions.py
包含很多通用函数，建议调用的时候直接调用整个工具包。   
调用方式为:    
import Common_Functions as cf    
#### cf.join(path_A,path_B)
连接两个目录，返回连接后的目录。常用于连接path和文件名。套壳而已。
#### cf.Get_File_Name(path,file_type='.bin',keyword='')
得到path目录下，具有特定拓展名和特定关键词的全部文件名（绝对目录），返回一个list。file_type是目标文件的拓展名，keyword留空则不指定关键词，全部目标类型的文件都被选择。
#### cf.Get_Subfolders(root_path,keyword='',method='Whole')
得到root_path下，具有特定关键词的的全部**子文件夹**名称。method可选'Whole'和'Relative'，返回子文件夹的绝对目录或相对目录。
#### cf.mkdir(path,mute=False)
创造名称为path的文件夹目录。请确保上级目录存在，不然会报错。如果目录已存在则返回提示，mute=True可以关闭提示。
#### cf.Save_Variable(save_folder,name,variable,extend_name='.pkl')
将变量保存至特定位置。save_folder是文件夹，name是保存的文件名，variable是要保存的变量，extend_name是要保存的文件拓展名。
#### cf.Load_Variable(save_folder,file_name=False)
读取使用上一个工具保存的变量。save_folder和文件名可以分开填写，或者设置file_name=False,在save_folder处填写完整路径，都可以用。
#### cf.List_Extend(input_list,front,tail)
是刺激对齐的一个小依赖Lite，实际使用场合不多。对一个全由数字构成的list进行延拓，front把list的头部重复数次，tail把list的最后一个元素重复数次。front和tail值为正则重复，为负值则剪切。
#### cf.kill_files(folder)
删除文件夹下的全部文件，一般用于清缓存。谨慎使用。
#### cf.Kill_Cache(root_folder)
使用UMAP的时候有的时候会报错(kernel dead)，这种时候就需要对缓存进行清理。root_folder是python环境所在的目录，这个函数会删除整个环境中的所有临时文件，跑的会有点慢，并且需要管理员权限。

---
### Graph_Aligner.py
用于对frame stack进行平移对齐的工具包。    
demo位于/Tutorial/P5_Graph_Align.ipynb，效果还可以。但最好的办法是在成像过程中保持固定的稳定。    
调用方法推荐：    
from OI_Functions.Graph_Aligner import*
#### detect_boulders_canny(gray,canny_low=3,canny_high=50)
使用canny算法对灰阶图片进行边缘检测，返回图像边界图。
#### detect_boulders_sobel(gray,ksize=3)
使用Sobel算法进行边缘检测，返回图像的梯度图。实际对齐使用了这个函数。
#### motion_correction(target,template,motion_lim=5,ksize=5)
对单张图片进行对齐。把target图片对齐到template上，两张图片需要大小相同。motion_lim是对其中所允许移动的最大像素数。ksize是进行对齐的过程中，使用sobel算法的参数。这一函数会分别返回x方向的移动量，y方向的移动量，以及对齐后的目标图。

---
### Map_Subtractor.py
本文件内的工具用于做减图。需要提供本工具包预处理得到的dRR_dics 才可以使用。   
建议的调用方法：    
from OI_Functions.Map_Subtractor import Sub_Map_Generator    
calculator = Sub_Map_Generator(drr_file)    
#### calculator.Submap_Core(cond_A,cond_B,used_frame = np.arange(4,16),map = 'ttest')
计算减图的核心工具。定义完成后，输入减图的两组condition，分别为cond_A和cond_B，并选定计算减图时的帧范围（与MATLAB代码相同）和减图方法（map='ttest'返回t图，map='sub'返回减图）。这一功能将返回减图和减图对应的p值图。    
#### calculator.Get_Map(许多参数)
这是包装过后的减图工具，相比Submap_Core增加了clip，filter，保存等功能，一键式出图。
#### calculator.Condition_Response_Curve(mask=[])
这个工具可以计算得到各个condition的反应曲线，从0-16帧。mask可以圈定计算反应曲线的面积。不返回值，反应曲线保存在calculator.Response_Curves，是Pandas Frame

---
### OIS_Preprocessing.py
用于对OIS数据进行预处理的一些工具。教程见\Tutorial\FC_Analysis_Demo\Part1_Preprocessing\P1A1_Preprocessing.ipynb
#### Single_Folder_Processor(path,save_format='python',subfolder='Preprocessed',keepna=False)
一键式地处理OIS采集的单个Run。path是'.bin'文件所在的位置，save_format目前只支持'python'，请不要修改。subfolder是预处理过后的图像序列所保存的位置。keepna则决定是否将nan填充为0，默认填充。
#### One_Key_OIS_Preprocessor(root_folder,save_formate= 'python')
一键式地处理多个OIS采集的Run，操作更方便，root_folder是当天采集的所有Run所在的目录。这个函数比分别处理的更方便，但要是掉盘了会更麻烦。
#### One_Key_Graph_Reader(path,keepna=False)
#### One_Key_AI_Reader(path)
这两个函数被用于单独处理文件夹中的图像文件和ai signal文件，其实是拆解了一键式函数的功能。

---
### OIS_Tools.py
一些用于OIS信号处理的内核级函数，可以在高度定制化的任务中使用。
#### Analog_Bin_Reader_Core(filename)
用于对单个bin文件的模拟信号stack进行解包。返回头文件和12道通道的模拟signal。
#### Analog_Reader_All(analog_file_names,mute = False)
提供全部模拟信号的bin文件list，并对其进行解包和*直接拼接*，一般用于直接得到一个Run之内的全部模拟信号list
#### Graph_Bin_Reader_Core(filename)
读取单个bin文件的图像stack，返回头文件和全部图片的stack，图片数据格式为uint16。要注意如果采集了多个通道，不同通道的信号不会被分割，需要进一步处理。
#### Graph_Reader_All(img_file_lists,channel_names=['Red'],mute=False,keepna=False)
输入全部图像stack的bin文件名列表，并提供采集channel的名字，返回头文件和图像stack的字典，key是channel的名称，每个channel的数据分开存储。
#### Info_Reader(txt_path)
读取当前的info文件，返回info文件提供的信息，包括采集频率、采集通道名称和speckel曝光时间。

---
### Seed_Functions.py
用于种子点计算的一些函数，可以得到种子点相关图。
#### Generate_Circle_Mask(center,radius,height,width)
得到圆形种子点mask。center为种子点圆心，radius为半径，height和width是生成Mask的形状。
#### Seed_Corr_Core(seed_series,response_matrix)
用于计算种子点相关的核心函数。seed_series提供种子点的活动序列，注意需要提前将种子点活动平均。response_matrix是全部像素的活动序列，每个pix都与种子点做相关。
#### Seed_Window_Slide(seed_mask,response_matrix,win_size,win_step)
进行滑窗种子点相关的函数。这个函数的计算速度较慢，使用的时候需要提前预估耗时。    seed_mask为种子点mask，response_matrix为全部像素的活动序列，win_size和win_step是滑窗的窗宽和步长，可以设置窗宽为序列长度得到不滑窗的相关。    
返回值为一个3D matrix，三个维度为(N_window,height,width)

---
### Seed_Grid.py
将成像区域划分成小格子，平均后再进行相关分析。这一方法相对较旧，已经弃用，如有需求可自行研究。

---
### Stim_Frame_Align.py
这个文件中的函数是一些刺激时间对齐的工具，用于根据analog signal，把OIS记录到的stimulus onset与相机时间进行同步，得到每一张图像帧采集时，所呈现的刺激是什么的对应关系。
#### Series_Clean(input_series,min_length)
用于过滤信号序列的毛刺，把小于min_length的毛刺去掉，以免后面图片切割出现问题
#### Pulse_Timer(input_series,skip_step)
用于找到每个成像帧的采集时间，skip_step是最小的两个trigger时间差。
#### Stim_Align(cleaned_series,stim_series,stim_check = True)
根据刺激序列对成像序列进行标记，标注每个图像帧对应的刺激id
#### Stim_Extend(input_series,head_extend,tail_extend)
对输入的刺激序列进行onset时间的延拓，在刺激onset前/后延长数帧。
#### Stim_Camera_Align(许多参数)
完整的用于进行图片对齐的工具。输入camera_trigger,stim_trigger和stim_series，返回每个成像帧对应的刺激id。注意要根据采样率对间隔时间和电平进行调整。

---
### Stimulus_dRR_Calculator.py
根据连续的成像序列，计算dR/R的工具包。
#### Find_Condition_IDs(series,id)
输入frame的ID序列和目标ID，返回具有特定ID的frame index，是用于计算dRR的Lite
#### dRR_Generator(frames,stim_frame_align,base_method = 'previous',base = [0,1])
输入全部图像的frames(N_Frame,height,width)，图像-刺激对照表(stim_frame_align)，base_method支持'previous'和'global'，决定以刺激onset前作为baseline还是以全局isi的平均作为baseline。base为计算dRR选择的帧范围，只在'previous'方法中被使用。    
其实实践中推荐使用global方法，得到的结果更可靠。

---
### VDaQ_dRR_Generator.py
这里的函数被用于对VDaQ采集的数据进行dRR计算。VDaQ采集的数据内置了ID，因此dRR的计算更容易。    
调用方法为：    
from OI_Functions.VDaQ_dRR_Generator import BLK2DRR    
reader = BLK2DRR(wp) # wp是blk文件的目录
#### reader.Read_All_Frames()
读取当前文件夹下的全部blk文件，并保存于变量*reader.all_graphs*，格式为(N_blk,N_Stim,N_Frame_Per_Stim,Height,Width)
#### reader.dR_R_Calculator(base_frame = [0,1],save = True)
根据全部的blk文件计算dRR，方法与MATLAB的函数类似，base_frame是计算dRR使用的baseline帧范围。save=True则会将生成的dRR文件保存到目录下。    
生成的dRR series变量为*reader.dRR_dic*，是字典型，key为id。

---
## Signal_Functions
本文件夹中包含一些用于信号处理的工具，滤波和FFT等。

---
### Filters.py
包含了包装过后的滤波器
#### Signal_Filter_1D(series,HP_freq,LP_freq,fps,keep_DC=True,order=5)
series是1D的信号，对信号进行高通和低通的巴特沃茨滤波。HP_freq与LP_freq分别是高通和低通滤波的参数，fps是采样率，keep_DC=True将会保留直流信号（即滤波后的信号均值不为零），order是巴特沃茨滤波的阶数越大滤波边界越陡峭。

---
###  Pattern_Tools.py
用于模式识别的工具包，目前只支持PCA。
#### Do_PCA