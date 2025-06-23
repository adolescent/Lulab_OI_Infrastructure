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
### Align_Tools.py
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
### Contra_Similar(series,bin=4)
这一函数用来计算一个序列的左右对称性，即比较每个pix和镜面对称的pix的相关系数。    
series为原始的图片序列，顺序为(N_Frame x N_height x N_width)    
调用方式为:    
from OI_Functions.Atlas_Corr_Tools import Contra_Similar    
返回变量为每个pix的左右相似度图。

---
### Paiwise_Calculator(matrix,mask)
计算mask内全部pixel之间的两两相关，返回一个pandas Frame，包含相关的两个像素ID，两个像素之间的距离，以及两个像素之间的相关系数，使用皮尔逊相关。
matrix于上面的series使用相同，mask为要做两两相关的mask范围，可以自定义，形状必须于matrix相同。

### Pairwise_ID_Loc(mask,pixel_id)
是*Paiwise_Calculator*函数的Lite，用于还原回每个pixel id中，当前pixel所在的位置。需要提供mask和id。

---

