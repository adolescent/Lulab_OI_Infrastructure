'''
This script will produce dR/R frames for orientation and color runs, for preprocess of classifier
'''

#%%
import Common_Functions as cf
from OI_Functions.Map_Subtractor import Sub_Map_Generator
from OI_Functions.VDaQ_dRR_Generator import BLK2DRR
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
import pandas as pd



wp = r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI'
orien_folder = r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\Run01_G8'
color_folder = r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\Run12_RGLum4'

#%% 
'''
Step1, generate dR/R frames for orien and color. Already done.
'''
orien_reader = BLK2DRR(orien_folder)
orien_reader.Read_All_Frames()
orien_reader.dR_R_Calculator(base_frame=[0,1],save=True)
color_reader = BLK2DRR(color_folder)
color_reader.Read_All_Frames()
color_reader.dR_R_Calculator(base_frame=[0,1],save=True)

#%% 
'''
Step 2, get correct unbaised func maps.
'''
## color first 
# calculator = Sub_Map_Generator(color_reader.dRR_dic)
color_drr = cf.Load_Variable(r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\Run12_RGLum4\Processed\dRR_Dictionaries.pkl')
calculator = Sub_Map_Generator(color_drr)
rglum,raw_drr,flited_drr,p = calculator.Get_Map([1,2],[3,4],clip_value = 5,savepath = r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\_V2_Sripe_Method',filter_flag=True,HP_sigma=20,LP_sigma=0.75,graph_name = 'RG-Lum')
# sns.heatmap(flited_drr,center = flited_drr.mean(),square = True) # Q-check

## then AO and HV
orein_drr = cf.Load_Variable(r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\Run01_G8\Processed\dRR_Dictionaries.pkl')
calculator = Sub_Map_Generator(orein_drr)
ao,raw_drr,flited_drr,p = calculator.Get_Map([2,6],[4,8],clip_value = 5,savepath = r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\_V2_Sripe_Method',filter_flag=True,HP_sigma=20,LP_sigma=0.75,graph_name = 'A-O')
sns.heatmap(flited_drr,center = flited_drr.mean(),square = True)


hv,raw_drr,flited_drr,p = calculator.Get_Map([1,5],[3,7],clip_value = 5,savepath = r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\_V2_Sripe_Method',filter_flag=True,HP_sigma=20,LP_sigma=0.75,graph_name = 'H-V')
sns.heatmap(flited_drr,center = flited_drr.mean(),square = True)

#%%
'''
Step3, calculate proper Orien and Color index for this data.
Norm Tuning into 0-1.
'''
from scipy.ndimage import gaussian_filter

# filt and normalize tunings.
wp = r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\_V2_Sripe_Method'
LP_sigma = 0.75
HP_sigma = 20
clip_std = 2.5
hv_raw = cf.Load_Variable(wp,'H-V_Raw.pkl')
ao_raw = cf.Load_Variable(wp,'A-O_Raw.pkl')
rg_raw = cf.Load_Variable(wp,'RG-Lum_Raw.pkl')
filted_graph = []

for i,c_map in tqdm(enumerate([hv_raw,ao_raw,rg_raw])):
    c_clipped = np.clip(c_map,c_map.mean()-clip_std*c_map.std(),c_map.mean()+clip_std*c_map.std())
    HP_graph = gaussian_filter(input = c_clipped, sigma = HP_sigma)
    LP_graph = gaussian_filter(input = c_clipped, sigma = LP_sigma)
    c_filted_graph = (LP_graph-HP_graph)
    filted_graph.append(c_filted_graph)

# sns.heatmap(filted_graph[0],center = 0,square = True)
tunings = np.array(filted_graph)

orien_tunings = abs(tunings[:2,:,:]).max(0)
color_tunings = abs(tunings[2,:,:])


orien_tunings = orien_tunings/orien_tunings.max()
color_tunings = color_tunings/color_tunings.max()
tuning_array = np.array([orien_tunings,color_tunings])
cf.Save_Variable(wp,'tunings',tuning_array)

sns.heatmap(tuning_array[0,:,:],center = 0,square = True)

#%%
'''
Step4, label graph into thick and thin stripe, aka TRAIN-SET
'''
# load thick and thin domain's mask
mask_path = r'D:\ZR\_Data_Temp\VDaQ_Data\200910_L80_LM_OI\_V2_Sripe_Method\Masks'
thick_mask = cv2.imread(cf.join(mask_path,'orien_mask.png'),0)>128
thin_mask = cv2.imread(cf.join(mask_path,'colormask.png'),0)>128
v2_mask = cv2.imread(cf.join(mask_path,'V2_Mask.png'),0)>128

# get masked tuning data
thick_tunings = tuning_array[:,thick_mask]
thin_tunings = tuning_array[:,thin_mask]
train_sets = np.concatenate((thick_tunings,thin_tunings),axis = 1)
train_sets = train_sets.T # N_sample*N_dim
# get label ids,mark thick as 1, thin as 2.
thick_ids = np.ones(thick_tunings.shape[1])*1
thin_ids = np.ones(thin_tunings.shape[1])*2
train_ids = np.concatenate((thick_ids,thin_ids))

# plot train set's tuning ID and it's tuning location.
plotable = pd.DataFrame(0.0,index = range(len(train_ids)),columns = ['Orien_Tuning','Color_Tuning','Stripe'])
for i in tqdm(range(len(train_ids))):
    c_id = ['Error','Thick','Thin'][int(train_ids[i])]
    plotable.loc[i,:] = [train_sets[i,0],train_sets[i,1],c_id]
#%% plot part
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
# sns.scatterplot(data = plotable,x = 'Orien_Tuning',y = 'Color_Tuning',hue = 'Stripe',ax = ax,s = 5,linewidth=0,alpha = 0.5)
# sns.histplot(data = plotable,x = 'Orien_Tuning',y = 'Color_Tuning',hue = 'Stripe',ax = ax,alpha = 0.4, palette="tab10",bins = 25)
sns.kdeplot(data = plotable,x = 'Orien_Tuning',y = 'Color_Tuning',hue = 'Stripe',ax = ax,alpha = 0.4, palette="tab10",fill=True,levels = 7,thresh=0.1)


#%%
'''
Step5, train SVM, and plot SVM predicted results.
'''

# we use linear kernal svm and prob. model.
from sklearn import svm
from sklearn.model_selection import cross_val_score

# fit svm, and test 5-fold score validation.
classifier = svm.SVC(C = 1,probability=True,kernel='linear')
scores = cross_val_score(classifier,train_sets, list(train_ids), cv=5)
print(f'Score of 5 fold SVC on unsupervised : {scores.mean()*100:.2f}%')
classifier.fit(train_sets,list(train_ids))

# and train prob. svm on given data set.
v2_datas = tuning_array[:,v2_mask]
test_data = v2_datas.reshape(2,-1).T
pred_ids = classifier.predict(test_data)
pred_probs = classifier.predict_proba(test_data)
dists = classifier.decision_function(test_data)
# we can also get determin function if you need. y = k*x +b
w = classifier.coef_[0]
k = -w[0] / w[1]
b = -classifier.intercept_[0]/ w[1]
# save train data in dataframe.
test_dataframe = pd.DataFrame(0.0,index = range(len(test_data)),columns = ['Orien_Tuning','Color_Tuning','Predicted_id','Thick_Prob','Thin_Prob','Dists'])

for i in tqdm(range(len(test_data))):
    c_id = ['Error','Thick','Thin'][int(pred_ids[i])]
    test_dataframe.loc[i,:] = [test_data[i,0],test_data[i,1],c_id,pred_probs[i,0],pred_probs[i,1],dists[i]]
cf.Save_Variable(wp,'Predicted_V2_ID_Linear',test_dataframe)
cf.Save_Variable(wp,'SVM_Model',classifier)

#%% quality test, on dist and tuning location.
plt.clf()
plt.cla()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
sns.scatterplot(data = test_dataframe,x = 'Orien_Tuning',y = 'Color_Tuning',hue = 'Predicted_id',ax = ax,s = 5,linewidth=0,alpha = 0.5)

#%%
'''
Step 6, recover prediction on all V2 data.
'''
recovered_data = np.zeros(shape = (540,654))+0.5
recovered_data[v2_mask == 1] = pred_probs[:,0]
sns.heatmap(recovered_data,vmax = 1,vmin = 0,center = 0.5)
