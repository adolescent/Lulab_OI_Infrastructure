This folder will provide a full demonstration of functional connection analysis on OIS 200 Captured Data.
The demo data was located at 'D:\ZR\_Data_Temp\Ois200_Data\Full_Demo'

To-Do-Lists are as follows:
----------------------------------Process1：Preprocessing----------------------------------------------------
################## Analysis 1(P1A1) : Basic Preprocessing
1.(Done) Transfer .bin file into python-readable format
2.(Done) Time-course bin, down sample series into freq required.
3.(Done) Project series into standard space.
4.(Done) Filter(0.005-0.5Hz), generate dR/R and Z matrix.
5.(Done) Get data matrix of all brain area. Chamber mask, is important for it.

################## Analysis 2(P1A2) : Video Related Preprocessing
This part will show how to handel pupil and motion data captured from video.
1. Sample Video cut for model Tranning
2. Facemap return motion and pupil data(Not included, in ppt)
3. Get Valid capture duration
4. resample data to fit dR/R & Z matrix's sampling rate.



----------------------------------Process2：Direct Correlation Method----------------------------------------
##################  Analysis 1(P2A1) : Seedpoint and Corr Matrix
1.（Done） Seed point Correlation(window-slide)
2.（Done） Correlation matrix(re-arrange added)
3. (Done) Contralateral similarity calculation(brain-area avr added.)

##################  Analysis 2(P2A2) : Inner Area Consistent
1. Pairwise Correlation inner brain area
2. Single Pixel Relaxation Distance(Inner and Outside)
3.

##################  Analysis 3(P2A3) : Frequency and Specturm Analysis
1. 2-area(can combine area as required) specturm analysis(wavelet and slide window fourier)
    1.1. Contralateral Similarity calculation based on Specturm Methods
2. Phase Locking Analysis(Hilbert transform) *This method need "narrow-band-signal", which means we need to makesure or filt data into narrow band firstly.
 FYI:https://pmc.ncbi.nlm.nih.gov/articles/PMC3674231/
 https://www.youtube.com/watch?v=vKgm1Zxoscc


----------------------------------Process3：Pattern Recognition-----------------------------------------------
################### Analysis1 (P3A1): PCA Analysis
1. PCA analysis (binned & cut) for small memory.
2. Area-Averaged PCA Response 
3. PCA Network Timecourse 

################### Analysis2 (P3A2): Hierarchical and Seedpoint Growing Method
1. Hierarchical clustering pattern Recognition
2. Seed point growing method


################### Analysis3 (P3A3): Community-Structured Classification Method
method used on https://doi.org/10.1038/s41467-023-44363-z
https://cloud.tencent.com/developer/article/1801337
Methods below can be your reference.
Girvan-Newman：基于边介数的层次化算法。
Louvain：基于模块度优化的高效算法。
标签传播算法（LPA）：通过标签扩散发现社区。
谱聚类：基于图拉普拉斯矩阵的聚类方法。
Infomap：基于信息论的流压缩算法



----------------------------------Process4：Dynamical Analysis-------------------------------------------------
################### Analysis1 (P4A1): Timecourse Analysis
1. Ensemble Timecourse Repeat Analysis.(Peakfind & Waittime)
2. Timecourse Info combined with behavior.



################### Analysis2 (P4A2): Neuro-Trace
1. PCA-based trace analysis
2. UMAP-based trace analysis
2. High-Dimension Trace Analysis



