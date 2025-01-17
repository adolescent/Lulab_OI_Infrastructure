This folder will provide a full demonstration of functional connection analysis on OIS 200 Captured Data.
The demo data was located at 'D:\ZR\_Data_Temp\Ois200_Data\Full_Demo'

To-Do-Lists are as follows:
----------------------------------Process1：Preprocessing-----------------------------------------------
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



----------------------------------Process2：Direct Correlation Method-----------------------------------------------
##################  Analysis 1(P2A1) : Seedpoint and Corr Matrix
1.（Done） Seed point Correlation(window-slide)
2.（Done） Correlation matrix(re-arrange added)
3. Contralateral similarity calculation(brain-area avr added.)

##################  Analysis 2(P2A2) : Inner Area Consistent
1. Pairwise Correlation inner brain area
2. Single Pixel Relaxation Distance(Inner and Outside)
3.

##################  Analysis 3(P2A3) : Frequency and Specturm Analysis
1. 2-area(can combine area as required) specturm analysis(wavelet and slide window fourier)
    1.1. Contralateral Similarity calculation based on Specturm Methods
2. Phase Locking Analysis(Hilbert transform)



----------------------------------Process3：Pattern Recognition-----------------------------------------------
################### Analysis1 (P3A1): PCA Analysis
0. ROI Timecourse Heamap(re-arrange added)
1. ROI-based PCA analysis
2. Raw PCA analysis (Mosiac )

################### Analysis2 (P3A2): Clustering & Classification Method
1. Hierarchical clustering pattern Recognition
2. Seed point growing method



----------------------------------Process4：Dynamical Analysis-----------------------------------------------
################### Analysis1 (P4A1): Timecourse Analysis
1. Ensemble Timecourse Repeat Analysis.(Peakfind & Waittime)
2. Timecourse Info combined with behavior.

################### Analysis2 (P4A2): Neuro-Trace
1. PCA-based trace analysis
2. UMAP-based trace analysis
2. High-Dimension Trace Analysis



