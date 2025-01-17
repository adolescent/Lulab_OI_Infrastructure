'''
These functions provide multiple ways of pattern recognition.
Actually they're just normal method with a SHELL on it, so it's easier to understand.

'''


def Do_PCA(Z_frame,feature = 'Area',pcnum = 20):
    # INPUT FRAME SHALL BE IN SIZE (N_Frame,N_Cell)
    # feature can be 'Area' or 'Time', indicating which axis is feature and which is sample.
    pca = PCA(n_components = pcnum)
    data = np.array(Z_frame)
    if sample == 'Area':
        data = data.T# Use cell as sample and frame as feature.
    elif sample == 'Time':
        data = data
    else:
        raise ValueError('Sample method invalid.')
    pca.fit(data)
    PC_Comps = pca.components_# out n_comp*n_feature
    point_coords = pca.transform(data)# in n_sample*n_feature,out n_sample*n_comp
    return PC_Comps,point_coords,pca