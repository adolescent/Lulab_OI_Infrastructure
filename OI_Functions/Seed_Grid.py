'''
In this script, we will try to make a grid divide of imaging series, and try seed point method to calculate functional connection.
The Raw R frame need to be input to the class, for versatility of different systems.
'''
#%%
import Common_Functions as cf
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm



class Seed_Correlation_Grid(object):
    name = 'Seed Correlation of imaging data.'

    def __init__(self,frame,fps,grid_size = 16,save_folder='\\'):
        '''
        Initialization of processor. Here are brief descrip of vars
        frame : (np array)
        Raw input data frame. Must be in shape (Number*Height*Width). We need origional R value frame here.
        fps : (float)
        Capture frequency of graph. Bin need to be done before input. If you binned data, please calculate fps.



        '''
        self.raw_R_frame = frame
        self.fps = fps
        self.frame_num = frame.shape[0]
        self.grid_size = grid_size
        self.height = frame.shape[1]
        self.width = frame.shape[2]
        self.save_folder = save_folder

    def __len__(self): # define len(class)
        return len(self.raw_R_frame)

    def __getitem__(self,key): # define get class.
        return self.raw_R_frame[key,:,:]


    def Grid_dRR_Generator(self,method):
        ver_num = self.height//self.grid_size
        hor_num = self.width//self.grid_size
        if method == 'before':
            print('Do dR/R Before Averaging Grid. This way is quite SLOW = =')
            # get pixelwise drr matrix
            pix_drr_matrix = np.zeros(shape=self.raw_R_frame.shape,dtype='f8')
            for i in tqdm(range(self.height)):
                for j in range(self.width):
                    c_line = self.raw_R_frame[:,i,j]
                    base = c_line.mean()
                    pix_drr_matrix[:,i,j] = (c_line-base)/base
            # average time line of given grids
            print('Pix done, generating grid dR/R...')
            self.grid_drr = np.zeros(shape=(self.frame_num,ver_num,hor_num),dtype='f8')
            for i in tqdm(range(ver_num)):
                for j in range(hor_num):
                    c_grid = pix_drr_matrix[:,i*self.grid_size:(i+1)*self.grid_size,j*self.grid_size:(j+1)*self.grid_size].mean(1).mean(1)
                    self.grid_drr[:,i,j] = c_grid


        elif method =='after':
            print('Do dR/R After Averaging Grid.')
        else:
            raise ValueError('Invalid dR/R Method!')
        

    def Grid_Cutter(self,clip=2,font_size=10,drr_method = 'after'):
        '''
        This will cut graph into grids, get each grid's avr response curve, and plot grid graph, for seed selection.
        '''
        ver_num = self.height//self.grid_size
        hor_num = self.width//self.grid_size
        # check if it cannot be divided evenly
        if ver_num*self.grid_size != self.height:
            warnings.warn('Vertical cannot be divided evenly. Last grid ignored.')
        if hor_num*self.grid_size != self.width:
            warnings.warn('Horizontal cannot be divided evenly. Last grid ignored.')

        # draw example grid frame for seed selection
        print(f'Generating Grid graph, save at {self.save_folder}')
        base = self.raw_R_frame.mean(0)
        base = np.clip(base,base.mean()-base.std()*clip,base.mean()+base.std()*clip)
        fig, ax = plt.subplots(figsize=(10,10),dpi = 450)
        ax.imshow(base,cmap = 'gist_gray')
        # Draw the vertical lines
        for x in range(self.grid_size, self.grid_size*ver_num,self.grid_size):
            ax.axvline(x, color='red', linewidth=1,alpha = 0.5)
        # Draw the horizontal lines
        for y in range(self.grid_size, self.grid_size*hor_num,self.grid_size):
            ax.axhline(y, color='red', linewidth=1,alpha = 0.5)

        # Add the grid numbers on the left and right sides
        for i in range(0,ver_num):
            # Top
            ax.text((i+0.5)*self.grid_size,-font_size,str(i),ha='center',va='top',fontsize=font_size)
            # Bottom
            ax.text((i+0.5)*self.grid_size,self.height+font_size,str(i),ha='center',va='bottom', fontsize=font_size)
        for i in range(0,hor_num):
            # Left
            ax.text(-font_size/2,(i+0.5)*self.grid_size,str(i),ha='right',va='center',fontsize=font_size)
            # Right
            ax.text(self.width+font_size/2,(i+0.5)*self.grid_size,str(i),ha='left',va='center',fontsize=font_size)
        # Remove the axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()
        fig.savefig(cf.join(self.save_folder,'Grid_Info.png'))

        ## below is for generating drr matrix.
        # Average response of each grid

        # 
    def Seed_Determine(self,seed_coords):# generate seed correlation of given graph.
        ## GIVE IN SEQ Y,X!!!
        pass

    def Correlate_Grids(self): # correlate graph directly.
        pass
    def Correlate_Grid_Slide_Window(self,win_len,win_step): # give win_len and win_step in s is okay.
        pass

#%% Test run part
if __name__ == '__main__':
    wp = r'D:\YJX\spon_data\240719_RF_OIS_TEST\Run02_RF_VBar_20Hz_2ms\Preprocessed'
    frames = np.load(cf.join(wp,'Red.npy'))
    raw_R_frame = np.reshape(frames[:30180,:,:],(7545,4,256,256)).mean(1)
    fps = 5 # 4 binned 20Hz is 5Hz



    SCG = Seed_Correlation_Grid(frame = raw_R_frame,fps = 5,grid_size=8,save_folder=r'D:\YJX\spon_data\240719_RF_OIS_TEST\Run02_RF_VBar_20Hz_2ms\Preprocessed\Seed_Tests')
    SCG.Grid_Cutter()
#%% GPT Generated grid method.

# Assuming 'graph' is a 2D numpy array representing the graph
fig, ax = plt.subplots(figsize=(8, 8))
clip = 2
base = SCG.raw_R_frame.mean(0)
base = np.clip(base,base.mean()-base.std()*clip,base.mean()+base.std()*clip)
ax.imshow(base,cmap = 'gist_gray')
# Draw the vertical lines
for x in range(8, 256, 8):
    ax.axvline(x, color='red', linewidth=1,alpha = 0.5)
# Draw the horizontal lines
for y in range(8, 256, 8):
    ax.axhline(y, color='red', linewidth=1,alpha = 0.5)
# Add the grid numbers on the left and right sides
for i in range(0, 32):
    # Top
    ax.text(i * 8 + 4, -12, str(i), ha='center', va='top', fontsize=8)
    # Bottom
    ax.text(i * 8 + 4, 268, str(i), ha='center', va='bottom', fontsize=8)
    # Left
    ax.text(-20, i * 8 + 4, str(i), ha='right', va='center', fontsize=8)
    # Right
    ax.text(264, i * 8 + 4, str(i), ha='left', va='center', fontsize=8)
# Remove the axis ticks and labels
ax.set_xticks([])
ax.set_yticks([])

# Show the plot
plt.show()
