'''
These functions will align graph captured to brain atlas.
User need to provide bregma and lambda location.

'''
#%%
import cv2
import numpy as np
import time
from Brain_Atlas.Atlas_Mask import Mask_Generator
import matplotlib.pyplot as plt
import copy

avr = cv2.imread(r'D:\_DataTemp\Graph_Affine\avr_graph.png',0)

#%%

class Match_Pattern(Mask_Generator): # success parent class

    name = 'Align Stack to pattern'

    def __init__(self,avr,bin=4): # avr are averaged graph,
        super().__init__(bin)
        # self.MG = Mask_Generator(bin=bin)
        self.height,self.width = self.idmap.shape
        print(f'After Align Resolution:{self.height}x{self.width}')
        self.avr = ((avr/avr.max())*255).astype('u1')
        self.lbd = 420/bin # pix distance between lambda and bregma. 4.2mm
        self.pad_num = int(800/bin) # number of pad used for graph rotation

        self.idmap_sym = copy.deepcopy(self.idmap)
        self.idmap_sym[self.idmap_sym>32] -= 31

    def Select_Anchor(self): # select anchor 

        self.anchor = []# anchor points. in seq Y,X
        img = cv2.cvtColor(self.avr,cv2.COLOR_GRAY2RGB) # set graph to color

        def selector(event,x,y,flags, param):# define point selector first
            if event == cv2.EVENT_LBUTTONDOWN:
                self.anchor.append([y,x])
                if len(self.anchor) == 1:
                    color = (0, 0, 255) # red for bregma
                elif len(self.anchor) == 2:
                    color = (0,255,0)
                else:
                    color = (255,0,0)

                # draw point on graph
                cv2.circle(img, (x, y), 3, color, -1) 
                cv2.imshow("Image", img)

        cv2.imshow("Image", img)
        cv2.setMouseCallback("Image", selector)
        flagbreg = False
        flaglamb = False
        flagother = False
        while len(self.anchor) < 5:
            if len(self.anchor)==0 and flagbreg == False:
                print('Selecting Bregma..')
                flagbreg = True
            if len(self.anchor)==1 and flaglamb == False:
                print('Selecting Lambda..')
                flaglamb = True
            if len(self.anchor)>1 and flagother == False:
                print('Selecting Other Points..')
                flagother = True
            cv2.waitKey(1)
        cv2.destroyAllWindows()
        ### After this we can select 5 points. Then we need to determine whether ok to continue.
        cv2.imshow("Graph", img)
        cv2.waitKey(1)
        # Checkpoint
        # user_input = input("Do you want to continue? (Y/N): ").strip().upper()
        # if user_input != 'Y':
        #     cv2.destroyAllWindows()
        #     raise ValueError("Process terminated by user.")
        # else:
        print("Continuing with the process...")
        cv2.destroyAllWindows()
        self.anchor = np.array(self.anchor)
        self.realbreg = self.anchor[0,:]
        self.reallamb = self.anchor[1,:]
        self.point_demo = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        print(f"Select points saved in self.anchor")

    def Fit_Align_Matrix(self): # this part will return 

        # calculate rescale index.
        real_lbd = np.linalg.norm(self.realbreg-self.reallamb)
        self.scale = self.lbd/real_lbd

        # rotation and rescale
        self.slope,_ = np.polyfit(self.anchor[:,1],self.anchor[:,0] , 1)
        self.rot_angle = 90+np.degrees(np.arctan(self.slope))
        if self.realbreg[0]>self.reallamb[0]:
            print('Graph is upside down, be aware.')
            self.rot_angle += 180

        # transform.
        padded_graph = np.pad(self.avr, ((self.pad_num, self.pad_num), (self.pad_num, self.pad_num)), mode='constant', constant_values=0)
        self.breg_pad = (int(self.realbreg[0]+self.pad_num),int(self.realbreg[1]+self.pad_num))
        # extend graph for 
        rot_mat = cv2.getRotationMatrix2D((self.breg_pad[1],self.breg_pad[0]),self.rot_angle,self.scale)

        result_raw = cv2.warpAffine(padded_graph, rot_mat,padded_graph.shape[1::-1], flags=cv2.INTER_LINEAR)
        # if self.realbreg[0]>self.reallamb[0]: # add another 180 to rotation.
        #     result = cv2.rotate(result, cv2.ROTATE_180)
        LU_point = [MP.breg_pad[0]-MP.breg[0],MP.breg_pad[1]-MP.breg[1]]
        self.result = result_raw[LU_point[0]:LU_point[0]+self.height,LU_point[1]:LU_point[1]+self.width]
        plt.imshow(self.result,cmap='gray')
        plt.imshow(self.idmap_sym,alpha = 0.2,cmap='jet')

    
    def Transform_Series(self,stacks):
        print('Fitting')




if __name__ == '__main__':

    MP = Match_Pattern(avr,4)
    MP.Select_Anchor()
    MP.Fit_Align_Matrix()
    # print(MP.anchor)


