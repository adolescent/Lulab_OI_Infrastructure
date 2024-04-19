
'''
Commonly used OI functions. The most useful & important part, make sure you know what you are adjusting.



########################## LOGS ###############################
(Actually you can do this on git = =)


ver 0.0.1 by ZR, function created. 2024/04/18


'''




import os
import pickle


#%% F1 join path. Actually just a rename of os function.
def join(path_A,path_B):
    new_path = os.path.join(path_A,path_B)
    return new_path

#%% F2 Get all file of same extend name.

def Get_File_Name(path,file_type = '.bin',keyword = ''):
    """
    Get all file names of specific type.

    Parameters
    ----------
    path : (str)
        Root path you want to cycle.
    file_type : (str), optional
        File type you want to get. The default is '.tif'.
    keyword : (str), optional
        Key word you need to screen file. Just leave '' if you need all files.

    Returns
    -------
    Name_Lists : (list)
       Return a list, all file names contained.

    """
    Name_Lists=[]
    for root, dirs, files in os.walk(path):
        for file in files:# walk all files in folder and subfolders.
            if root == path:# We look only files in root folder, subfolder ignored.
                if (os.path.splitext(file)[1] == file_type) and (keyword in file):# we need the file have required extend name and keyword contained.
                    Name_Lists.append(os.path.join(root, file))

    return Name_Lists

#%% F3 Get all subfolders.
def Get_Subfolders(root_path,method = 'Whole'):
    '''
    Input a path, return sub folders. Absolute path.

    Parameters
    ----------
    root_path : (str)
        The path you want to operate.
    method : ('Whole' or 'Relative')
        Determine whether we return only relative file path or the whole path.

    Returns
    -------
    subfolder_paths : (list)
        List of all subfolders. Absolute path provided to simplify usage.

    '''
    if method == 'Relative':
        subfolder_names = [name for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    elif method == 'Whole':
        subfolder_paths = [os.path.join(root_path, name) for name in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, name))]
    else:
        raise IOError('Method ILLEGAL.')

    return subfolder_paths


#%% F4 Mkdir, a little adjustment.
def mkdir(path,mute = False):
    '''
    This function will generate folder at input path. If the folder already exists, then do nothing.
    
    Parameters
    ----------
    path : (str)
        Target path you want to generate folder on.
    mute : (bool),optional
        Message will be ignored if mute is True. Default is False
        
    Returns
    -------
    bool
        Whether new folder is generated.

    '''
    isExists=os.path.exists(path)
    if isExists:
        # 如果目录存在则不创建，并提示目录已存在
        if mute == False:
            print('Folder',path,'already exists!')
        return False
    else:
        os.mkdir(path)
        return True
#%% F5 