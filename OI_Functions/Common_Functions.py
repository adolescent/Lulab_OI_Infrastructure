
'''
Commonly used OI functions. The most useful & important part, make sure you know what you are adjusting.



########################## LOGS ###############################
(Actually you can do this on git = =)


ver 0.0.1 by ZR, function created. 2024/04/18


'''




import os
import pickle
import pandas as pd

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
def Get_Subfolders(root_path,keyword = '',method = 'Whole'):
    '''
    Input a path, return sub folders. Absolute path.

    Parameters
    ----------
    root_path : (str)
        The path you want to operate.
    keyword : (str),optional
        If keyword given, only folder have keyword will return.
    method : ('Whole' or 'Relative')
        Determine whether we return only relative file path or the whole path.

    Returns
    -------
    subfolder_paths : (list)
        List of all subfolders. Absolute path provided to simplify usage.

    '''
    all_path = []
    for root, dirs, files in os.walk(root_path):
        if root == root_path:
            for dir_name in dirs:
                if keyword in dir_name:
                    if method == 'Whole':
                        all_path.append(os.path.join(root, dir_name))
                    elif method == 'Relative':
                        all_path.append(dir_name)
    return all_path


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
#%% F5 Save in pickle way.
def Save_Variable(save_folder,name,variable,extend_name = '.pkl'):
    """
    Save a variable as binary data.

    Parameters
    ----------
    save_folder : (str)
        Save Path. Only save folder.
    name : (str)
        File name.
    variable : (Any Type)
        Data you want to save.
    extend_name : (str), optional
        Extend name of saved file. The default is '.pkl'.

    Returns
    -------
    bool
        Nothing.

    """
    if os.path.exists(save_folder):
        pass 
    else:
        os.mkdir(save_folder)
    real_save_path = save_folder+r'\\'+name+extend_name
    fw = open(real_save_path,'wb')
    pickle.dump(variable,fw)
    fw.close()
    return True
#%% F6 load pickled data.
def Load_Variable(save_folder,file_name=False):
    if file_name == False:
        real_file_path = save_folder
    else:
        real_file_path = save_folder+r'\\'+file_name
    if os.path.exists(real_file_path):
        pickle_off = open(real_file_path,"rb")
        loaded_file = pd.read_pickle(pickle_off)
        pickle_off.close()
    else:
        loaded_file = False

    return loaded_file

#%% F7 List Extend,used to extend and cut list.
def List_Extend(input_list,front,tail):
    """
    extend or cut list length.If extend, boulder value will be used.

    Parameters
    ----------
    input_list : (list)
        Input list. All element shall be number.
    front : (int)
        Length want to extend in the front. Negative number will cut list.
    tail : (int)
        Length want to extend at last. Negative number will cut list.

    Returns
    -------
    extended_list : (list)
        Cutted list.

    """
    front_element = input_list[0] # First element at front
    last_element = input_list[-1] # Last element at last
    # Process front first.
    if front >0:
        processing_list = [front_element]*front
        processing_list.extend(input_list)
    else:
        processing_list = input_list[abs(front):]
    # Then process tail parts.    
    if tail > 0:
        tail_list = [last_element]*tail
        processing_list.extend(tail_list)
    elif tail == 0:
        pass
    else:
        processing_list = processing_list[:tail]
    extended_list = processing_list

    return extended_list


# F8, kill caches, for umap it's sometimes necessary.
def kill_files(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print("failed on filepath: %s" % file_path)

def Kill_Cache(root_folder): 
    
    # root folder shall be anaconda folder

    # root_folder = r'C:\ProgramData\anaconda3'
    i =0
    for root, dirnames, filenames in os.walk(root_folder):
        for dirname in dirnames:
            if dirname == "__pycache__":
                try:
                    kill_files(root + "/" + dirname)
                    i += 1
                except Exception as e:
                    print("failed on %s", root)
    print(f'Total {i} cache folder killed.')