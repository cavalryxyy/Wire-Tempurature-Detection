import pandas as pd
import numpy as np
# =============================================================================
# filter the last wire position
# =============================================================================
def find_pos(df, my_loc):
    res = pd.DataFrame()
    res = df.iloc[my_loc[0]:my_loc[1]]
    return res

# =============================================================================
# filter air data
# =============================================================================
def filter_air(df):
    for i in np.arange(0,640):    
        df[i] = df[i].apply(lambda x: x if x >= 350 else 0)
    return df

# =============================================================================
# find max temp for each 640 rows
# =============================================================================
def loc_mp(df):
    xm_pos = []; r_max = []
    _rows = np.arange(0, 640)
    
    for r in (_rows):
        r_max.append(df[r].max())   
     
    xm_pos = np.where(r_max == np.min(np.array(r_max)))
    t_min  = np.min(np.array(r_max))
    return r_max, xm_pos, t_min

# =============================================================================
# 1.2 get temp of each wire
#   
# =============================================================================


# =============================================================================
# 1.1 extract_wire position 
# normalize the data t0 [0,255]
# adaptive threshold 
# return [temp data] -> array
# =============================================================================
import cv2
from skimage.measure import regionprops
from skimage import measure

class preprocess():
    def extract_wire(df):
        ori = df.to_numpy()
        ori_nor = (255.0/ori.max() * ori).astype('uint8')
        mask = cv2.adaptiveThreshold(ori_nor, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,blockSize = 27, C = 0)
        removal = mask
        removal[ori <= 200] = 0
        ori[removal == 0] = 0 
        return ori

    def filter_small(data, threshold = 3):
        
        temp = np.zeros(np.shape(data))
        temp[data != 0] = 255

        labels = measure.label(temp, connectivity = 2)
        props  = measure.regionprops(labels)
        Area = [prop.area for prop in props] 
        Bbox = [prop.bbox for prop in props]

        fal_points = [t[0] for t in filter(lambda a: a[1] < threshold, enumerate(Area, 1))]
        for label in fal_points:
            temp[labels == label] = 0
    
        # # cal particle num
        # label = measure.label(temp, connectivity=2)
        # prop  = measure.regionprops(label)
        # number = len(list(prop))
        # return temp, number
        
        fal_labels = np.array(fal_points) - 1  
        wire_label = [i for i in range(0, len(Area)) if i not in fal_labels]
        region = []
        for index in wire_label:
            region.append(Bbox[index]) 
        wire_num = len(Area) - len(fal_points)

        return temp, region, wire_num