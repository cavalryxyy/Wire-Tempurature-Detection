import cv2
from skimage.measure import regionprops
from skimage import measure
import pandas as pd
import numpy as np

# 1.1 extract_wire position 
def extract_wire(df):
    ori = df.to_numpy()
    ori_nor = (255.0/ori.max() * ori).astype('uint8')

    mask = cv2.adaptiveThreshold(ori_nor, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY
        ,blockSize = 27, C = 0)
    removal = mask
    removal[ori <= 200] = 0
    # cv2.imshow('test_ada_method', removal)

    ori[removal == 0] = 0 
    return ori

# 1.2 filter small parts, return temp mask and weld spot num
def global_filter_small(data, threshold = 3):
    
    temp = np.zeros(np.shape(data))
    temp[data != 0] = 255

    labels = measure.label(temp, connectivity = 2)
    props  = measure.regionprops(labels)
    # particles_num1 = len(list(props)) # check initial partical num 
    # print(particles_num1)
    Area = [prop.area for prop in props] 
    Bbox = [prop.bbox for prop in props]

    # Find objects > threshold, get their labels, set them to 0-clear.
    fal_points = [t[0] for t in filter(lambda a: a[1] < threshold, enumerate(Area, 1))]
    # delete small parts in temp 
    for label in fal_points:
        temp[labels == label] = 0
    
    # create label for wire to store bounding box coordidates 
    fal_labels = np.array(fal_points) - 1  
    wire_label = [i for i in range(0, len(Area)) if i not in fal_labels] 
    region = []
    for index in wire_label:
        region.append(Bbox[index]) 
    
    wire_num = len(Area) - len(fal_points)

    res = data
    res[temp == 0] = 0
    
    return res, region, wire_num

# 1.3 iterate region 
def get_coor_pairs(region):
    return [region[0], region[2]]

# 1.4 filter adjacent wires
#   return wire position
def local_filter_small(array, threshold = 640):
    # if only one label, return 
    temp = np.zeros(np.shape(array))
    temp[array != 0] = 255
    labels = measure.label(temp, connectivity = 2)
    props  = measure.regionprops(labels)

    if len(props) == 1: 
        return array
    else: 
        Area = [prop.area for prop in props] 
        fal_points = [t[0] for t in filter(lambda a: a[1] < threshold, enumerate(Area, 1))]
        for label in fal_points:
            temp[labels == label] = 0
        res = array
        res[temp == 0] = 0
        
    return res

# 1.4 find the wire 
def find_pos(array, my_loc):
    return array[my_loc[0]:my_loc[1],:]

# 1.5 find location on wire 
# argmin_temp(max_temp of cols)
def loc_mp(arr):
    # temp max for each cols, store max temp value of 640 cols
    xm_pos, r_max = [], []  
    _rows = np.arange(0, 640)
    
    for r in (_rows):
        r_max.append(np.max(arr[:,r]))
    
    r_max = list(map(lambda x: x if x!=0 else np.inf, r_max))
    
    xm_pos = np.where(r_max == np.min(np.array(r_max)))
    t_min  = np.min(np.array(r_max)) 
       
    return r_max, xm_pos, t_min


path = '/Users/xuyuanyuan/Desktop/WireTemp/'
df = pd.read_csv(path + 'FLIR_A615.csv', header = None)
res = extract_wire(df)
wire, region, wire_num = global_filter_small(res, threshold = 640)

#### compute min temp ####
yloc = []
for sub_region in region:
    yloc = get_coor_pairs(sub_region)
    sub_wire = find_pos(wire, yloc)
    _wire = local_filter_small(sub_wire)
    r_max, xm_pos, t_min = loc_mp(_wire)
# =============================================================================
# 0.1 LOAD CSV POSITION
# 0.2 CHOOSE SAVE POSITION
# =============================================================================

# =============================================================================
# 
# =============================================================================
# report detected wires 
# =============================================================================
# 
# =============================================================================
# for item in region:
#     compute_min_temp()
#     return min_location min_temp_value
# =============================================================================
# 
# =============================================================================
# 