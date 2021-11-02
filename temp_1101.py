import cv2
import math

def filter_air2(df):
    ori = df.to_numpy()
    g_x = np.array((-1,0,1)).reshape((3,1))
    g_y = np.array((-1,0,1))
    dst_x = cv2.filter2D(ori, -1, g_x)
    dst_y = cv2.filter2D(ori, -1, g_y)
    dst = np.zeros(ori.shape, np.float32)
    w,h,_ = ori.shape 
    
    for i in range(w):
        for j in range(h):
            x_c, y_c = dst_x[i][j], dst_y[i][j]
            x_co, y_co = max(x_c), max(y_c)
            dst[i][j] = math.atan2(x_co, y_co)

    return dst


g_x = np.array((-1,0,1)).reshape((3,1))
g_y = np.array((-1,0,1))
dst_x = cv2.filter2D(ori, -1, g_x)
dst = (df)

# import seaborn as sns
# import matplotlib.pyplot as plt
# my_dpi = 24
# fig1 = plt.figure(figsize = (640/my_dpi, 480/my_dpi), 
#                   dpi = my_dpi)
# ax = sns.heatmap(df, cbar=False)
# ax.set(xticklabels = [], yticklabels = [])
# ax.tick_params(left = False, bottom = False)
# fig1.savefig(path + 'persudo_img.png', bbox_inches = 'tight')

dst_test = filter_air2(ori)

# check the histgram of temp data
ax2 = sns.histplot(ori.reshape((-1,1)), bins = 50
                   ,kde = True)

# =============================================================================
# normalize ori to 0-255
# =============================================================================
import pandas as pd
import numpy as np
path = 'C:/Users/xu.yuanyuan/Desktop/wire_tempurature/validate/'

df = pd.read_csv(path + 'FLIR_A615.csv', header = None)
ori = df.to_numpy()

ori_nor = (255.0/ori.max() * ori).astype('uint8')

mask = cv2.adaptiveThreshold(ori_nor, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY
    ,blockSize = 27, C = 0)

removal = mask
removal[ori <= 200] = 0
# cv2.imshow('test_ada_method', removal)

ori[removal == 0] = 0 

# =============================================================================
# main func
# =============================================================================
import pandas as pd
path = 'C:/Users/Administrator/Desktop/wire_tempurature/validate/'
df = pd.read_csv(path + 'FLIR_A615.csv', header = None)

from trans_temp_detection import preprocess as p
res = p.extract_wire(df)

# # for check only
# import cv2
# cv2.imshow('test_ada_method', res)

res_2 = p.filter_small(res, threshold = 640)
# cv2.imshow('test_ada_method', res_2)

