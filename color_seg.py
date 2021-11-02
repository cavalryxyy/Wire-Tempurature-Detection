# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 15:46:57 2021

@author: Xu.Yuanyuan
"""

import pandas as pd
import numpy as np
path = 'C:/Users/xu.yuanyuan/Desktop/wire_tempurature/Ori/'

df = pd.read_csv(path + '1.78S - Copy.csv', header = None)
# =============================================================================
# 1.8s ori
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns
ax = sns.heatmap(df)
plt.title('Tempterature heatmep (1.8 s)')
plt.xlabel('pixel'); plt.ylabel('pixel')

# =============================================================================
# filter 300
# =============================================================================
df1 = pd.DataFrame()
for i in np.arange(0,640):    
    df1[i] = df[i].apply(lambda x: x if x >= 350 else 999)

df2 = pd.DataFrame()
for i in np.arange(0,640):    
    df2[i] = df[i].apply(lambda x: x if x >= 350 else 0)

mask_1 = df1.to_numpy()[370:384,:]
rows_1 = np.arange(370, 385)

y_pos = []; r_min = []
for r in (rows_1):
    _min = df1[r].min()
    r_min.append( df1[r].min() )
    y_pos.append(
            [i for i, x in enumerate(df1[r]) if x == _min]
            )
# =============================================================================
# filter out the min of max 
# =============================================================================
mask_1 = df1.to_numpy()[370:384,:]
rows_1 = np.arange(0, 640)

x_pos = []; r_max = []
for r in (rows_1):
    _max = df2[r].max()
    r_max.append( df2[r].max() )
    x_pos.append(
            [i for i, x in enumerate(df2[r]) if x == _max]    
            )

xm_pos = np.where(r_max == np.min(r_max))
plt.plot(rows_1,r_max)
plt.grid(True, color = 'grey', linewidth = '0.8', linestyle = '-.')
plt.title('Wire temperature at 1.78 s'); plt.xlabel('Pixel Position'); plt.ylabel('Temperature [$^\circ$C]')
# =============================================================================
# 3D PLOT OF ALL 
# =============================================================================
