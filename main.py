from Solution import workflow
from func import Compute as fC, Preprocess as fP
from visual import v

import numpy as np
import time

start = time.process_time()
path = 'C:/Users/Administrator/Desktop/WireTemp/'
csv_name = 'FLIR_A615.csv'
df = fP.load_csv(path, csv_name) 

# print('Time for loading data: %.2fs' %(time.process_time() - start))
TempData, Position, minTempValue, yloc = workflow(df)

print('Time for whole process %.2fs' %(time.process_time() - start))

v(df, Position, yloc)
# save all temp result 
