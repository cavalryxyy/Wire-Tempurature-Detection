import seaborn as sns
import cv2
import matplotlib.pyplot as plt
import numpy as np

def v(df, position, yloc):
    fig = plt.figure(figsize=(12, 9))
    ax = sns.heatmap(df)
    
    x_coor = [value[0][0] for value in position]
    y_coor = [value[1] for value in yloc]
    text  = np.arange(len(x_coor)) + 1 
    
    for x, y, txt in zip (x_coor, y_coor, text):
        plt.annotate(txt, (x, y), color = 'blue', weight = 'bold', size = 16)
    
    plt.title('IR camera tempurature visualization')
    ax.set(xticklabels = [], yticklabels = [])
    ax.tick_params(left = False, bottom = False)
    plt.show()


