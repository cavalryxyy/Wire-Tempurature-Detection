# Wire-Tempurature-Detection
Extract the temperature data of each wire from the thermal imager raw data.

The motivation of this computer vision project is to location the minimal tempurature position of each wire. The raw data is generated from a thermal camera.

A pseudo color image (.png format) was generated from Python seaborn to present how the thermal camera data (.csv format file) looks like. For a standrad image taken from thermal camera, about 28~30 wires can be covered.

Hence, the workflow of this project can be descirbed as following steps:

1) Image segmantation. (Exluded the air and adjecant wires.)
2) Extract the tempurature data from each wire sub-image. 
3) Find the minial tempurature location of each wire.  (Research motivation)

2021/11/03
