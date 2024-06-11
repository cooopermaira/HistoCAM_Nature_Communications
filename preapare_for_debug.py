# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:39:28 2024

@author: Cooper Maira
"""
import sys
import os
import xml.etree.cElementTree as ET

directory = "D:\\sim\\ColoredFrames Panel1-Case1155" ## sys.argv[1]
print(directory)
#build input
k = []
for file in os.listdir(directory):
    if file.endswith('.tif'):
        k.append(file)

k.sort(key = lambda x:(int(x.split('.')[0][8:10]) , int(x.split('.')[0][10:])))

with open(os.path.join(directory,'input.txt'),'w') as f:
    for file in k:
        f.write(os.path.join(directory,file).replace('\\','/')+'\n')
        
    
    #build config
    root = ET.Element("config")
    
    ET.SubElement(root,"processing", {"threads":"12"})
    
    element_io = ET.SubElement(root, "io")
    ET.SubElement(element_io, "input_images").text = f.name.replace('/','\\')
    ET.SubElement(element_io,"output_log").text = os.path.join(directory,"log_new.txt")
    ET.SubElement(element_io,"output_image").text = os.path.join(directory,"result.png")
    flatfield = ET.SubElement(element_io,"flat_field_images")
    ET.SubElement(flatfield,"twoX").text = "C:\\Users\\Max Cooper\\Documents\\pathcam\\calibration\\2x_wb.tif"
    
    root.append(ET.fromstring("<registration><detector><features><type>ORB</type></features><FREAK>False</FREAK></detector><image><crop>0.5</crop><scale>0.25</scale><interpolation>CUBIC</interpolation><real>false</real><debayer>true</debayer></image><matcher>BRUTEFORCE_HAMMING</matcher><estimator>RANSAC</estimator></registration>"))
    feature = root.find("registration/detector/features")
    ET.SubElement(feature,"params",{"nfeatures":"500", "scaleFactor":"1.0", "nlevels":"1", "edgeThreshold":"31","firstLevel":"0", "WTA_K":"2", "scoreType":"HARRIS_SCORE", "patchSize":"31", "fastThreshold":"20"})
    print(ET.tostring(root))
    tree = ET.ElementTree(root)
    tree.write(os.path.join(directory,"config.xml"))