# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:16:19 2026

@author: Jesslyn Wong
"""

import cv2
import numpy as np
from matplotlib import pyplot as pt


# ==============
# Read Image
# ==============
img1 = cv2.imread("C:/Users/Jesslyn Wong/BCS/Y2S3/CSC2014/CSC 2014_Group Assignment_April Sem 2026/CSC 2014_Group Assignment_April Sem 2026/Converted Paper (8)/001.png", 0)


# ==============
# binary image (thresholding)
# ==============
ret, binImg1 = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# ==============
# Histogram Projection (horizontal, vertical)
# ==============
# sum of black pixels per row (counting ->) 
horizontal_hist = np.sum(binImg1 == 0, axis=1)

pt.figure() 
pt.plot(horizontal_hist)
pt.title("Horizontal Histogram")

pt.show()

# sum of black pixels per column (counting ^)
vertical_hist = np.sum(binImg1 == 0, axis=0)

pt.figure() 
pt.plot(vertical_hist)
pt.title("Vertical Histogram")

pt.show()


# ==============
# Convert binImg -> True/False (optional?)
# ==============
list1 = horizontal_hist > 0

# ==============
# Extraction Process
# ==============
# list of consectutive zeros, returns (index, amount)
def getConsecutiveZeros(aList):
    consecutiveZeros = []
    count = 0
    start = None
    
    for i, v in enumerate(aList):
        if not v:
            if count == 0:
                start = i
            count += 1
        
        else:
            if count > 0:
                consecutiveZeros.append((start, count))
                count = 0  
                
    # for last run of consecutive zeros
    if count > 0:
        consecutiveZeros.append((start, count))
                
    return consecutiveZeros


# filter the zeros more than 10
def getRangeZeros(aList):
    ranges = []
    
    for i, c in aList:
        if c > 11:
            ranges.append((i, i + c))
            
    return ranges


#get range of paragraph
top = []
bottom = []
def doTop(aList):
    for l, r in aList:
        top.append(r)        
            
    return

def doBottom(aList):
    iteration = 0
    for l, r in aList:
        if iteration > 0:
            bottom.append(l)  
        iteration += 1
            
    return   


# extract
def extractParagraphs(aRange, anImg):
    for l, r in aRange:
        par = anImg[l:r, :]
        pt.figure() 
        pt.imshow(par, cmap="gray")
        pt.title("jbg")

        pt.show()        
    
# ==============
# Run extraction process
# ==============       
myZeros = getConsecutiveZeros(list1)  
myRanges = getRangeZeros(myZeros)  
doTop(myRanges) # start of paragraph
doBottom(myRanges)    # end of paragraph
paragraphRanges = list(zip(top, bottom))    # put start and end together
extractParagraphs(paragraphRanges, binImg1)         
