import numpy as np
import matplotlib.pyplot as plt 





"""
Convolution formula is [(n+2p-f)/s + 1] which gives dimensions of the output image after running convolution operation 
on it. 

n : dimensions of input image 
p : padding 
f : dimensions of the filter 
s : stride (typically set as 1)

If we have 5x5 image with no padding and stride as 1 with a filter of dimension 3x3 

  1 2 3 4 5
  1 2 3 4 5        1 1 1 
  1 2 3 4 5   x    1 1 1
  1 2 3 4 5        1 1 1 
  1 2 3 4 5

"""


def convolve(img,filt):
    return np.sum(img*filt)

def get_outdims(n,p,f,s):
    return  int((n+2*p-f)/s + 1)

def get_outimg(input_img, filt, padding=0, stride=1):
    n = input_img.shape[0]
    f = filt.shape[0]    
    outdims = get_outdims(n,padding,f,stride)
    outimg  = np.zeros(outdims**2)
    k = 0
    for i in range(0,outdims,stride):
        for j in range(0,outdims,stride):
            outimg[k] = convolve(a[i:i+f , j:j+f],filt)
            k = k + 1
    return outimg.reshape(outdims,outdims)
    




if __name__ == "__main__":


    a = np.arange(25).reshape(5,5)
    filt = np.ones((3,3))    
    print(a)
    print(filt)
    print(get_outimg(a,filt))




