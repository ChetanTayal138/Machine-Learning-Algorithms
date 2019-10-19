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
            outimg[k] = convolve(input_img[i:i+f , j:j+f],filt)
            k = k + 1
    return outimg.reshape(outdims,outdims)
    

"""

We are going to feed an image with the following dimension : ( 3 x n x n )

First Channel is the RED channel 
Second Channel is the GREEN channel 
Third Channel is the BLUE channel 


"""

def get_outimg_rgb(input_img, filt, padding=0, stride=1, activation = 'relu', view = False):
    

    R_img = input_img[0, : , :]
    G_img = input_img[1, : , :]
    B_img = input_img[2, : , :]
    
    if(view == True):
        print(R_img.shape)
        print(G_img.shape)
        print(B_img.shape)

    R_conv = get_outimg(R_img, filt, padding = padding , stride = stride)
    G_conv = get_outimg(G_img, filt, padding = padding , stride = stride)
    B_conv = get_outimg(B_img, filt, padding = padding , stride = stride)


    RGB_img = R_conv + G_conv + B_conv 
    return conv_relu(RGB_img)
    






def conv_relu(img, rgb = False):
    unrolled = list(img.ravel())
    for i in range(len(unrolled)):
        if(unrolled[i] < 0):
            unrolled[i] = 0

    if(rgb == True):
        return np.array(unrolled).reshape((img.shape[0],img.shape[0],3))
    return np.array(unrolled).reshape((img.shape[0],img.shape[0]))


    




if __name__ == "__main__":



    np.random.seed(28)
#    img = np.random.rand(108).reshape(3,6,6)
    img = plt.imread("testimage.png")
    print(img.shape)
    plt.imshow(img)
    plt.show()
    filt = np.random.rand(4,4)
#    filt = np.array([[30., 0.,-30.], [30.,0.,-30.],[30.,0.,-30.]])
    final = get_outimg_rgb(img.T, filt,stride = 1)
    print(final)
    print(final.shape)
    plt.imshow(final.T)
    plt.show()


