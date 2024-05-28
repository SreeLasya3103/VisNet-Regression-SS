import cv2
import numpy as np

def grad_strengths(img, ksize):
    final = np.zeros(img.shape, dtype=np.float32)
    
    h, w = img.shape[:2]
    
    for y in range(h):
        for x in range(w):
            r = (ksize-1)//2
            
            kern = np.zeros((ksize, ksize, 1), dtype=np.float32)
            
            for ky in range(-r, r+1):
                for kx in range(-r, r+1):
                    py = ky+y
                    px = kx+x
                    
                    if py >= 0 and px >= 0 and py < h and px < w:
                        kern[ky+r][kx+r] = img[py][px]
    
                        
            mean, std = cv2.meanStdDev(kern)
            mean = mean.item()
            std = std.item()
            
            if std != 0.0:
                norm_kern = (kern - mean) / std
                
                scharrX = cv2.Scharr(norm_kern, cv2.CV_32F, 1, 0, borderType=cv2.BORDER_REFLECT)
                scharrY = cv2.Scharr(norm_kern, cv2.CV_32F, 0, 1, borderType=cv2.BORDER_REFLECT)
                
                scharr = np.sqrt(scharrX**2 + scharrY**2)

            scharr = scharr[1:-1, 1:-1]

            # gaussian = cv2.getGaussianKernel(ksize-2, -1, cv2.CV_32F)
            # scharr = scharr * gaussian
            # final[y][x] = np.sum(scharr)
            
            final[y][x] = np.average(scharr)
            
    
    return final
 
image = cv2.imread('/home/feet/Documents/LAWN/SITE10_ORNT90_VIS7mi.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# image = cv2.GaussianBlur(image, (3,3), 0)
image = grad_strengths(image, 9)
# image = cv2.GaussianBlur(image, (7,7), 0)
# image = cv2.resize(image, (image.shape[1]//4, image.shape[0]//4), interpolation=cv2.INTER_NEAREST_EXACT)

# scharrX = cv2.Scharr(image, cv2.CV_32F, 1, 0)
# scharrY = cv2.Scharr(image, cv2.CV_32F, 0, 1)
# image = np.sqrt(scharrX**2 + scharrY**2)

image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
image = image.astype(np.uint8)
image = cv2.applyColorMap(image, cv2.COLORMAP_RAINBOW)

cv2.imshow('Scharr', image)
cv2.waitKey(0)
cv2.destroyAllWindows()