# Modules
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import rotate
from scipy.stats import skewnorm

def gaussian(angle=0.0, size=(100, 100), skew=(0.0, 0.0), std=(1.0, 1.0)):
    """Generate asymmetric Gaussian kernel
    
    Keyword Arguments:
        angle {int} -- Rotation in degrees (default: {0.0})
        size {tuple} -- Kernel size (default: {(100, 100)})
        skew {tuple} -- Skews of the distribution (default: {(0.0, 0.0)})
        std {tuple} -- Standard deviations of the distribution (default: {(1.0, 1.0)})
    
    Returns:
        array -- Asymmetric Gaussian kernel
    """

    # Pad by a factor of 2 to account for kernel rotations
    padding = [(x // 2) + 1 for x in size]

    # Create 1D Gaussians
    pdfs = []
    for i in range(2):

        # Define padded size and shape of the kernel
        padded = 2 * padding[i] + size[i]
        pdf = np.arange(padded) - ((padded - 1) / 2)

        # Set the standard deviation
        pdf /= std[i]

        # Correct for the skew
        pdf *= skewnorm.std(skew[i])
        pdf += skewnorm.median(skew[i])

        # Generate probability density function
        pdf = skewnorm.pdf(pdf, skew[i])
        pdfs.append(pdf)
    
    # Create 2D Gaussian
    kernel = np.outer(*pdfs)

    # Rotate kernel and trim padding
    kernel = rotate(kernel, angle, reshape=False)
    kernel = kernel[padding[0]:-padding[0], padding[1]:-padding[1]]

    return kernel

angle = np.random.uniform(-180, 180)
size = (55, 55)
skew = np.random.uniform(-3, 3, 2)
std = np.random.uniform(1, 20, 2)
kernel = gaussian(angle=angle, size=size, skew=skew, std=std)
plt.figure()
plt.imshow(kernel)
plt.show()