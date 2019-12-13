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
    # Pad to account for kernel rotations
    padding = max(size)

    # Create 1D Gaussians
    pdfs = []
    for i in range(2):

        # Define padded size and shape of the kernel
        padded = 2 * padding + size[i]
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
    kernel = kernel[padding:-padding, padding:-padding]

    # Normalize probability density function
    kernel /= sum(sum(kernel))

    return kernel

def wave(angle=0.0, frequency=1.0, size=(100, 100)):
    """Generate Cosine wave kernel
    
    Keyword Arguments:
        angle {float} -- Rotation in degrees (default: {0.0})
        frequency {float} -- Number of waves per kernel length (default: {1.0})
        size {tuple} -- Kernel size (default: {(100, 100)})
    
    Returns:
        array -- Cosine wave kernel
    """
    # Pad to account for kernel rotations
    padding = max(size)

    # Create 2D wave pattern
    kernel = np.linspace(-np.pi, np.pi, size[0] + 2 * padding)
    kernel = np.tile(kernel, (size[1] + 2 * padding, 1))
    kernel = np.cos(frequency * kernel)

    # Rotate kernel and trim padding
    kernel = rotate(kernel, angle, reshape=False)
    kernel = kernel[padding:-padding, padding:-padding]

    return kernel
