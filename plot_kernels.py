# Modules
import matplotlib.pyplot as plt
import numpy as np
from kernel_palette import gaussian, wave

# Set up plot
fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)

# Generate three random Gaussian kernels
for i in range(3):
    kernel = gaussian(angle = np.random.uniform(-180, 180),
                      size = (50, 50),
                      skew = np.random.uniform(-3, 3, 2),
                      std = np.random.uniform(1, 20, 2))
    axs[0, i].imshow(kernel)
    axs[0, i].axis('off')

# Generate three random wave kernels
for i in range(3):
    kernel = wave(angle = np.random.uniform(-180, 180),
                  frequency = np.random.uniform(1, 10),
                  size = (50, 50))
    axs[1, i].imshow(kernel)
    axs[1, i].axis('off')

# Generate multiwave Gaussian patterns
for i in range(3):
    kernel_1 = gaussian(angle = np.random.uniform(-180, 180),
                        size = (50, 50),
                        skew = np.random.uniform(-3, 3, 2),
                        std = np.random.uniform(1, 20, 2))
    kernel_2 = wave(angle = np.random.uniform(-180, 180),
                    frequency = np.random.uniform(1, 10),
                    size = (50, 50))
    kernel_3 = wave(angle = np.random.uniform(-180, 180),
                    frequency = np.random.uniform(1, 10),
                    size = (50, 50))
    axs[2, i].imshow(kernel_1 * kernel_2 * kernel_3)
    axs[2, i].axis('off')
plt.show()