#shell stuff

import numpy as np
import skimage.io as io
import subprocess
import matplotlib.pyplot as plt
plt.ion()

def qsave(img):
    print(img.min(), img.max())
    io.imsave('qsave.tif', img)
    subprocess.call("open qsave.tif", shell=True)
