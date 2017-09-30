import numpy as np
from skimage import measure

def detect_labimg_boundaries(lab):
    """
    takes a labeled image.
    returns an image where pixval = number of neighboring pixels that had the same label
    TODO: generalize to g.t. 4 nearest neighbors
    """
    res_dim1 = lab[:-1]==lab[1:]
    res_dim2 = lab[:,:-1]==lab[:,1:]
    bimg = np.zeros(lab.shape)
    bimg[:-1] += res_dim1
    bimg[1:]  += res_dim1
    bimg[:,:-1] += res_dim2
    bimg[:,1:]  += res_dim2
    return bimg.astype('uint8')

def get_coords_and_labels(lab):
    rps = measure.regionprops(lab)
    coords = [np.mean(rp.coords, axis=0) for rp in rps]
    coords = np.array(coords)
    labels = [rp.label for rp in rps]
    labels = np.array(labels)
    return coords, labels
