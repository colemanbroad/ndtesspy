import numpy as np
import heapq


# ----

def show_basic_voronoi_img(coords, img):
    from scipy.spatial import Voronoi, voronoi_plot_2d
    vor = Voronoi(coords)
    fig = voronoi_plot_2d(vor)
    fig.gca().imshow(img.T)
    return fig, vor

# ----

NEIGHBOR_GRID = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])

def _inbounds(img):
    xmax, ymax = img.shape
    def f(x,y):
        return 0<=x<xmax and 0<=y<ymax
    return f

def distance(x0,y0,f0, x1,y1,f1):
    return np.abs(f0+f1) + 1.0

# ---- 

def initialize_heapq(labimg, distimg):
    heap=[]
    heapq.heapify(heap)
    inbounds = _inbounds(labimg)

    for x in range(labimg.shape[0]):
        for y in range(labimg.shape[1]):
            l = labimg[x,y]
            if l != 0:
                for dx,dy in NEIGHBOR_GRID:
                    x2,y2 = x+dx, y+dy
                    if inbounds(x2,y2):
                        d1 = distimg[x,y]
                        d2 = distimg[x2, y2]
                        dist = distance(x,y,d1, x2,y2,d2)
                        heapq.heappush(heap, (dist, x2, y2, l))
    return heap

def initialize_heapq_coords(coords, labels, distimg):
    """
    An alternative heapq initialization, if you want to initialize from a number
    of labeled seed points, instead of growing pre-existing label regions
    """
    boundary_value = labels.max() + 1

    vorimg = np.zeros(distimg.shape)
    coords = coords.astype('int')
    vorimg[coords[:,0], coords[:,1]] = labels
    vorimg = vorimg.astype('uint16')
    vorimg[0,:]  = boundary_value
    vorimg[-1,:] = boundary_value
    vorimg[:,0]  = boundary_value
    vorimg[:,-1] = boundary_value

    heap=[]
    heapq.heapify(heap)

    for i in range(len(coords)):
        x,y = coords[i]
        l   = labels[i]
        for dx,dy in NEIGHBOR_GRID:
            x2,y2 = x+dx, y+dy
            d1 = distimg[x,y]
            d2 = distimg[x2, y2]
            dist = distance(x,y,d1, x2,y2,d2)
            heapq.heappush(heap, (dist, x2, y2, l))
    return heap, vorimg

def build_vorimg(heap, vorimg_0, distimg):
    """
    tessellate an image by growing existing labeled regions
    """
    vorimg = vorimg_0.copy()

    inbounds = _inbounds(vorimg)
    while len(heap) > 0:
        d,x,y,l = heapq.heappop(heap)
        if vorimg[x,y]==0:
            vorimg[x,y] = l
            for dx, dy in NEIGHBOR_GRID:
                x2,y2 = x+dx, y+dy
                if inbounds(x2,y2):
                    nl = vorimg[x2, y2]
                    if nl == 0 : # then unclaimed
                        d1 = distimg[x,y]
                        d2 = distimg[x2, y2]
                        nd = distance(x,y,d1, x2,y2,d2) + d
                        heapq.heappush(heap, (nd, x2, y2, l))
    return vorimg

# ----

def tessellate_labimg(labimg, distimg=None):
    if distimg is None:
        distimg = np.zeros_like(labimg).astype('float32')
    heap = initialize_heapq(labimg, distimg)
    vorimg = build_vorimg(heap, labimg, distimg)
    return vorimg

