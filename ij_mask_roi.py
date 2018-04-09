from PIL import Image
import read_roi
import numpy as np
from os.path import splitext
import tifffile
import cv2

# Open a tif image using tifffile
def openTifImage(image_path):
    """
    Opens a tif image as a numpy ndarray
    Uses tifffile.imread
    """
    assert image_path.lower().endswith(('.tif','.tiff')), "Image is not a TIF file"
    return tifffile.imread(image_path)


# Get objects from pixel encoded masks
def getMaskObjects(image_path, class_id):
    """
    INPUT:
        image_path: the image mask to bo processed where each object is encoded by pixel value.
                    eg. the pixel value for object 1 is 1, object 2 is 2, etc...
    OUTPUT:
        masks:  A bool (0 or 1) numpy nd array of shape [height, width, instance] where 
                each instance is a binary mask of a single object within the image mask.
        class_ids:  A 1D array of class IDs of the instance masks.
                    All classes are the same for any given mask.
    """
    img = openTifImage(image_path)
    masks = []
    class_ids = []
    if img.max() > 0:
        # Find each object and make all other pixels 0 while object pixels are 1
        # Number of objects is max pixel value
        for i in range(1,int(img.max())+1):
            cell = img.copy()
            cell[cell != i] = 0
            cell[cell == i] = 1
            masks.append(cell)
            class_ids.append(class_id)
        masks = np.asarray(masks) # convert list to ndarray
        masks = np.rollaxis(masks,0,3)  # roll axis to return correct shape ()
        class_ids = np.asanyarray(class_ids)
    else:
        #handle case where there are no objects
        masks.append(img.copy()) # append all zeros
        masks = np.asarray(masks) # convert list to ndarray
        masks = np.rollaxis(masks,0,3)  # roll axis to return correct shape ()
        
        class_ids.append(class_id) # 0 is the background class
        class_ids = np.asarray(class_ids)
        
    return masks, class_ids  

# Get objects from  IJ Roi.zip file
def getROIobjects(zip_path, class_id, height=None, width=None):  # image_path=None,
    """
    Opens IJ/Fiji ROIs from .zip file and returns individual boolean object masks.
    Must provide either an image_path or height/width to determine the image shape.
    For datasets with fixed image dimensions, provide height/width.
    For datasets with variable image dimensions (like stitched images), provide image_path
    INPUT:
        image_path: OPTIONAL - required to determine shape of original image
        height: OPTIONAL - required if no image_path is given
        width:  OPTIONAL - required if no image_path is given
        class_id: the id of the object class
        zip_path: the roi.zip file to be processed where each object is a seperate ROI
    OUTPUT:
        masks:  A bool (0 or 1) numpy nd array of shape [height, width, instance] where 
                each instance is a binary mask of a single object within the image mask.
        class_ids:
    """
    if zip_path is None:
        zip_path = splitext(image_path)[0] + '.zip'
        
    # Must have a .zip file
    assert zip_path.lower().endswith('.zip'), "Must be a .ZIP file"
    
#     if image_path is not None:
#         img = openTifImage(image_path)
                
#         if len(img.shape) is 2:
#             shape = img.shape
#         elif len(img.shape) is 3:
#             shape = (img.shape[1],img.shape[2])
#     else:
#         shape = (height, width)
    
    shape = (height,width)
    img = np.zeros(shape, dtype=np.int32)
    
    # Read ROIs with read_roi
    roi_dict = read_roi.read_roi_zip(zip_path)
    rois = list(roi_dict.items())
    masks = []
    class_ids = []
    if len(rois) > 0:
        # Only continue if we have some ROIs
        for n, roi in enumerate(rois):
            #get x y arrays
            x = roi[1]['x']
            y = roi[1]['y']
            #create array of points for polygon path
            points = np.vstack((x, y)).T
            points = points.reshape((-1,1,2))
            mask = cv2.fillPoly(img.copy(),[points],True,1)
            masks.append(mask)
            class_ids.append(class_id)
        # convert lists to ndarrays
        masks = np.asarray(masks) 
        masks = np.rollaxis(masks,0,3)  # roll axis to return correct shape
        class_ids = np.asanyarray(class_ids)
        
    return masks, class_ids