import numpy
from pathlib import Path
import json
import imageio as io
import cv2

def extract_grayscale_patches( img, shape, offset=(0,0), stride=(1,1) ):
    """ Adopted from: http://jamesgregson.ca/extract-image-patches-in-python.html """

    """Extracts (typically) overlapping regular patches from a grayscale image

    Changing the offset and stride parameters will result in images
    reconstructed by reconstruct_from_grayscale_patches having different
    dimensions! Callers should pad and unpad as necessary!

    Args:
        img (HxW ndarray): input image from which to extract patches

        shape (2-element arraylike): shape of that patches as (h,w)

        offset (2-element arraylike): offset of the initial point as (y,x)

        stride (2-element arraylike): vertical and horizontal strides

    Returns:
        patches (ndarray): output image patches as (N,shape[0],shape[1]) array

        origin (2-tuple): array of top and array of left coordinates
    """
    px, py = numpy.meshgrid( numpy.arange(shape[1]),numpy.arange(shape[0]))
    l, t = numpy.meshgrid(
        numpy.arange(offset[1],img.shape[1]-shape[1]+1,stride[1]),
        numpy.arange(offset[0],img.shape[0]-shape[0]+1,stride[0]) )
    l = l.ravel()
    t = t.ravel()
    x = numpy.tile( px[None,:,:], (t.size,1,1)) + numpy.tile( l[:,None,None], (1,shape[0],shape[1]))
    y = numpy.tile( py[None,:,:], (t.size,1,1)) + numpy.tile( t[:,None,None], (1,shape[0],shape[1]))
    return img[y.ravel(),x.ravel()].reshape((t.size,shape[0],shape[1])), (t,l)


def reconstruct_from_grayscale_patches( patches, origin, epsilon=1e-12 ):
    """ Adopted from: http://jamesgregson.ca/extract-image-patches-in-python.html """

    """Rebuild an image from a set of patches by averaging

    The reconstructed image will have different dimensions than the
    original image if the strides and offsets of the patches were changed
    from the defaults!

    Args:
        patches (ndarray): input patches as (N,patch_height,patch_width) array

        origin (2-tuple): top and left coordinates of each patch

        epsilon (scalar): regularization term for averaging when patches
            some image pixels are not covered by any patch

    Returns:
        image (ndarray): output image reconstructed from patches of
            size ( max(origin[0])+patches.shape[1], max(origin[1])+patches.shape[2])

        weight (ndarray): output weight matrix consisting of the count
            of patches covering each pixel
    """
    patch_width  = patches.shape[2]
    patch_height = patches.shape[1]
    img_width    = numpy.max( origin[1] ) + patch_width
    img_height   = numpy.max( origin[0] ) + patch_height

    out = numpy.zeros( (img_height,img_width) )
    wgt = numpy.zeros( (img_height,img_width) )
    for i in range(patch_height):
        for j in range(patch_width):
            out[origin[0]+i,origin[1]+j] += patches[:,i,j]
            wgt[origin[0]+i,origin[1]+j] += 1.0

    return out/numpy.maximum( wgt, epsilon ), wgt

def reconstruct_from_img_list(image_list_file, out_path=None):
    """
    Reconstruct image patches using image list containing indices and origin for patches in subpaths of image list file directory

    :param image_list_file: Path of 'image_list.json'
    :param out_path:    optional - Instead of returning reconstructed images as list write them into ouput directory
                                    useful for large image collection
    :return:    tuple containing lists containing the reconstructed images for each subpath
    """
    img_list_path = Path(image_list_file)

    img_list = json.load(open(img_list_path, 'r'))


    sub_dirs = [dir for dir in img_list_path.parent.iterdir() if dir.is_dir()]

    reconstructed = {}

    for dir in sub_dirs:
        if any(f.suffix == '.png' for f in dir.iterdir()):

            if out_path and not Path(out_path, dir.name).exists():
                Path(out_path, dir.name).mkdir(parents=True)
            else:
                reconstructed[dir.name] = []

            for img_name, content in img_list.items():
                print(dir.name + '/' + img_name)
                origin = content['origin']
                origin = (numpy.array(origin[0]), numpy.array(origin[1]))   # convert lists to tuple with ndarrays
                patch_indices = content['indices']
                img_shape = content['img_shape']

                patches = []

                for patch_index in patch_indices:
                    patches.append(cv2.imread(str(Path(dir, str(patch_index) + '.png')), cv2.IMREAD_GRAYSCALE))

                patches = numpy.array(patches)
                img, weights = reconstruct_from_grayscale_patches(patches, tuple(origin))
                img = img[:img_shape[0], :img_shape[1]].astype(numpy.uint8)     # crop to original shape

                if out_path:
                    io.imsave(Path(out_path, dir.name, img_name + '.png'), img)
                else:
                    reconstructed[dir.name].append(img)

    if len(reconstructed) > 0:
        return reconstructed
    else:
        return None

if __name__ is '__main__':
    reconstruct_from_img_list('/home/andreas/uni/thesis/src/output/out/image_list.json', out_path='/home/andreas/uni/thesis/src/output/reconstruction')