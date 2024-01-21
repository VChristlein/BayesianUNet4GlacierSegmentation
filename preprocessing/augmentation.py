import cv2
def flip_rotate(img):
    imgs_out = []
    augmentations = ['', '_hflip', '_r90', '_r180', '_r270', '_hflip_r90', '_hflip_r180', '_hflip_r270']
    img_flip = cv2.flip(img, 0)
    for img_ in [img, img_flip]:   # Rotate original and flipped image
        imgs_out.append(img_)
        imgs_out.append(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
        imgs_out.append(cv2.rotate(img, cv2.ROTATE_180))
        imgs_out.append(cv2.rotate(img_, cv2.ROTATE_90_CLOCKWISE))

    return imgs_out, augmentations
def flip(img):
    imgs_out = []
    augmentations = ['', '_hflip']
    imgs_out.append(img)
    imgs_out.append(cv2.flip(img, 0))

    return imgs_out, augmentations
