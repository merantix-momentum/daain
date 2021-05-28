import PIL
from PIL import ImageFile

# make PIL more flexible to allow preloading images
# source: https://stackoverflow.com/a/23575424/3622198.
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_image(handle):
    img = PIL.Image.open(handle)
    img.load()

    return img
