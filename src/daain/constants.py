# Pytorch tensors obtained from PIL Images are always normalized between [0, 1]
# so  we can use Imagenet means in the given range for normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
