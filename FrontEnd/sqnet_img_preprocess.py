import os, sys
import numpy as np
import scipy.io
import scipy.misc
from PIL import Image
import time
import numpy

def imread_resize(path):
    # img_orig =imread(path)
    img_orig = Image.open(path).convert("RGB")
    img_orig = numpy.asarray(img_orig)
    img = scipy.misc.imresize(img_orig, (227, 227)).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img, img, img))
    return img, img_orig.shape

mean_pixel = np.array([104.006, 116.669, 122.679], dtype=np.float32)


def preprocess(image, mean_pixel):
    swap_img = np.array(image)
    img_out = np.array(swap_img)
    img_out[:, :, 0] = swap_img[:, :, 2]
    img_out[:, :, 2] = swap_img[:, :, 0]
    return img_out - mean_pixel

def dumpImageDataFloat(imgData, filename, writeMode):
    with open(filename, writeMode) as ff:
        for xx in numpy.nditer(imgData, order="C"):
            ff.write(str(xx) + " ")
        ff.write("\n\n")

def dumpImageDataScale(imgData, filename, writeMode, scale):
    with open(filename, writeMode) as ff:
        for xx in numpy.nditer(imgData, order="C"):
            xx = str(round(float(xx) * pow(2, int(scale))))
            ff.write(xx + " ")
        ff.write("\n\n")

def sqnet_img_preprocess(input_img_filename, scale):
    # if not (len(sys.argv) == 3):
    #     print(
    #         "Args : <input_img_filename> <scale>",
    #         file=sys.stderr,
    #     )
    #     exit(1)
    # input_img_filename = sys.argv[1]
    # scale = int(sys.argv[2])
    imgData, imgShape = imread_resize(input_img_filename)
    outp = preprocess(imgData, mean_pixel)
    name = input_img_filename.split('/')
    if os.path.exists("sqnet_img_output") == False:
        os.makedirs('sqnet_img_output')
    saveFilePath = os.path.join(
        "sqnet_img_output/", name[-1] + ".inp"
    )
    dumpImageDataScale(outp, saveFilePath, "w", scale)

# if __name__ == "__main__":
#     main()