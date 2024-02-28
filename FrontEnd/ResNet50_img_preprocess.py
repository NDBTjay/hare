import os, sys
import numpy
import random
import tensorflow as tf
import _pickle as pickle

DEFAULT_IMAGE_SIZE = 224
NUM_CHANNELS = 3
_RESIZE_MIN = 256
_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

# Creating tf record
class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that converts PNG to JPEG data.
        self._png_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_png(self._png_data, channels=3)
        self._png_to_jpeg = tf.image.encode_jpeg(image, format="rgb", quality=100)

        # Initializes function that converts CMYK JPEG data to RGB JPEG data.
        self._cmyk_data = tf.placeholder(dtype=tf.string)
        image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
        self._cmyk_to_rgb = tf.image.encode_jpeg(image, format="rgb", quality=100)

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def png_to_jpeg(self, image_data):
        return self._sess.run(self._png_to_jpeg, feed_dict={self._png_data: image_data})

    def cmyk_to_rgb(self, image_data):
        return self._sess.run(
            self._cmyk_to_rgb, feed_dict={self._cmyk_data: image_data}
        )

    def decode_jpeg(self, image_data):
        image = self._sess.run(
            self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data}
        )
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _process_image(filename, coder):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.GFile(filename, "rb") as f:
        image_data = f.read()

    # Decode the RGB JPEG.
    image = coder.decode_jpeg(image_data)

    # Check that image converted to RGB
    assert len(image.shape) == 3
    height = image.shape[0]
    width = image.shape[1]
    assert image.shape[2] == 3

    return image_data, height, width

def _central_crop(image, crop_height, crop_width):
    """Performs central crops of the given image list.

    Args:
      image: a 3-D image tensor
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.

    Returns:
      3-D tensor with cropped image.
    """
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = height - crop_height
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = width - crop_width
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

def _mean_image_subtraction(image, means, num_channels):
    """Subtracts the given means from each image channel.

    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.
      num_channels: number of color channels in the image that will be distorted.

    Returns:
      the centered image.

    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError("Input must be of size [height, width, C>0]")

    if len(means) != num_channels:
        raise ValueError("len(means) must match the number of channels")

    # We have a 1-D tensor of means; convert to 3-D.
    means = tf.expand_dims(tf.expand_dims(means, 0), 0)

    return image - means

def _smallest_size_at_least(height, width, resize_min):
    """Computes new shape with the smallest side equal to `smallest_side`.

    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.

    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: an int32 scalar tensor indicating the new width.
    """
    resize_min = tf.cast(resize_min, tf.float32)

    # Convert to floats to make subsequent calculations go smoothly.
    height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

    smaller_dim = tf.minimum(height, width)
    scale_ratio = resize_min / smaller_dim

    # Convert back to ints to make heights and widths that TF ops will accept.
    new_height = tf.cast(height * scale_ratio, tf.int32)
    new_width = tf.cast(width * scale_ratio, tf.int32)

    return new_height, new_width

def _aspect_preserving_resize(image, resize_min):
    """Resize images preserving the original aspect ratio.

    Args:
      image: A 3-D image `Tensor`.
      resize_min: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      resized_image: A 3-D tensor containing the resized image.
    """
    shape = tf.shape(input=image)
    height, width = shape[0], shape[1]

    new_height, new_width = _smallest_size_at_least(height, width, resize_min)

    return _resize_image(image, new_height, new_width)


def _resize_image(image, height, width):
    """Simple wrapper around tf.resize_images.

    This is primarily to make sure we use the same `ResizeMethod` and other
    details each time.

    Args:
      image: A 3-D image `Tensor`.
      height: The target height for the resized image.
      width: The target width for the resized image.

    Returns:
      resized_image: A 3-D tensor containing the resized image. The first two
        dimensions have the shape [height, width].
    """
    return tf.image.resize_images(
        image,
        [height, width],
        method=tf.image.ResizeMethod.BILINEAR,
        align_corners=False,
    )

def preprocess_image(
    image_buffer, output_height, output_width, num_channels, is_training=False
):
    
    # For validation, we want to decode, resize, then just crop the middle.
    image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
    image = _aspect_preserving_resize(image, _RESIZE_MIN)
    image = _central_crop(image, output_height, output_width)

    image.set_shape([output_height, output_width, num_channels])
    print(image)
    return _mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)

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

def resnet50_img_preprocess(input_img_filename, scale):
    # if not (len(sys.argv) == 3):
    #     print(
    #         "Args : <input_img_filename> <scale>",
    #         file=sys.stderr,
    #     )
    #     exit(1)
    # input_img_filename = sys.argv[1]
    # scale = int(sys.argv[2])
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    coder = ImageCoder()
    image_buffer, height, width = _process_image(input_img_filename, coder)
    image = preprocess_image(
        image_buffer=image_buffer,
        output_height=DEFAULT_IMAGE_SIZE,
        output_width=DEFAULT_IMAGE_SIZE,
        num_channels=NUM_CHANNELS,
        is_training=False,
    )
    name = input_img_filename.split('/')
    #print(name[-1])
    if os.path.exists("ResNet50_img_output") == False:
        os.makedirs('ResNet50_img_output')
    outp = sess.run([image], feed_dict={})
    saveFilePath = os.path.join(
            "ResNet50_img_output/", name[-1] + ".inp"
        )
    # dumpImageDataFloat(outp, saveFilePath, "w")
    dumpImageDataScale(outp, saveFilePath, "w", scale)


# if __name__ == "__main__":
#     main()