import os.path
import io
import urllib2

from httplib import HTTPException
from ssl import CertificateError
from PIL import Image
#from resizeimage import resizeimage
import numpy as np
import matplotlib.image as mpimg
from scipy import misc

# Minimum size will eliminate single pixel and flickr missing photo images
MINIMUM_FILE_SIZE = 5000

#IMAGENET_LINKS_URL = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid="
IMAGE_DIRECTORY = "/home/litian/data/very_small"
#URL_DIRECTORY = os.path.join(IMAGE_DIRECTORY, "bad_urls")

RAW_IMAGE_HEIGHT = 256
RAW_IMAGE_WIDTH = 256

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224



def load_image_as_array(filepath):
    im = mpimg.imread(filepath)
    im = misc.imresize(im,(RAW_IMAGE_HEIGHT, RAW_IMAGE_WIDTH))
    if len(np.shape(im)) is 2:
        array = np.empty((RAW_IMAGE_HEIGHT, RAW_IMAGE_WIDTH, 3), dtype=np.uint8)
        array[:, :, :] = np.array(im)[:, :, np.newaxis]
    else:
        array = np.array(im)

    return array.astype(np.float32)


def create_one_hot_vector(index, length):
    assert length > 0, "One-hot vector length must be a positive number"
    assert 0 <= index < length, "Index (%s) must be between 0 and length(%s)" % (index, length)

    vector = np.zeros(length)
    vector[index] = 1
    return vector


def load_all_images(class_ids, num_images):
    num_classes = len(class_ids)
    all_images = []
    all_labels = []

    for index, class_id in enumerate(class_ids):
        class_path = os.path.join(IMAGE_DIRECTORY, class_id)
        if os.path.isdir(class_path):
            files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            num_class_files = min(len(files), num_images) #
            print num_class_files
            for n in range(0, num_class_files):
                image = load_image_as_array(os.path.join(class_path, files[n]))
                all_images.append(image)
                all_labels.append(create_one_hot_vector(index, num_classes))

    return np.array(all_images), np.array(all_labels)

def transform_images(images, randomize = False):
    assert IMAGE_WIDTH <= RAW_IMAGE_WIDTH
    assert IMAGE_HEIGHT <= RAW_IMAGE_HEIGHT

    transformed = []

    images = images.reshape(images.shape[0], RAW_IMAGE_HEIGHT, RAW_IMAGE_WIDTH, 3)

    for i in range(0, len(images)):
        image = images[i]
        if randomize:
            left_padding = np.random.randint(0, RAW_IMAGE_WIDTH - IMAGE_WIDTH)
            top_padding = np.random.randint(0, RAW_IMAGE_HEIGHT - IMAGE_HEIGHT)
            cropped_image = image[top_padding:top_padding + IMAGE_HEIGHT, left_padding:left_padding + IMAGE_WIDTH]

            if np.random.ranf() <= 0.5:
                cropped_image = cropped_image[:, ::-1, :]
        else:
            left_padding = (RAW_IMAGE_WIDTH - IMAGE_WIDTH)/2
            top_padding = (RAW_IMAGE_HEIGHT - IMAGE_HEIGHT)/2
            cropped_image = image[top_padding:top_padding + IMAGE_HEIGHT, left_padding:left_padding + IMAGE_WIDTH]
        transformed.append(cropped_image)

    transformed = np.asarray(transformed)
    return transformed.reshape(transformed.shape[0], IMAGE_HEIGHT*IMAGE_WIDTH, 3)


class DataSet(object):
    def __init__(self, images, labels):
        """Construct a DataSet using the given images and labels
        """

        assert images.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (images.shape,
                                                   labels.shape))
        self._num_examples = images.shape[0]

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns, 3] (assuming depth == 3)
        assert images.shape[3] == 3
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2], 3)

        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, random_crop=False):
        """Return the next `batch_size` examples from this data set.
        Images are cropped to final image size by selecting a random sample"""
        assert batch_size <= self._num_examples

        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size

        end = self._index_in_epoch
        raw_images = self._images[start:end]

        return transform_images(raw_images, randomize=random_crop), self._labels[start:end]


def create_datasets(class_ids, num_samples=1000, val_fraction=0.1, test_fraction=0.1):
    #num_sample
    """
    Creates training, validation, and test datasets from the given class ids using the desired proportions
    :param class_ids: ImageNet class ids of all classes to include
    :param num_samples: maximum sample images for each class
    :param val_fraction: fraction of images to put into validation set
    :param test_fraction: fraction of images to put into test set
    :return: training_set, validation_set, test_dataset
    """

    assert 0 <= val_fraction <= 0.25, "Validation fraction %s must be between 0 and 0.25" % val_fraction
    assert 0 <= test_fraction <= 0.25, "Test fraction %s must be between 0 and 0.25" % test_fraction

    all_images, all_labels = load_all_images(class_ids, num_samples)

    total_num_images = len(all_images)
    # Shuffle all images before splitting
    perm = np.arange(total_num_images)
    np.random.shuffle(perm)
    all_images = all_images[perm]
    all_labels = all_labels[perm]

    validation_size = int(total_num_images * val_fraction)
    test_size = int(total_num_images * test_fraction)

    validation_images = all_images[:validation_size]
    validation_labels = all_labels[:validation_size]

    test_images = all_images[validation_size:validation_size + test_size]
    test_labels = all_labels[validation_size:validation_size + test_size]

    train_images = all_images[validation_size + test_size:]
    train_labels = all_labels[validation_size + test_size:]

    # Mean normalization
    training_mean = np.mean(train_images)
    train_images -= training_mean
    validation_images -= training_mean
    test_images -= training_mean

    # Std dev normalization
    training_std_dev = np.std(train_images)
    train_images /= training_std_dev
    validation_images /= training_std_dev
    test_images /= training_std_dev

    train_dataset = DataSet(train_images, train_labels)
    validation_dataset = DataSet(validation_images, validation_labels)
    test_dataset = DataSet(test_images, test_labels)

    return train_dataset, validation_dataset, test_dataset