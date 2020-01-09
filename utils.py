import cv2
from tqdm import tqdm
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random
import torch
import torch.utils.data as data
from scipy.misc import imread, imresize
import xml.etree.ElementTree as ET
import glob
import tensorflow as tf
import scipy.misc
import numpy as np
import os

from PIL import Image

from torch.utils.data import Dataset

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x

# Grayscale preprocessing
def color2gray(root,save):

    file_list = glob.glob(os.path.join(root, 'JPEGImages/*.jpg'))
    print(len(file_list))

    if not os.path.isdir(save):
        os.makedirs(save)
    for image in tqdm(file_list):
        # print(image[-10:])

        rgb_img = cv2.imread(image, 0)
        # gray_img = cv2.cvtColor(cv2.imread(image, 0), cv2.COLOR_GRAY2BGR)
        # dst_img = np.hstack((rgb_img, gray_img))
        print(os.path.join(save, image[-15:]))
        cv2.imwrite(os.path.join(save, image[-15:]), rgb_img)
        # print(image[-15:])

# color2gray('./data/tgt_data/', './data/src_data/JPEGImages')
# print('color2gray already done')

def data_load(path, subfolder, transform, batch_size, shuffle=False, drop_last=True):
    dset = datasets.ImageFolder(path, transform)
    ind = dset.class_to_idx[subfolder]

    n = 0
    for i in range(dset.__len__()):
        if ind != dset.imgs[n][1]:
            del dset.imgs[n]
            n -= 1
        n += 1

    return torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


def Gray(tensor, size, batch_size):
    R = tensor[:, 0, :, :]
    G = tensor[:, 1, :, :]
    B = tensor[:, 2, :, :]
    tensor = 0.299 * R + 0.587 * G + 0.114 * B

    return tensor.view(batch_size, 1, size, size)

# def tensor2image(tensor):
#     image = 127.5*(tensor[0].cpu().float().numpy() + 1.0)
#     if image.shape[0] == 1:
#         image = np.tile(image, (3,1,1))
#     return image.astype(np.uint8)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# def initialize_weights(*models):
#     for model in models:
#         for module in model.modules():
#             if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
#                 nn.init.kaiming_normal(module.weight)
#                 if module.bias is not None:
#                     module.bias.data.zero_()
#             elif isinstance(module, nn.BatchNorm2d):
#                 module.weight.data.fill_(1)
#                 module.bias.data.zero_()

# class DataLoader(data.Dataset):
#     def __init__(self, data_path, train, transform, random_crops=0):
#         self.data_path = data_path
#         self.transform = transform
#         self.random_crops = random_crops
#         self.trainval = train
#
#         self.__init_classes()
#         self.names, self.labels = self.__dataset_info()
#
#     def __getitem__(self, index):
#         x = imread(self.data_path + '/JPEGImages/' + self.names[index] + '.jpg', mode='RGB')
#         x = Image.fromarray(x)
#
#         x = self.transform(x)
#         y = self.labels[index]
#
#         return x, y
#
#     def __len__(self):
#         return len(self.names)
#
#     def __dataset_info(self):
#         # annotation_files = os.listdir(self.data_path+'/Annotations')
#         with open(self.data_path + '/ImageSets/Main/' + self.trainval + '.txt') as f:
#             annotations = f.readlines()
#
#
#         annotations = [n[:-1] for n in annotations]
#
#         names = []
#         labels = []
#         for af in annotations:
#             if len(af) != 6:
#                 continue
#             filename = os.path.join(self.data_path, 'Annotations', af)
#             tree = ET.parse(filename + '.xml')
#             objs = tree.findall('object')
#             num_objs = len(objs)
#
#             boxes = np.zeros((num_objs, 4), dtype=np.uint16)
#             boxes_cl = np.zeros((num_objs), dtype=np.int32)
#
#             for ix, obj in enumerate(objs):
#                 bbox = obj.find('bndbox')
#                 # Make pixel indexes 0-based
#                 x1 = float(bbox.find('xmin').text) - 1
#                 y1 = float(bbox.find('ymin').text) - 1
#                 x2 = float(bbox.find('xmax').text) - 1
#                 y2 = float(bbox.find('ymax').text) - 1
#
#                 cls = self.class_to_ind[obj.find('name').text.lower().strip()]
#                 boxes[ix, :] = [x1, y1, x2, y2]
#                 boxes_cl[ix] = cls
#
#             lbl = np.zeros(self.num_classes)
#             lbl[boxes_cl] = 1
#             labels.append(lbl)
#             names.append(af)
#
#         return np.array(names), np.array(labels).astype(np.float32)
#
#     def __init_classes(self):
#         self.classes = ('__background__','aeroplane', 'bicycle', 'bird', 'boat',
#                          'bottle', 'bus', 'car', 'cat', 'chair',
#                          'cow', 'diningtable', 'dog', 'horse',
#                          'motorbike', 'person', 'pottedplant',
#                          'sheep', 'sofa', 'train', 'tvmonitor')
#         self.num_classes = len(self.classes)
#         self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))


# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.FileWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        self.writer.add_summary(summary, step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""

        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        self.writer.add_summary(summary, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

##voc12 load
EXTENSIONS = ['.jpg', '.png']

def load_image(file):
    return Image.open(file)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS)

def image_path(root, basename, extension):
    return os.path.join(root, basename+extension)

def image_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

class VOC12(Dataset):

    def __init__(self, root, input_transform=None, target_transform=None):
        self.images_root = os.path.join(root, 'JPEGImages')
        self.labels_root = os.path.join(root, 'SegmentationClass')

        self.filenames = [image_basename(f)
            for f in os.listdir(self.labels_root) if is_image(f)]
        self.filenames.sort()

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        filename = self.filenames[index]

        with open(image_path(self.images_root, filename, '.jpg'), 'rb') as f:
            image = load_image(f).convert('RGB')
        with open(image_path(self.labels_root, filename, '.png'), 'rb') as f:
            label = load_image(f).convert('P')

        if self.input_transform is not None:
            image = self.input_transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.filenames)

######transforms
class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert isinstance(tensor, torch.LongTensor), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight = None):
        super().__init__()
        self.loss = nn.NLLLoss(weight)

    def forward(self, output, target):
        return self.loss(F.log_softmax(output, dim=1), target)


def colormap(n):
    cmap=np.zeros([n, 3]).astype(np.uint8)

    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1<<(7-j))*((i&(1<<(3*j))) >> (3*j))
            g = g + (1<<(7-j))*((i&(1<<(3*j+1))) >> (3*j+1))
            b = b + (1<<(7-j))*((i&(1<<(3*j+2))) >> (3*j+2))

        cmap[i,:] = np.array([r, g, b])

    return cmap

class Colorize:

    def __init__(self, n=22):
        self.cmap = colormap(256)
        self.cmap[n] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(1, len(self.cmap)):
            mask = gray_image[0] == label

            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image
