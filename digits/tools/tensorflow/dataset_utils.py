from __future__ import absolute_import
from tensorpack import imgaug
from tensorpack import *
import multiprocessing
import cv2


_ID_TO_DATASET = {
    '20180223-022129-4cfc': 'Mnist',
    '20180313-090950-5e9b': 'ImageNet',
}


def fbresnet_augmentor():
    augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
        ]
    return augmentors


def get_imagenet_dataflow(
        datadir, name, batch_size,
        augmentors, parallel=None):
    """
    See explanations in the tutorial:
    http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html
    """
    assert name in ['train', 'val', 'test']
    assert datadir is not None
    assert isinstance(augmentors, list)
    isTrain = name == 'train'
    if parallel is None:
        parallel = min(40, multiprocessing.cpu_count() // 2)  # assuming hyperthreading
    if isTrain:
        ds = dataset.ILSVRC12(datadir, name, shuffle=True)
        ds = AugmentImageComponent(ds, augmentors, copy=False)
        if parallel < 16:
            logger.warn("DataFlow may become the bottleneck when too few processes are used.")
        ds = PrefetchDataZMQ(ds, parallel)
        ds = BatchData(ds, batch_size, remainder=False)
    else:
        ds = dataset.ILSVRC12Files(datadir, name, shuffle=False)
        aug = imgaug.AugmentorList(augmentors)

        def mapf(dp):
            fname, cls = dp
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            im = aug.augment(im)
            return im, cls
        ds = MultiThreadMapData(ds, parallel, mapf, buffer_size=2000, strict=True)
        ds = BatchData(ds, batch_size, remainder=True)
        ds = PrefetchDataZMQ(ds, 1)
    return ds


def get_data(batch_size, dataset_dir):
    name = None
    train = None
    test = None
    for k, v in _ID_TO_DATASET.iteritems():
        if k in dataset_dir:
            name = v

    if str(name).lower() in "mnist":
        train = BatchData(dataset.Mnist('train'), batch_size)
        test = BatchData(dataset.Mnist('test'), batch_size, remainder=True)
    elif str(name).lower() in "imagenet":
        augmentors = fbresnet_augmentor()
        train = get_imagenet_dataflow(dataset_dir, 'train', batch_size, augmentors)
        test = get_imagenet_dataflow(dataset_dir, 'val', batch_size, augmentors)

    return train, test
