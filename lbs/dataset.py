"""
This module procures data loaders for various datasets.

Note datasets are cached.
"""
import errno
import os
from os.path import exists, join
import tarfile
from urllib.request import urlretrieve

import subprocess
import json
from absl import flags
import numpy as np

import torchvision.datasets
import torchvision.transforms as T
import torch
from torch.utils import data
from torch.utils.data.dataset import Subset
from PIL import Image
import lbs.drn_data_transforms as transforms
from lbs.get_cityscapes_data.create_fake_cityscapes import get_fake_cityscapes

from . import log
from .utils import WrappedIterator, compute_num_mini_mini_batches

flags.DEFINE_string('dataroot', './data', 'data caching directory')

flags.DEFINE_float(
    'train_val_split',
    0.7,
    'proportion of training data to use for training; '
    'rest goes to validation',
    lower_bound=0,
    upper_bound=1)

flags.DEFINE_bool('use_fake_data', False, 'use fake data for faster tests')

LANGUAGE_MODELS = ['wikitext-2', 'penntreebank']
LANGUAGE_MODEL_URLS = {
    'wikitext-2': \
    'https://www.dropbox.com/s/ok7bvo34dynfvwj/wikitext-2.tar.gz?dl=1'}


def split_dataset(dataset):
    """Split the dataset by a fixed ratio specified in flags"""
    train_len = int(len(dataset) * flags.FLAGS.train_val_split)
    val_len = len(dataset) - train_len
    if not flags.FLAGS.dataset in LANGUAGE_MODELS:
        train, val = data.dataset.random_split(dataset, [train_len, val_len])
    else:
        train_indices = range(train_len)
        val_indices = range(train_len, len(dataset))
        train, val = [
            Subset(dataset, train_indices),
            Subset(dataset, val_indices)
        ]
    torch.save(train.indices,
               os.path.join(log.logging_directory(), 'train_indices.pth'))
    torch.save(val.indices,
               os.path.join(log.logging_directory(), 'val_indices.pth'))
    return train, val


class Dictionary:
    """ Maps a word to an index in a corpus vocabulary """

    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """ add a word to the dictionary """
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    """ Represents a text corpus with per-word tokenization"""

    def __init__(self, name, download=True):
        self.dictionary = Dictionary()
        self.path = path = os.path.join(flags.FLAGS.dataroot, name)
        if download:
            self.download(name)
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids

    def _check_exists(self):
        train_f = os.path.join(self.path, 'train.txt')
        valid_f = os.path.join(self.path, 'valid.txt')
        test_f = os.path.join(self.path, 'test.txt')
        return os.path.exists(train_f) and os.path.exists(valid_f) \
            and os.path.exists(test_f)

    def download(self, name):
        """ Download a corpus to the dataroot directory..."""
        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(flags.FLAGS.dataroot, name))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        assert name in LANGUAGE_MODEL_URLS.keys(), "must have URL for corpus!"
        url = LANGUAGE_MODEL_URLS[name]
        file_path = os.path.join(flags.FLAGS.dataroot, name + '.tar.gz')
        actual_fname, _ = urlretrieve(url, file_path)
        tar = tarfile.open(actual_fname, mode='r:gz')
        tar.extractall(flags.FLAGS.dataroot)
        tar.close()


def _create_corpus_loader(dataset, batch_size, endless=True):
    """ Creates an iterator for a language corpus """
    dataset = dataset[:]  # Turn to tensor
    nb = compute_num_mini_mini_batches(batch_size)
    chunk_bs = batch_size // nb
    nbatch = dataset.size(0) // chunk_bs
    # Trim off extra data
    dataset = dataset[:].narrow(0, 0, nbatch * chunk_bs)
    dataset = dataset.view(chunk_bs, -1).t().contiguous()

    bptt = 35

    def _seq_iter():
        for i in range(0, dataset.size(0) - 1, nb * bptt):
            # Return nb chunks per iteration
            chunk_inputs = []
            chunk_targets = []
            for j in range(i, i + nb * bptt, bptt):
                seq_len = min(bptt, len(dataset) - j - 1)
                chunk_inputs.append(dataset[j:j + seq_len])
                chunk_targets.append(dataset[j + 1:j + 1 + seq_len].view(-1))
            yield chunk_inputs, chunk_targets

    return WrappedIterator(
        _seq_iter, callable=True, len=len(dataset) // bptt, endless=endless)


def _create_sampled_loader(dataset, dataset_name, batch_size, endless=True):
    """
    Creates an iterator for an unordered dataset by sampling with replacement
    """
    if endless:
        sampler = data.sampler.WeightedRandomSampler(
            np.ones(len(dataset)), batch_size)
    else:
        sampler = None


    if dataset_name == 'svhn':
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=False,
            num_workers=0)
    else:
        dataloader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=False,
            num_workers=0)
    if endless:
        return WrappedIterator(
            dataloader, callable=False, len=len(dataset), endless=endless)
    return dataloader


def create_loader(dataset, dataset_name, batch_size, endless=True):
    """
    Returns a dataloader for the dataset by looking at flags.FLAGS.dataset.
        If endless is True, the iterator repeatedly passes over
        the same dataset.
    """
    if dataset_name in LANGUAGE_MODELS:
        return _create_corpus_loader(dataset, batch_size, endless=endless)
    return _create_sampled_loader(dataset, dataset_name, batch_size, endless)


def from_name(name):
    """
    Create a dataset on-demand from its lower case name.
    iterators: if True, return iterators over dataset. If false,
               return dataset objects alone.
    Returns: train, val, test data iterators,
             loss criterion, and dataset params
    """
    if name == 'cifar10':
        return cifar10()
    if name == 'cifar100':
        return cifar100()
    if name == 'mnist':
        return mnist()
    if name == 'cityscapes':
        return cityscapes()
    if name == 'svhn':
        return svhn()
    if name in LANGUAGE_MODELS:
        return _load_language_model(name)
    raise ValueError('dataset {} not recognized'.format(name))

import torch.utils.data as data
class SVHNFusion(data.Dataset):
    def __init__(self):
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize((.5, .5, .5), (.5, .5, .5))
        ])

        self.train = torchvision.datasets.SVHN(
            root=flags.FLAGS.dataroot,
            split='train',
            download=True,
            transform=transform)
        self.extra = torchvision.datasets.SVHN(
            root=flags.FLAGS.dataroot,
            split='extra',
            download=True,
            transform=transform)
        self.n_train = len(self.train)
        self.n_extra = len(self.extra)

    def __getitem__(self, index):
        if index < self.n_train:
          return self.train[index]
        else:
          return self.extra[index - self.n_train]

    def __len__(self):
        return self.n_train + self.n_extra

def svhn():
    """" train, test, num_classes for svhn house digits classification """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    train = SVHNFusion()
    test = torchvision.datasets.SVHN(
        root=flags.FLAGS.dataroot,
        split='test',
        download=True,
        transform=transform)
    dataset_params = {'num_classes': 10}
    return train, test, dataset_params


def cifar10():
    """returns train, test, num_classes for CIFAR10"""
    # precomputed from cifar10 training data
    # per https://github.com/meliketoy/wide-resnet.pytorch/blob/master/main.py
    mean = np.array([125.3, 123.0, 113.9]) / 255.0
    std = np.array([63.0, 62.1, 66.7]) / 255.0

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    transform_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train = torchvision.datasets.CIFAR10(
        root=flags.FLAGS.dataroot,
        train=True,
        download=True,
        transform=transform_train)
    test = torchvision.datasets.CIFAR10(
        root=flags.FLAGS.dataroot,
        train=False,
        download=False,
        transform=transform_test)

    dataset_params = {'num_classes': 10}
    return train, test, dataset_params


def cifar100():
    """returns train, test, num_classes for CIFAR10"""
    # precomputed from cifar10 training data
    # per https://github.com/meliketoy/wide-resnet.pytorch/blob/master/main.py
    mean = np.array([125.3, 123.0, 113.9]) / 255.0
    std = np.array([63.0, 62.1, 66.7]) / 255.0

    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    transform_test = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train = torchvision.datasets.CIFAR100(
        root=flags.FLAGS.dataroot,
        train=True,
        download=True,
        transform=transform_train)
    test = torchvision.datasets.CIFAR100(
        root=flags.FLAGS.dataroot,
        train=False,
        download=False,
        transform=transform_test)

    dataset_params = {'num_classes': 100}
    return train, test, dataset_params


def mnist():
    """returns train, test, num_classes for MNIST"""
    # per https://github.com/pytorch/examples/blob/master/mnist
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307, ), (0.3081, ))])
    train = torchvision.datasets.MNIST(
        flags.FLAGS.dataroot, train=True, download=True, transform=transform)
    test = torchvision.datasets.MNIST(
        flags.FLAGS.dataroot, train=False, download=True, transform=transform)

    dataset_params = {'num_classes': 10}
    return train, test, dataset_params


class SegList(torch.utils.data.Dataset):
    """
    Stores cityscapes dataset.
    Taken from https://github.com/fyu/drn/
    """

    def __init__(self,
                 data_dir,
                 phase,
                 transforms_list,
                 list_dir=None,
                 out_name=False):
        self.list_dir = data_dir if list_dir is None else list_dir
        self.data_dir = data_dir
        self.out_name = out_name
        self.phase = phase
        self.transforms = transforms_list
        self.image_list = None
        self.label_list = None
        self.bbox_list = None
        self.read_lists()

    def __getitem__(self, index):
        data_img = [Image.open(join(self.data_dir, self.image_list[index]))]
        if self.label_list is not None:
            data_img.append(
                Image.open(join(self.data_dir, self.label_list[index])))
        data_img = list(self.transforms(*data_img))
        if self.out_name:
            if self.label_list is None:
                data_img.append(data_img[0][0, :, :])
            data_img.append(self.image_list[index])
        return tuple(data_img)

    def __len__(self):
        return len(self.image_list)

    def read_lists(self):
        """reads image and label locations"""
        image_path = join(self.list_dir, self.phase + '_images.txt')
        label_path = join(self.list_dir, self.phase + '_labels.txt')
        assert exists(image_path)
        self.image_list = [line.strip() for line in open(image_path, 'r')]
        if exists(label_path):
            self.label_list = [line.strip() for line in open(label_path, 'r')]
            assert len(self.image_list) == len(self.label_list)


def cityscapes():
    """returns training, test, and num classes for the Cityscapes dataset"""
    print("FAKE DATA FLAG:", flags.FLAGS.use_fake_data)
    data_dir = flags.FLAGS.dataroot + "/cityscapes"
    required_files = [
        "gtFine", "leftImg8bit", "train_images.txt", "train_labels.txt",
        "val_images.txt", "val_labels.txt", "test_images.txt", "info.json"
    ]
    if os.path.isdir(data_dir) and \
       (set(required_files) <= set(os.listdir(data_dir))):
        print("Cityscapes data already downloaded")
    else:
        if flags.FLAGS.use_fake_data:
            if os.path.isdir(flags.FLAGS.dataroot + "/cityscapes"):
                print("Fake cityscapes already downloaded")
            else:
                print("Downloading fake cityscapes")
                get_fake_cityscapes(flags.FLAGS.dataroot)
        else:
            print("Downloading cityscapes dataset")
            lbs_src_dir = os.path.dirname(__file__)
            cityscapes_script = os.path.join(
                lbs_src_dir, "get_cityscapes_data", "get_cityscapes.sh")
            subprocess.check_call(cityscapes_script, shell=True)
    crop_size = 896
    info = json.load(open(join(data_dir, 'info.json'), 'r'))
    normalize = transforms.Normalize(mean=info['mean'], std=info['std'])
    t_train = [
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomScale((0.25, 0.25)),
        transforms.ToTensor(), normalize
    ]
    train_dataset = SegList(data_dir, 'train', transforms.Compose(t_train))
    test_dataset = SegList(
        data_dir, 'val',
        transforms.Compose([
            transforms.RandomCrop(crop_size),
            transforms.RandomScale((0.25, 0.25)),
            transforms.ToTensor(), normalize
        ]))
    num_classes = 19
    return train_dataset, test_dataset, {'num_classes': num_classes}


def _load_language_model(name):
    """ Loads train, test, and vocab len for some language corpus """
    corpus = Corpus(name)
    return corpus.train, corpus.test, {'vocab_len': len(corpus.dictionary)}
