# encoding: utf-8
from __future__ import print_function, absolute_import
import os.path as osp
import glob
import re
import urllib
import zipfile

from ..utils.data import BaseImageDataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class PersonX(BaseImageDataset):
    """
    personX (source domain): only consains training samples
    Dataset statistics:
    # identities: 700
    # images: 20280 (train)
    # cams: 6
    """

    dataset_dir = 'personX'

    def __init__(self, root, verbose=True):
        super(PersonX, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'image_train/')

        self._check_before_run()
        train = self._process_dir(self.train_dir, relabel=True)

        if verbose:
            print("=> personX loaded")
            self.print_dataset_statistics(train)

        self.train = train
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d*)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 700  # pid == 0 means background
            assert 1 <= camid <= 12 # num_camid=6, max_cam_index=12
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        return dataset
