import glob
import torch
import os
import numpy as np
import torch.utils.data as data
import cv2
from collections import OrderedDict
from torchvision.datasets.vision import VisionDataset


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class VAD_train_set(VisionDataset):
    # video anomaly detection training dataset class
    def __init__(self, root, clip_len, input_channel, resize_height=256, resize_width=256):
        super(VAD_train_set, self).__init__(root)
        self.clip_len = clip_len
        self.input_channel = input_channel
        self.gray = 1 if input_channel == 1 else 0
        self._resize_height = resize_height
        self._resize_width = resize_width

        # find classes, i.e., different videos, and map to indices
        self.classes, self.class_to_idx = self._find_classes(self.root)

        # create the dataset of video clips
        self.samples = self.make_vid_dataset(self.root, self.clip_len, self.class_to_idx)

        # check if no samples found
        if len(self.samples) == 0:
            raise RuntimeError(
                f"Found 0 files in subfolders of: {self.root}. Supported extensions are: {', '.join(IMG_EXTENSIONS)}")


    def _find_classes(self, dir):
        # find the class folders, i.e., different videos, in a dataset.

        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_vid_dataset(self, directory, clip_len, class_to_idx):
        # create the dataset by traversing the directories and finding valid video frames
        instances = []
        directory = os.path.expanduser(directory)

        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue

            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames)[:(len(fnames) - clip_len + 1)]:
                    path = os.path.join(root, fname)
                    if path.lower().endswith(IMG_EXTENSIONS):
                        instances.append((path, class_index))

        return instances

    def video_loader(self, path):

        # get the start file name and generate the rest
        vid_dir = os.path.dirname(path)
        frame_ext = os.path.splitext(os.path.basename(path))[1]

        # generate list of frame indices
        frame_idx_list = [os.path.splitext(os.path.basename(path))[0]]
        for i in range(self.clip_len-1):
            next_frame = str(int(frame_idx_list[-1])+1).zfill(5)
            frame_idx_list.append(next_frame)

        # generate empty video clip
        if self.gray == 1:
            data = np.zeros((self.clip_len, 1, self._resize_height, self._resize_width),
                        dtype=np.float32)
        elif self.gray == 0:
            data = np.zeros((self.clip_len, self.input_channel, self._resize_height, self._resize_width),
                        dtype=np.float32)

        # load frames and apply transforms
        for idx, frame_idx in enumerate(frame_idx_list):
            frame_path = os.path.join(vid_dir,frame_idx+frame_ext)
            image_decoded = cv2.imread(frame_path)
            image_resized = cv2.resize(image_decoded, (self._resize_width, self._resize_height))
            if self.gray == 1:
                image_resized = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY) # change to grayscale
            image_resized = (image_resized.astype(dtype=np.float32)/ 127.5) - 1.0 # normalize to range [-1, 1]

            if self.gray == 1:
                data[idx, 0, :, :] = image_resized
            elif self.gray == 0:
                data[idx, :, :, :] = image_resized.transpose(2, 0, 1)  # change to (C, H, W) format

        return data

    def __getitem__(self, index):

        path, _ = self.samples[index]
        images = self.video_loader(path)
        out = [index, images, np.zeros(1)]

        return out

    def __len__(self):
        return len(self.samples)


class VAD_test_set(data.Dataset):
    # video anomaly detection test dataset class
    def __init__(self, root, clip_len, input_channel, grayscale=1,  resize_height=256, resize_width=256):

        super(VAD_test_set, self).__init__()
        self.vids = OrderedDict()
        self.real_dir = root
        self.v_id = 0
        self.pre_vid = 0
        self.clip_len = clip_len # total length of a clip
        self.input_c = input_channel
        self._rsz_h = resize_height
        self._rsz_w = resize_width
        self.clip_init = 0
        self.gray = grayscale
        self.setup()

    def fetch_clip_idx(self, vid_name):
        # fetch the index of the initial frame of different clips based on manually-defined conditions
        init_idx_set = []
        for i in range(self.vids[vid_name]['length']-self.clip_len + 1):
            init_idx_set.append(i)

        self.vids[vid_name]['clip_init_idx'] = init_idx_set
        return len(init_idx_set)

    def setup(self):
        total_clip_num = 0
        self.vid_num = len(os.listdir(self.real_dir))
        videos = glob.glob(os.path.join(self.real_dir, '*'))
        for video in sorted(videos):
            vid_name = video.split('/')[-1]
            self.vids[vid_name] = {}
            self.vids[vid_name]['path'] = video
            self.vids[vid_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.vids[vid_name]['frame'].sort()
            self.vids[vid_name]['length'] = len(self.vids[vid_name]['frame']) # get the length of each video
            total_clip_num += self.fetch_clip_idx(vid_name)

        self.length = total_clip_num  # compute the number of clips in total
        print(self.length)


    def generate_clip(self):

        self.pre_vid = self.v_id

        # generate a single video clip
        vid_info_list = list(self.vids.values())

        # randomly select a video and a start point in it
        self.v_id = self.v_id % self.vid_num
        vid_info = vid_info_list[self.v_id]

        # generate a data space to store a clip
        if self.gray == 1:
            data = np.zeros((self.clip_len, 1, self._rsz_h, self._rsz_w),
                        dtype=np.float32)
        elif self.gray == 0:
            data = np.zeros((self.clip_len, self.input_c, self._rsz_h, self._rsz_w),
                        dtype=np.float32)

        # fetch the idx of the initial frame of this clip
        input_init = vid_info['clip_init_idx'][self.clip_init]
        for idx, frame_id in enumerate(range(input_init, input_init + self.clip_len)):
            # print(self.real_frame_idx, frame_id)
            try:
                img = cv2.imread(vid_info['frame'][frame_id])
            except:
                continue

            img_rsz = cv2.resize(img, (self._rsz_w, self._rsz_h))
            if self.gray == 1:
                img_rsz = cv2.cvtColor(img_rsz, cv2.COLOR_BGR2GRAY) # change to grayscale
            img_rsz = img_rsz.astype(dtype=np.float32)
            img_rsz = (img_rsz / 127.5) - 1.0 # to range [-1,1]
            if self.gray == 1:
                data[idx, 0, :, :] = img_rsz
            elif self.gray == 0:
                data[idx, :, :, :] = img_rsz.transpose(2, 0, 1)  # to (C, H, W)

        if input_init == vid_info['clip_init_idx'][-1]:
            self.v_id += 1
            self.clip_init = 0
        else:
            self.clip_init += 1

        return data, input_init

    def __getitem__(self, idx):

        images, input_init_idx  = self.generate_clip()
        img_idx = torch.arange(input_init_idx, input_init_idx + self.clip_len)
        out = [idx, images, img_idx, (self.pre_vid+1)]

        return out

    def __len__(self):
        return self.length
