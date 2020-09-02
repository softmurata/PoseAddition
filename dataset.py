from torch.utils.data import Dataset
import torchvision.transforms as transform
import torch
import numpy as np
import random
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    
    def __init__(self, image_size, dataset_dir, phase):
        super(ImageDataset, self).__init__()
        self.phase = phase
        self.image_size = image_size
        self.dataset_dir = dataset_dir
        
        self.transform = transform.ToTensor()  # tensor conversion for picture format
        
        self.images = []  # rgb images
        self.pose_images = []  # pose rgb images
        self.poses = []  # poses(intrinsic matrix and extrinsic matrix)
        
        self.create_dummy_data()
        # self.load_data()
        
    def create_dummy_data(self):
        dataset_num = 100
        images = [np.random.rand(self.image_size, self.image_size, 3) for _ in range(dataset_num)]
        pose_images = [np.random.rand(self.image_size, self.image_size, 3) for _ in range(dataset_num)]
        
        extrs = [np.random.rand(3, 4) for _ in range(dataset_num)]
        poses = []
        for extr in extrs:
            extr = extr.reshape((1, 12)).tolist()[0]
            vector = extr
            poses.append(np.array(vector))
            
        self.images = images
        self.pose_images = pose_images
        self.poses = poses
        
    def load_data(self):
        if self.phase == 'train':
            txt_path = 'train.txt'
        else:
            txt_path = 'val.txt'
        image_numbers = []
        with open(train_txt_path, 'r') as f:
            contents = f.readlines()
            for content in contents:
                image_numbers.append(content.split('\n')[0])
        
        rgb_dir = self.dataset_dir + 'rgb/'
        pose_img_dir = self.dataset_dir + 'pose_rgb/'
        pose_dir = self.dataset_dir + 'pose/'
                
        rgb_pathes = [rgb_dir + img + '.png' for img in image_numbers]
        pose_img_pathes = [pose_img_dir + img + '.png' for img in image_numbers]
        extrinsic_mat_pathes = [pose_dir + 'extrinsic{}.npy'.format(img) for img in image_numbers]
                
        images = [Image.open(image_path).convert('RGB') for image_path in image_pathes]
        images = [np.asarray(img).astype(np.uint8) for img in images]
        images = [cv2.resize(img, (self.image_size, self.image_size)) / 255 for img in images]
        
        pose_images = [Image.open(image_path).convert('RGB') for img in pose_img_pathes]
        pose_images = [np.asarray(img).astype(np.uint8) for img in pose_images]
        pose_images = [cv2.resize(img, (self.image_size, self.image_size)) / 255 for img in pose_images]
        
        poses = []
        extrinsic_mats = [np.load(extr) for extr in extrinsic_mat_pathes]
        extrinsic_mats = [extr.reshape((1, 12)) for extr in extrinsic_mats]
        
        for extr in extrinsic_mats:
            poses.append(extr)
        
        
        self.images = images
        self.pose_images = pose_images
        self.poses = poses
    
    def __len__(self):
        
        return len(self.images)
        
    def __getitem__(self, idx):
        # get real rgb image
        real_rgb = self.images[idx]
        # get pose image
        real_pose_rgb = self.pose_images[idx]
        # get poses
        real_pose = self.poses[idx]
        
        # to float tensor
        real_rgb = self.transform(real_rgb).float()
        real_pose_rgb = self.transform(real_pose_rgb).float()
        real_pose = torch.tensor(real_pose).float()
        
        return real_rgb, real_pose_rgb, real_pose



