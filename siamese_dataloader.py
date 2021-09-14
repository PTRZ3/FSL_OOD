import torch
from torch.utils.data import Dataset
import numpy as np
import random


class CIFAR10_TrainPair(Dataset):
    def __init__(
            self,
            cifar10_dataset,
            train_size = 10000,
            seen_classes = list(range(6)),
            seed = 2021
    ):
        self.cifar10_dataset = cifar10_dataset
        self.train_size = train_size
        self.seen_classes = seen_classes
        self.seed = seed
        #dictionary to store image id of each class
        self.class_dict = {k:[] for k in seen_classes}       
        for class_id in seen_classes:
            self.class_dict[class_id] = [idx for idx, item in enumerate(self.cifar10_dataset) if item[1] == class_id]

    def __len__(self) -> int:
        return self.train_size
    
    def __getitem__(self, index: int):
        image1 = None
        image2 = None
        image1_class = None
        image2_class = None
        is_diff = None
        is_ood = 0
        
        
        random.seed(self.seed + index)
        # get image from same class
        if index % 2 == 1:
            is_diff = 0
            class_id = random.choice(self.seen_classes)
            image1_class = class_id
            image2_class = class_id
            image_id1 = random.choice(self.class_dict[class_id])
            image_id2 = random.choice(self.class_dict[class_id])
            image1 = self.cifar10_dataset[image_id1][0]
            image2 = self.cifar10_dataset[image_id2][0]
        # get image from different class
        else:
            is_diff = 1
            image1_class = random.choice(self.seen_classes)
            image2_class = random.choice(self.seen_classes)
            while image1_class == image2_class:
                image2_class = random.choice(self.seen_classes)
            image_id1 = random.choice(self.class_dict[image1_class])
            image_id2 = random.choice(self.class_dict[image2_class])
            image1 = self.cifar10_dataset[image_id1][0]
            image2 = self.cifar10_dataset[image_id2][0]
            
        #output
        image1_class = torch.from_numpy(np.array([image1_class], dtype=np.int))
        image2_class = torch.from_numpy(np.array([image2_class], dtype=np.int))
        is_diff = torch.from_numpy(np.array([is_diff], dtype=np.int))
        is_ood = torch.from_numpy(np.array([is_ood], dtype=np.int))

        return image1, image2, image1_class, image2_class, is_diff, is_ood

    
    
    
class CIFAR10_ValPair(Dataset):
    def __init__(
            self,
            cifar10_dataset,
            val_size = 10000,
            seen_classes = list(range(6)),
            ood_classes = [6,7],
            seed = 2021
    ):
        self.cifar10_dataset = cifar10_dataset
        self.val_size = val_size
        self.seen_classes = seen_classes
        self.ood_classes = ood_classes
        self.seed = seed
        #dictionary to store image id of each class
        self.seen_class_dict = {k:[] for k in seen_classes}    
        self.ood_class_dict = {k:[] for k in ood_classes}  
        for class_id in seen_classes:
            self.seen_class_dict[class_id] = [idx for idx, item in enumerate(self.cifar10_dataset) if item[1] == class_id]     
        for class_id in ood_classes:
            self.ood_class_dict[class_id] = [idx for idx, item in enumerate(self.cifar10_dataset) if item[1] == class_id]

    def __len__(self) -> int:
        return self.val_size
    
    def __getitem__(self, index: int):
        image1 = None
        image2 = None
        image1_class = None
        image2_class = None
        is_diff = None
        is_ood = None
        
        
        random.seed(self.seed + index)
        if index % 4 == 1:
        # is_diff = 0 for same class, is_diff = 1 for diff classes
        # get image from different class, both from seen classes (25%)
            is_diff = 1
            is_ood = 0
            image1_class = random.choice(self.seen_classes)
            image2_class = random.choice(self.seen_classes)
            while image1_class == image2_class:
                image2_class= random.choice(self.seen_classes)
            image_id1 = random.choice(self.seen_class_dict[image1_class])
            image_id2 = random.choice(self.seen_class_dict[image2_class])
            image1 = self.cifar10_dataset[image_id1][0]
            image2 = self.cifar10_dataset[image_id2][0]
        elif index % 4 == 2:
        # get image from different class, one from seen classes and one from ood classes (25%)
            is_diff = 1
            is_ood = 1
            image1_class = random.choice(self.seen_classes)
            image2_class = random.choice(self.ood_classes)
            image_id1 = random.choice(self.seen_class_dict[image1_class])
            image_id2 = random.choice(self.ood_class_dict[image2_class])
            image1 = self.cifar10_dataset[image_id1][0]
            image2 = self.cifar10_dataset[image_id2][0]
        else:
        #get image from same class, both from seen classes (50%)
            is_diff = 0
            is_ood = 0
            class_id = random.choice(self.seen_classes)
            image1_class = class_id
            image2_class = class_id
            image_id1 = random.choice(self.seen_class_dict[class_id])
            image_id2 = random.choice(self.seen_class_dict[class_id])
            image1 = self.cifar10_dataset[image_id1][0]
            image2 = self.cifar10_dataset[image_id2][0]
        

        #output
        image1_class = torch.from_numpy(np.array([image1_class], dtype=np.int))
        image2_class = torch.from_numpy(np.array([image2_class], dtype=np.int))
        is_diff = torch.from_numpy(np.array([is_diff], dtype=np.int))
        is_ood = torch.from_numpy(np.array([is_ood], dtype=np.int))

        return image1, image2, image1_class, image2_class, is_diff, is_ood
    
    
class CIFAR10_TrainValPair_OE(Dataset):
    def __init__(
            self,
            cifar10_dataset,
            oe_dataset,
            size = 10000,
            seen_classes = list(range(6)),
            oe_classes = list(range(5)),
            seed = 2021
    ):
        self.cifar10_dataset = cifar10_dataset
        self.oe_dataset = oe_dataset
        self.size = size
        self.seen_classes = seen_classes
        self.oe_classes = oe_classes
        self.seed = seed
        #dictionary to store image id of each class
        self.seen_class_dict = {k:[] for k in seen_classes}       
        for class_id in seen_classes:
            self.seen_class_dict[class_id] = [idx for idx, item in enumerate(self.cifar10_dataset) if item[1] == class_id]
        self.oe_class_dict = {k:[] for k in oe_classes}       
        for class_id in oe_classes:
            self.oe_class_dict[class_id] = [idx for idx, item in enumerate(self.oe_dataset) if item[1] == class_id]

    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, index: int):
        image1 = None
        image2 = None
        image1_class = None
        image2_class = None
        is_diff = None
        is_ood = None
          
        random.seed(self.seed + index)
        # get image from same class
        if index % 2 == 1:
            is_diff = 0
            is_ood = 0
            class_id = random.choice(self.seen_classes)
            image1_class = class_id
            image2_class = class_id
            image_id1 = random.choice(self.seen_class_dict[class_id])
            image_id2 = random.choice(self.seen_class_dict[class_id])
            image1 = self.cifar10_dataset[image_id1][0]
            image2 = self.cifar10_dataset[image_id2][0]
        # get image from different seen class
        elif index % 4 == 0:
            is_diff = 1
            is_ood = 0
            image1_class = random.choice(self.seen_classes)
            image2_class = random.choice(self.seen_classes)
            while image1_class == image2_class:
                image2_class = random.choice(self.seen_classes)
            image_id1 = random.choice(self.seen_class_dict[image1_class])
            image_id2 = random.choice(self.seen_class_dict[image2_class])
            image1 = self.cifar10_dataset[image_id1][0]
            image2 = self.cifar10_dataset[image_id2][0]
        # get image 1 from seen classes and image 2 from oe dataset
        elif index % 4 == 2:
            is_diff = 1
            is_ood = 1
            image1_class = random.choice(self.seen_classes)
            image_id1 = random.choice(self.seen_class_dict[image1_class])
            image1 = self.cifar10_dataset[image_id1][0]
            
            image2_class = random.choice(self.oe_classes)
            image_id2 = random.choice(self.oe_class_dict[image2_class])
            image2 = self.oe_dataset[image_id2][0]
            image2_class = 999 #set class as 999 for outlier image
            
        #output
        image1_class = torch.from_numpy(np.array([image1_class], dtype=np.int))
        image2_class = torch.from_numpy(np.array([image2_class], dtype=np.int))
        is_diff = torch.from_numpy(np.array([is_diff], dtype=np.int))
        is_ood = torch.from_numpy(np.array([is_ood], dtype=np.int))

        return image1, image2, image1_class, image2_class, is_diff, is_ood
    
    
    
