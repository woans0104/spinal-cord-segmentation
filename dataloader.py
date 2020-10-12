import os
import glob
import random
import numpy as np
import ipdb
from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as tF
import torch.utils.data as data
from torch.utils.data.dataset import Dataset

import cv2

##

class Spinal_Dataset(Dataset):

    def __init__(self, image_paths,target_paths,input_size,transform,
                 aug_mode=False,aug_range='aug1',dataset='site1'):


        self.image_paths_pre ,self.target_paths_pre  \
            = np.array(image_paths), np.array(target_paths)
        self.dataset = dataset
        self.input_size = input_size
        self.transforms = transform
        self.aug_mode =aug_mode
        self.aug_range = aug_range



    def center_crop(self,img,mask,site):

        coordinate = np.where(mask == 255)


        w = coordinate[0][len(coordinate[0]) //2]
        h = coordinate[1][len(coordinate[1]) // 2]



        # transform pil image
        img = np.expand_dims(img, -1)
        mask = np.expand_dims(mask, -1)

        pil_image = transforms.ToPILImage()
        img = pil_image(img)
        mask = pil_image(mask)



        if site == 'site1':
            img =tF.crop(img,(w)-30,(h)-30,60,60) #site1
            mask = tF.crop(mask, (w) - 30, (h) - 30, 60, 60)  # site1

        elif site == 'site2':
            img = tF.crop(img,(w)-30,(h)-30,60,60) #site2
            mask = tF.crop(mask, (w)-30, (h)-30, 60, 60)  # site2

        elif site == 'site3':
            img =tF.crop(img,(w)-60,(h)-60,120,120) #site3
            mask = tF.crop(mask, (w) - 60, (h) - 60, 120, 120)  # site3
        elif site == 'site4':
            img = tF.crop(img, w - 50, h - 50, 100, 100)  # site4
            mask = tF.crop(mask, w - 50, h - 50, 100, 100)  # site4
        else:
            print('not not not')


        resize = transforms.Resize((self.input_size, self.input_size))


        img = resize(img)
        mask = resize(mask)


        mask = np.array(mask) ####################
        mask[mask > 0] = 1 ################
        mask[mask < 0] = 0


        return img,mask



    def aug(self,image,aug_range):
        """
        # Resize

        resize = transforms.Resize(size=(256, 256))
        image = resize(image)
        mask = resize(mask)

        #from PIL import Image
        #image = Image.fromarray(image)
        #print(type(image))

        # Random horizontal flipping
        horizontal=random.random()
        #print('horizontal flipping', horizontal)
        if horizontal > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        # Random vertical flipping
        vertical=random.random()
        #print('vertical flipping', vertical)
        if vertical > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)
        
        # Random rotation
        Rrotation = random.random()
        #print('Rrotation', Rrotation)
        if Rrotation > 0.5:
            angle = random.randint(-30, 30)
            image = transforms.functional.rotate(image,angle)
            mask = transforms.functional.rotate(mask,angle)


        """
        if aug_range == 'aug7':
            brightness_factor = random.uniform(0.4, 1.4)
            image = tF.adjust_brightness(image, brightness_factor)

            contrast_factor = random.uniform(0.4, 1.4)
            image = tF.adjust_contrast(image, contrast_factor)

        elif aug_range == 'aug1':
            brightness_factor = random.uniform(0.8, 1.2)
            image = tF.adjust_brightness(image, brightness_factor)

            contrast_factor = random.uniform(0.8, 1.2)
            image = tF.adjust_contrast(image, contrast_factor)


        elif aug_range == 'aug2':
            brightness_factor = random.uniform(0.6, 1.4)
            image = tF.adjust_brightness(image, brightness_factor)

            contrast_factor = random.uniform(0.6, 1.4)
            image = tF.adjust_contrast(image, contrast_factor)

        elif aug_range == 'aug9':

            color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)
            image = color_jitter(image)

        elif aug_range == 'aug10':

            color_jitter = transforms.ColorJitter(brightness=0.4,contrast=0.4)
            image = color_jitter(image)

        elif aug_range == 'aug11':

            color_jitter = transforms.ColorJitter(brightness=(0.6,1.4),contrast=(0.6,1.4))
            image = color_jitter(image)
        else:
            print('not augmentation')


        image = np.array(image)


        return image


    def __getitem__(self, index):

        # indexing test

        image_name = self.image_paths_pre[index]
        mask_name = self.target_paths_pre[index]

        image = np.array(Image.open(image_name).convert('L'), 'uint8')
        mask = np.array(Image.open(mask_name).convert('L'), 'uint8')

        # histogram equalization
        #image = cv2.equalizeHist(image)  ##########

        # center_crop
        image_crop, mask_crop = self.center_crop(image, mask, self.dataset)

        # arg
        if self.aug_mode:
            image_crop = self.aug(image_crop,self.aug_range)
            image_tensor = self.transforms(image_crop)
        else:
            image_tensor = self.transforms(image_crop)


        mask_crop = np.array(mask_crop)


        if np.max(mask_crop) > 1:
            mask_crop = mask_crop / 255
            mask_crop[mask_crop > 0.5] = 1
            mask_crop[mask_crop < 0.5] = 0

        mask_crop = mask_crop * 255
        mask_crop = np.expand_dims(mask_crop, -1)
        mask_crop = np.array(mask_crop, dtype=np.uint8)

        assert len(set(mask_crop.flatten())) == 2, 'mask label is wrong'

        toTensor = transforms.ToTensor()
        mask_tensor = toTensor(mask_crop)

        return image_tensor, mask_tensor, np.array(image_crop), image_name



    def __len__(self):  # return count of sample we have

        return len(self.image_paths_pre)


def dataset_condition(trainset_condition):
    dataset = {
        'site2': ['site1','site3','site4'],
        'site3': ['site1','site2','site4'],
        'site4': ['site1','site2','site3']
    }

    if trainset_condition in dataset.keys():
        print('*' * 50)
        print('train dataset : ', trainset_condition)
        print('test dataset1 : ', dataset[trainset_condition][0])
        print('test dataset2 : ', dataset[trainset_condition][1])
        print('test dataset3 : ', dataset[trainset_condition][2])
        print('*' * 50)

        train_datset = trainset_condition
        test_dataset1 = dataset[trainset_condition][0]
        test_dataset2 = dataset[trainset_condition][1]
        test_dataset3 = dataset[trainset_condition][2]

        return train_datset, test_dataset1, test_dataset2, test_dataset3

    else:
        import ipdb;
        ipdb.set_trace()



def get_loader(server, dataset, train_size, batch_size, input_size,
               aug_mode, aug_range, work_dir=None):

    # transform
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5], [0.5])])

    print('train_size ::',train_size)
    if train_size != 1:
        trn_image_path, trn_label_path, tst_image_path, tst_label_path \
            = load_data_path(server, dataset,train_size=train_size)


        np.save(os.path.join(work_dir, '{}_test_path.npy'.format(dataset)),
                [tst_image_path, tst_label_path])  ########



        trn_dataset = Spinal_Dataset(trn_image_path,
                                     trn_label_path,
                                     input_size,
                                     transform,
                                     aug_mode=aug_mode,
                                     aug_range=aug_range,
                                     dataset=dataset)


        trn_loader = data.DataLoader(trn_dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       drop_last = True, ####################
                                       num_workers=4)

        tst_dataset = Spinal_Dataset(tst_image_path,
                                    tst_label_path,
                                    input_size,
                                    transform,
                                    dataset=dataset)

        tst_loader = data.DataLoader(tst_dataset,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=0)

        return trn_loader, tst_loader

    else: # train_size == 1

        trn_image_path, trn_label_path = load_data_path(server,
                                                        dataset,
                                                        train_size=train_size)



        trn_dataset = Spinal_Dataset(trn_image_path,
                                   trn_label_path,
                                   input_size,
                                   transform,
                                   aug_mode=aug_mode,
                                   aug_range=aug_range,
                                   dataset=dataset)



        trn_loader = data.DataLoader(trn_dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=4)


        return trn_loader , trn_loader



def load_data_path(server, dataset, train_size):

    def read_data(data_folder):

        valid_exts = ['.jpg', '.gif', '.png', '.tga', '.jpeg']

        data_paths = []
        for f in data_folder:
            ext = os.path.splitext(f)[1]

            if ext.lower() not in valid_exts:
                continue
            data_paths.append(f)

        return data_paths

    def match_data_path(img_path, target_path):

        img_path = np.array(sorted(img_path))
        target_path = np.array(sorted(target_path))

        # list name sort
        imgName_li = []
        for i in range(len(img_path)):
            img_name = img_path[i].split('/')[-1].split('.')[0]
            imgName_li.append(img_name)

        total_img_li = []
        for i in range(len(target_path)):
            img_name \
                = target_path[i].split('/')[-1].split('.')[0].split('_mask')[0]

            idx = np.where(imgName_li == np.array(img_name))[0][0]

            total_img_li.append(img_path[idx])

        return total_img_li, target_path



    ##########################################################################
    if server == 'server_A':
        image_folder = sorted(glob.glob(('/data2/woans0104/Spinal_cord_dataset/'
                                         'voting2/{}/image/*'.format(dataset))))


        target_folder = sorted(glob.glob(('/data2/woans0104/Spinal_cord_dataset/'
                                          'voting2/{}/label/*'.format(dataset))))

    elif server == 'server_B':
        image_folder = sorted(glob.glob(('/data2/spinal_cord_dataset/voting2/'
                                         '{}/image/*'.format(dataset))))


        target_folder = sorted(glob.glob(('/data2/spinal_cord_dataset/voting2/'
                                          '{}/label/*'.format(dataset))))

    elif server == 'server_D':
        image_folder = sorted(glob.glob("/daintlab/data/"
                                        "spinal_cord_dataset/voting2/{}/image/*"
                                        .format(dataset)))
        target_folder = sorted(glob.glob("/daintlab/data/"
                                         "spinal_cord_dataset/voting2/{}/label/*"
                                         .format(dataset)))

    ##########################################################################


    image_paths = read_data(image_folder)
    target_paths = read_data(target_folder)

    if len(image_paths) != len(target_paths):
        image_paths, target_paths = match_data_path(image_paths, target_paths)

    # 'different length img & mask'
    assert len(image_paths) == len(target_paths), print(target_paths)

    # last sort
    image_paths = sorted(image_paths)
    target_paths = sorted(target_paths)

    len_data = len(image_paths)
    indices_image = list(range(len_data))

    np.random.shuffle(indices_image)

    image_paths = np.array(image_paths)
    target_paths = np.array(target_paths)

    train_image_no = indices_image[:int(len_data * train_size)]
    test_image_no = indices_image[int(len_data * train_size):]

    train_image_paths = image_paths[train_image_no]
    train_mask_paths = target_paths[train_image_no]

    test_image_paths = image_paths[test_image_no]
    test_mask_paths = target_paths[test_image_no]


    if train_size == 1:

        return train_image_paths, train_mask_paths
    else:
        return train_image_paths, train_mask_paths, \
               test_image_paths, test_mask_paths

