import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader

ImageFile.LOAD_TRUNCATED_IMAGES = True


class IndianaDataset(Dataset):
    def __init__(self, dataset_info, dataset_type, img_transform=None,
                 masked_img_transform=None):
        self.img_paths = dataset_info['path']
        self.reports = dataset_info['reports']
        self.report_lens = dataset_info['report_lens']
        self.dataset_type = dataset_type

        # transform for images
        self.img_transform = img_transform  # for original image
        self.masked_img_transform = masked_img_transform  # for masked image

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """Original image processing"""
        img = Image.open(self.img_paths[idx][0]).convert('RGB')
        img = cv2.resize(np.array(img), (256, 256))

        if self.img_transform is not None:
            img = self.img_transform(img)
        img = np.array(img)
        if self.img_transform is None:
            img = np.transpose(img, (2, 0, 1))
        img = torch.FloatTensor(img / 255.)

        """Masked image processing"""
        masked_img = Image.open(self.img_paths[idx][1]).convert('RGB')
        masked_img = cv2.resize(np.array(masked_img), (256, 256))

        if self.masked_img_transform is not None:
            masked_img = self.masked_img_transform(masked_img)

        masked_img = np.array(masked_img)
        if self.masked_img_transform is None:
            masked_img = np.transpose(masked_img, (2, 0, 1))
        masked_img = torch.FloatTensor(masked_img / 255.)

        """reports"""
        report = torch.LongTensor(self.reports[idx][0])
        report_lens = torch.LongTensor([self.report_lens[idx][0]])
        all_reports = torch.LongTensor(self.reports[idx])

        return img, masked_img, report, report_lens, all_reports


def get_splitted_dataset(train_path, cv_path, test_path, image_folder, masked_folder, word2id):
    """
    split the images and captions into train, val and test
    """
    padded_len = 100

    train_data = pd.read_csv(train_path, usecols=['uid', 'filename', 'findings'])
    test_data = pd.read_csv(cv_path, usecols=['uid', 'filename', 'findings'])
    cv_data = pd.read_csv(test_path, usecols=['uid', 'filename', 'findings'])

    train_info = {
        'path': [],
        'reports': [],
        'report_lens': []
    }

    val_info = {
        'path': [],
        'reports': [],
        'report_lens': []
    }

    test_info = {
        'path': [],
        'reports': [],
        'report_lens': []
    }

    reports_list = [train_data['findings'].to_list(), cv_data['findings'].to_list(), test_data['findings'].to_list()]
    img_filenames_list = [train_data['filename'].to_list(), cv_data['filename'].to_list(), test_data['filename'].to_list()]
    info_list = [train_info, val_info, test_info]

    for idx in range(len(reports_list)):
        reports = reports_list[idx]
        image_filenames = img_filenames_list[idx]
        for i in range(len(reports)):
            report = reports[i].split(' ')
            if len(report) > padded_len:
                continue

            # get id sequence
            report_id = [word2id[i] if i in word2id.keys() else word2id['<unk>'] for i in report]
            report_id.insert(0, word2id['<start>'])
            report_id.append(word2id['<end>'])
            report_id += [word2id['<pad>']] * (padded_len - len(report)) # add padding

            # images path for original and masked images
            path = f'{image_folder}/{image_filenames[i]}'
            masked_path = f'{masked_folder}/Masked_{image_filenames[i]}'

            # store information to the dataset
            info_list[idx]['path'].append([path, masked_path])
            info_list[idx]['reports'].append([report_id])
            info_list[idx]['report_lens'].append([len(report) + 2])

    # return train_info, val_info and test_info
    return info_list[0], info_list[1], info_list[2]


def get_DataLoader(train_path, cv_path, test_path, image_folder, masked_folder, word2id, batch_size, workers):
    # get split dataset
    train_info, val_info, test_info = get_splitted_dataset(train_path, cv_path, test_path, image_folder, masked_folder, word2id)

    # load train, val, test dataset information
    train_dataset = IndianaDataset(train_info, 'TRAIN')
    val_dataset = IndianaDataset(val_info, 'VAL')
    test_dataset = IndianaDataset(test_info, 'TEST')

    # dataloader for train and val set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    return train_loader, val_loader, test_loader
