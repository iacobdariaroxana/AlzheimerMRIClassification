from torch.utils.data import Dataset
import os
from glob import glob
import warnings
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
import uuid
import shutil
import splitfolders

warnings.simplefilter('ignore')

splitfolders.ratio('OriginalDataset', ratio=(.8, .1, .1))


class ClassificationDataset(Dataset):
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        class_dict = {'MildDemented': 0, 'ModerateDemented': 1, 'NonDemented': 2, 'VeryMildDemented': 3}
        label = class_dict[os.path.normpath(image_filepath).split(os.sep)[-2]]
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label


class_names = os.listdir('OriginalDataset')

datasets = {
    'train': [],
    'val': [],
    'test': []
}

for phase in ['train', 'val', 'test']:
    l = []
    for i in glob(f'./output/{phase}/**/*'):
        l.append(i)
    datasets[phase] = l

train_transform = A.Compose(
    [A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
     A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
     A.RandomBrightnessContrast(p=0.5),
     A.ColorJitter(),
     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
     ToTensorV2(),
     ]
)

original_transform = A.Compose(
    [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
     ToTensorV2(),
     ]
)

alb_dataset = ClassificationDataset(images_filepaths=datasets['train'], transform=train_transform)
original_dataset = ClassificationDataset(images_filepaths=datasets['train'], transform=original_transform)

dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

try:
    os.mkdir('./prepdata')
    os.mkdir('./prepdata/train')
    os.mkdir('./prepdata/train/MildDemented')
    os.mkdir('./prepdata/train/ModerateDemented')
    os.mkdir('./prepdata/train/NonDemented')
    os.mkdir('./prepdata/train/VeryMildDemented')
except IsADirectoryError as e:
    print(e)


def original_save(original_dataset, limit):
    s = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented', 3: 'VeryMildDemented'}
    original_dataset.transform = A.Compose(
        [t for t in original_dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])

    for idx in range(limit):
        try:
            image, label = original_dataset[idx]

            cv2.imwrite(f'./prepdata/{s[label]}/{str(uuid.uuid4())}.jpg', image)
        except:
            print('Error')


original_save(original_dataset, dataset_sizes['train'])


def alb_save(alb_dataset, limit):
    s = {0: 'MildDemented', 1: 'ModerateDemented', 2: 'NonDemented', 3: 'VeryMildDemented'}

    alb_dataset.transform = A.Compose([t for t in alb_dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    lens = {0: 6000//716, 1: 6000//51, 2: 6000//2560, 3: 6000//1792}
    for idx in range(limit):
        try:
            image, label = alb_dataset[idx]
            for _ in range(lens[label]):
                cv2.imwrite(f'./prepdata/train/{s[label]}/{str(uuid.uuid4())}.jpg', image)
                image, label = alb_dataset[idx]
        except:
            print('Error')


alb_save(alb_dataset, dataset_sizes['train'])
shutil.move('./output/val', './prepdata/')
shutil.move('./output/test', './prepdata/')