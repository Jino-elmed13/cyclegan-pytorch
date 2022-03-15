from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random

class ImageTransform():
    def __init__(self, size, mean, std):
        self.data_transform = {
            'train': transforms.Compose([
                transforms.Resize(int(size * 1.12), Image.BICUBIC),
                transforms.RandomCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),

            'test': transforms.Compose([
                transforms.Resize(int(size * 1.0), Image.BICUBIC),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
        }

    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)


# データセットクラス
class ImageDataset(Dataset):
    def __init__(self, filelistA, filelistB, transform, phase='train', aligned=False):
        self.filelistA = filelistA
        self.filelistB = filelistB
        self.transform = transform
        self.phase = phase
        self.aligned = aligned

    def __len__(self):
        return max(len(self.filelistA), len(self.filelistB))

    def __getitem__(self, index):

        img_pathA = self.filelistA[index % len(self.filelistA)]

        # ドメイン間のデータ数が揃っていない場合は，ドメインBからはランダムに選ばれる
        if self.aligned:
            img_pathB = self.filelistB[index % len(self.filelistB)]
        else:
            img_pathB = self.filelistB[random.randint(0, len(self.filelistB) - 1)]

        imgA = Image.open(img_pathA).convert('RGB')
        imgB = Image.open(img_pathB).convert('RGB')

        img_tfA = self.transform(imgA, self.phase)
        img_tfB = self.transform(imgB, self.phase)

        return {'A': img_tfA, 'B': img_tfB, 'A_path': img_pathA, 'B_path': img_pathB}
