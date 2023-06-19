'''
文件夹中保存了trainA和trainB
通过util.py中的get_traindata_link解析然后使用CycleDataset加载，详见train.py
ImageFolder
'''
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import PIL.Image as Image
import torch.utils.data as data
import os
import utils
class CycleDataset(data.Dataset):
    def __init__(self, dataset_name, transform):
        self.dataset=dataset_name
        self.data = [self.dataset+'/'+ d for d in os.listdir(self.dataset)]
        self.transform=transform

    def __getitem__(self, index):
        path= self.data[index]
        image0 = Image.open(path)
        # image0 = image0.convert('L')#深度图
        if self.transform:
            if image0.mode != 'RGB':
                image0 = image0.convert('RGB')
            image0 = self.transform(image0)
        return image0
    def __len__(self):
        return len(self.data)
if __name__ == '__main__':
    transform = transforms.Compose(
        [
            # transforms.Scale((224, 224), interpolation=2),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            # transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
    batch_size=1
    dataset_dirs = utils.get_traindata_link("D:/pythoncode/CycleGAN-and-pix2pix/transforA-B/")
    # Pytorch dataloader
    a_dataset = CycleDataset(dataset_name=dataset_dirs['trainA'], transform=transform)
    b_dataset = CycleDataset(dataset_name=dataset_dirs['trainB'], transform=transform)
    a_loader = torch.utils.data.DataLoader(a_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    b_loader = torch.utils.data.DataLoader(b_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
        # a_real = Variable(a_real[0])
        # b_real = Variable(b_real[0])
        print(a_real.shape)
        print(b_real.shape)
        break

