import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets
from discriminators import define_Dis
from generators import define_Gen
from dataloader import CycleDataset
import torchvision
import torchvision.transforms as transforms
from argparse import ArgumentParser
from torchvision.models.resnet import resnet50
import utils
n_classes =8
colors = []
def get_args():
    parser = ArgumentParser(description='cycleGAN PyTorch')
    parser.add_argument('--testmodelpath', type=str, default="./train1203/rgb2CTmodel/epoch155.ckpt")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--load_height', type=int, default=240)#286
    parser.add_argument('--load_width', type=int, default=320)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--crop_height', type=int, default=240)#256,为了符合流程，使用480，640训练
    parser.add_argument('--crop_width', type=int, default=320)
    parser.add_argument('--dataset_dir', type=str, default='../datasets/rgbToCtdatas/')#文件夹的存放：rgbToCtdatas下面  trainA，trainB，testA,testB四个文件夹
    parser.add_argument('--checkpoint_dir', type=str, default='../../../Data.CoronaryCT.1/PanYanqi/cycGAN/cycleModel/modelselect')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--no_dropout', action='store_true',default=True, help='no dropout for the generator')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--gen_net', type=str, default='resnet_9blocks')
    parser.add_argument('--dis_net', type=str, default='n_layers')#choose pixel,n_layers
    args = parser.parse_args()
    return args
def getCycFeature():
    '''
    将图片输入到D（b）进行特征分析，判断特征是否融合得比较好
    '''
    args = get_args()
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)

    Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                            use_dropout=not args.no_dropout, gpu_ids=args.gpu_ids)
    Db = define_Dis(input_nc=3, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                            gpu_ids=args.gpu_ids)
    ckpt = utils.load_checkpoint('%s/epoch175.ckpt' % (args.checkpoint_dir))
    Db.load_state_dict(ckpt['Db'])
    Gba.load_state_dict(ckpt['Gba'])
    dataset_dirs = utils.get_traindata_link(args.dataset_dir)

    # Pytorch dataloader
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.Resize((args.load_height, args.load_width)),
         # transforms.RandomCrop((args.crop_height, args.crop_width)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    a_dataset = CycleDataset(dataset_name=dataset_dirs['trainA'], transform=transform)
    b_dataset = CycleDataset(dataset_name=dataset_dirs['trainB'], transform=transform)
    a_loader = torch.utils.data.DataLoader(a_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    b_loader = torch.utils.data.DataLoader(b_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)
    Gba.eval()
    Db.eval()
    ad_len=len(a_loader)
    print("数据长度：",ad_len)
    areal_file="feature/a_real"#_resnet50
    bfake_file="feature/b_fake"
    breal_file="feature/b_real"
    # model = resnet50(pretrained=True).cuda()
    with torch.no_grad():
        for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
            a_real, b_real = utils.cuda([a_real, b_real])
            b_fake = Gba(a_real)
            # b_fake_dis = model(b_fake).cpu().numpy().squeeze()
            # a_real_dis = model(a_real).cpu().numpy().squeeze()
            # b_real_dis = model(b_real).cpu().numpy().squeeze()
            b_fake_dis = Db(b_fake).cpu().numpy().squeeze().reshape(28*38)
            a_real_dis = Db(a_real).cpu().numpy().squeeze().reshape(28*38)
            b_real_dis = Db(b_real).cpu().numpy().squeeze().reshape(28*38)

            np.save(areal_file+"/"+str(i), a_real_dis)
            np.save(bfake_file+"/"+str(i), b_fake_dis)
            np.save(breal_file+"/"+str(i), b_real_dis)
            print("{}|{}".format(i,ad_len))
            # print(b_fake_dis.shape)
            # pic = (torch.cat([a_real, b_fake,b_real],
            #                          dim=0).data + 1) / 2.0
            # torchvision.utils.save_image(pic,  'sample.jpg', nrow=args.batch_size)
            # break

def tsne(embeddings, targets):
    # digits = datasets.load_digits(n_class=10)
    # X, y = digits.data, digits.target
    # n_samples, n_features = X.shape
    tsne = manifold.TSNE(n_components=2,init='pca',)
    X_tsne = tsne.fit_transform(embeddings)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # 归一化
    fig = plt.figure()
    # ax = Axes3D(fig)
    for i in range(1,n_classes):
        if i in targets:
            inds = np.where(targets==i)
            plt.scatter((X_norm[inds,0]), (X_norm[inds,1]), alpha=0.8, color=colors[i-1],s=5)
            # ax.scatter((X_norm[inds,0]), (X_norm[inds,1]),(X_norm[inds,2]), alpha=0.8, color=colors[i-1],s=5)

    # for ii in range(0,360,2):
    #     ax.view_init(elev=ii, azim=10)
    #     plt.savefig("C:/Users/Yu.Tao.VICO/Pictures/Saved Pictures/3Dx%d"%ii+".png")

    plt.show()
    # for i in range(embeddings.shape[0]):
    #     plt.text(X_norm[i, 0], X_norm[i, 1], str(targets[i]), color=plt.cm.Set1(targets[i]),
    #              fontdict={'weight': 'bold', 'size': 7})
    # plt.xticks([])
    # plt.yticks([])

    print('done')

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def plot_embeddings(embeddings, targets, xlim=None, ylim=None):

    plt.figure(figsize=(8,8))
    for i in range(1,n_classes):
        inds = np.where(targets==i)
        plt.scatter((embeddings[inds,0]), (embeddings[inds,1]), alpha=0.8, color=colors[i-1],s=100)
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    # plt.legend(mnist_classes)
    plt.show()


def get_embeddings_targets(path,num):
    k = 0
    with open(path, 'r')as r:
        lines = r.readlines()
        embeddings = np.zeros((len(lines), num))
        targets = np.zeros(len(lines))
        for line in lines:
            result = line.strip().split(' ')
            print(result)
            aaa = [0 for p in range(num)]
            for i in range(num):
                aaa[i] = float(result[i+2])
            embeddings[k] = aaa
            bbb= int(result[1])
            print(bbb)
            targets[k]= bbb
            k+=1
            # break


    np.array(embeddings)
    np.array(targets)
    return  embeddings,targets

    pass
if __name__ == '__main__':
    getCycFeature()
    # 颜色随机
    # for i in range(1, n_classes):
    #     while (True):
    #         color = randomcolor()
    #         if color not in colors:
    #             colors.append(str(color))
    #             break
    #
    # aaa,bbb = get_embeddings_targets('L:/visual-dim4-cross-good1.txt',4)
    # plot_embeddings(aaa,bbb)
    # print(aaa)
    # print(bbb)
    # tsne(aaa,bbb)
    # print('good')