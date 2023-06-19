import os
from argparse import ArgumentParser
import train as md
from utils import create_link
import test as tst
from PIL import ImageFile

'''
代码参考：https://github.com/arnab39/cycleGAN-PyTorch
'''


# To get arguments from commandline
def get_args():
    parser = ArgumentParser(description='cycleGAN PyTorch')
    parser.add_argument('--testmodelpath', type=str,
                        default=r'D:\Project_Code\CycleGAN\model\cho.ckpt')  # 100.ckpt,/modelselect
    parser.add_argument('--loadepoch', type=int, default=175)
    parser.add_argument('--testImagepath', type=str, default="D:/Project_Code/CycleGAN/testmodel/249")
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--decay_epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=.0002)
    parser.add_argument('--load_height', type=int, default=480)  # 286
    parser.add_argument('--load_width', type=int, default=480)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--crop_height', type=int, default=480)  # 256,为了符合流程，使用480，640训练
    parser.add_argument('--crop_width', type=int, default=480)
    parser.add_argument('--lamda', type=int, default=10)
    parser.add_argument('--idt_coef', type=float, default=0.5)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--results_dir', type=str, default='D:/Project_Code/CycleGAN/output')
    parser.add_argument('--dataset_dir', type=str,
                        default='D:/DATA/SimCol_Challenge_2022/gantrain_cholec/')  # 文件夹的存放：rgbToCtdatas下面  trainA，trainB，testA,testB四个文件夹'../datasets/rgbToCtdatas/'
    parser.add_argument('--checkpoint_dir', type=str, default='D:/Project_Code/CycleGAN/model')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--no_dropout', action='store_true', default=True, help='no dropout for the generator')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--gen_net', type=str, default='resnet_9blocks')
    parser.add_argument('--dis_net', type=str, default='n_layers')  # choose pixel,n_layers
    args = parser.parse_args()
    return args


def main():
    ImageFile.LOAD_TRUNCATED_IMAGES = True  # 跳过损坏图像
    args = get_args()
    # args.training = True
    args.testing=True

    # create_link(args.dataset_dir)

    # gpu设置
    str_ids = args.gpu_ids.split(',')
    args.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            args.gpu_ids.append(id)

    # print("args.",args.no_dropout)
    # print(not args.no_dropout)
    if args.training:
        print("Training")
        model = md.cycleGAN(args)
        model.train(args)
    if args.testing:
        print("Testing")
        tst.test(args)


if __name__ == '__main__':
    main()
