import os
import torch
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from dataloader import CycleDataset
from generators import define_Gen


def test(args):
    transform = transforms.Compose(
        [transforms.Resize((args.crop_height, args.crop_width)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    dataset_dirs = utils.get_testdata_link(args.dataset_dir)

    # a_test_data = dsets.ImageFolder(dataset_dirs['testA'], transform=transform)
    # b_test_data = dsets.ImageFolder(dataset_dirs['testB'], transform=transform)
    a_test_data = CycleDataset(dataset_name=dataset_dirs['testA'], transform=transform)
    b_test_data = CycleDataset(dataset_name=dataset_dirs['testB'], transform=transform)
    a_test_loader = torch.utils.data.DataLoader(a_test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    b_test_loader = torch.utils.data.DataLoader(b_test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG='resnet_9blocks', norm=args.norm,
                     use_dropout=not args.no_dropout, gpu_ids=args.gpu_ids)
    Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG='resnet_9blocks', norm=args.norm,
                     use_dropout=not args.no_dropout, gpu_ids=args.gpu_ids)

    utils.print_networks([Gab, Gba], ['Gab', 'Gba'])

    # loadmodel
    # Gabpth = "D:/pythoncode/mymode/latest_net_G_A_dict.pth"
    # Gbapth = "D:/pythoncode/mymode/latest_net_G_B_dict.pth"
    # Gab.load_state_dict(torch.load(Gabpth))
    # Gba.load_state_dict(torch.load(Gbapth))
    try:
        ckpt = utils.load_checkpoint('%s' % (args.testmodelpath))  # /latest.ckpt
        Gab.load_state_dict(ckpt['Gab'])
        Gba.load_state_dict(ckpt['Gba'])
    except:
        print(' [*] No checkpoint!')
    """ run """
    # a_real_test = Variable(iter(a_test_loader).next(), requires_grad=True)
    # b_real_test = Variable(iter(b_test_loader).next(), requires_grad=True)
    # a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])

    Gab.eval()
    Gba.eval()
    print("test len:", len(a_test_loader))
    if not os.path.isdir(args.testImagepath):
        os.makedirs(args.testImagepath)
    with torch.no_grad():
        for i, (a_real, b_real) in enumerate(zip(a_test_loader, b_test_loader)):
            a_real_test, b_real_test = utils.cuda([a_real, b_real])
            a_fake_test = Gab(b_real_test)
            b_fake_test = Gba(a_real_test)
            a_recon_test = Gab(b_fake_test)
            b_recon_test = Gba(a_fake_test)
            # 只保存初始的一张图片
            pic = (torch.cat([a_real_test, b_fake_test, a_recon_test, b_real_test, a_fake_test, b_recon_test],
                             dim=0).data + 1) / 2.0
            torchvision.utils.save_image(pic, args.testImagepath + '/sample{}_{}.jpg'.format(i,args.loadepoch),
            nrow=args.batch_size)
            # 用于输出单张图片，需要修改batch为1
            # pic = (a_real_test.data + 1) / 2.0
            # torchvision.utils.save_image(pic, os.path.join(args.testImagepath, 'a') + '/sample{}_{}.jpg'.format(i, args.loadepoch),
            #                              nrow=args.batch_size)
            # pic = (b_fake_test.data + 1) / 2.0
            # torchvision.utils.save_image(pic, os.path.join(args.testImagepath, 'a2b') + '/sample{}_{}.jpg'.format(i, args.loadepoch),
            #                              nrow=args.batch_size)
