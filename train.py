import itertools

import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from generators import define_Gen
from discriminators import define_Dis
from torch.optim import lr_scheduler
from gdcommon import set_grad
from dataloader import CycleDataset
from tensorboardX import SummaryWriter
from loss import ssim

'''
Class for CycleGAN with train() as a member function
'''
# device='cpu'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class cycleGAN(object):
    def __init__(self, args):
        # Define the network
        #####################################################
        self.Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                              use_dropout=not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                              use_dropout=not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Da = define_Dis(input_nc=3, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=3, ndf=args.ndf, netD=args.dis_net, n_layers_D=3, norm=args.norm,
                             gpu_ids=args.gpu_ids)
        # print("判别器：",self.Da)
        # print("生成器：",self.Gab)
        utils.print_networks([self.Gab, self.Gba, self.Da, self.Db], ['Gab', 'Gba', 'Da', 'Db'])

        # Define Loss criterias
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        # Optimizers
        #####################################################
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(), self.Gba.parameters()), lr=args.lr,
                                            betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(), self.Db.parameters()), lr=args.lr,
                                            betas=(0.5, 0.999))

        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer,
                                                                lr_lambda=utils.LambdaLR(args.epochs, 0,
                                                                                         args.decay_epoch).step)

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        # 模型重新训练
        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0

    def train(self, args):
        # For transforming the input image
        a_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize((args.load_height, args.load_width)),
             # transforms.RandomCrop((args.crop_height, args.crop_width)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        b_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize((args.load_height, args.load_width)),
             # transforms.RandomCrop((args.crop_height, args.crop_width)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        dataset_dirs = utils.get_traindata_link(args.dataset_dir)

        # Pytorch dataloader
        a_dataset = CycleDataset(dataset_name=dataset_dirs['trainA'], transform=a_transform)
        b_dataset = CycleDataset(dataset_name=dataset_dirs['trainB'], transform=b_transform)
        a_loader = torch.utils.data.DataLoader(a_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                               drop_last=True)
        b_loader = torch.utils.data.DataLoader(b_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                               drop_last=True)
        # dsets.ImageFolder(dataset_dirs['trainB'], transform=transform)

        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()
        print("one epoch data iter num:", len(a_loader))
        writer = SummaryWriter()
        iter = 0
        for epoch in range(self.start_epoch, args.epochs):

            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)
            saveiter = 0

            for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
                # step
                step = epoch * min(len(a_loader), len(b_loader)) + i + 1
                # Generator Computations
                ##################################################

                set_grad([self.Da, self.Db], False)
                self.g_optimizer.zero_grad()

                # a_real = Variable(a_real[0])
                # b_real = Variable(b_real[0])
                a_real, b_real = utils.cuda([a_real, b_real])
                #  print("a shape:",a_real.shape)
                #  print("b shape:", b_real.shape)
                # Forward pass through generators
                ##################################################
                a_fake = self.Gab(b_real)
                b_fake = self.Gba(a_real)

                a_recon = self.Gab(b_fake)
                b_recon = self.Gba(a_fake)

                a_idt = self.Gab(a_real)
                b_idt = self.Gba(b_real)

                # ssim losses
                ###################################################
                a_ssim_loss = torch.clamp((1 - ssim(a_fake, b_real)) * 0.5, 0, 1) * args.lamda * args.idt_coef
                b_ssim_loss = torch.clamp((1 - ssim(b_fake, a_real)) * 0.5, 0, 1) * args.lamda * args.idt_coef

                # Identity losses
                ###################################################
                a_idt_loss = self.L1(a_idt, a_real) * args.lamda * args.idt_coef
                b_idt_loss = self.L1(b_idt, b_real) * args.lamda * args.idt_coef
                # a_idt_loss = self.L1(a_fake, b_real) * args.lamda * args.idt_coef
                # b_idt_loss = self.L1(b_fake, a_real) * args.lamda * args.idt_coef

                # Adversarial losses
                ###################################################
                a_fake_dis = self.Da(a_fake)
                b_fake_dis = self.Db(b_fake)

                real_label = utils.cuda(Variable(torch.ones(a_fake_dis.size())))

                a_gen_loss = self.MSE(a_fake_dis, real_label)
                b_gen_loss = self.MSE(b_fake_dis, real_label)

                # Cycle consistency losses
                ###################################################
                a_cycle_loss = self.L1(a_recon, a_real) * args.lamda
                b_cycle_loss = self.L1(b_recon, b_real) * args.lamda

                # Total generators losses
                ###################################################
                gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                # Update generators
                ###################################################
                gen_loss.backward()
                self.g_optimizer.step()

                # Discriminator Computations
                #################################################

                set_grad([self.Da, self.Db], True)
                self.d_optimizer.zero_grad()

                # Sample from history of generated images
                #################################################
                a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                a_fake, b_fake = utils.cuda([a_fake, b_fake])

                # Forward pass through discriminators
                #################################################
                a_real_dis = self.Da(a_real)
                a_fake_dis = self.Da(a_fake)
                b_real_dis = self.Db(b_real)
                b_fake_dis = self.Db(b_fake)
                real_label = utils.cuda(Variable(torch.ones(a_real_dis.size())))
                fake_label = utils.cuda(Variable(torch.zeros(a_fake_dis.size())))

                # real_label=Variable(torch.ones(a_real_dis.size())).to(device)
                # fake_label=Variable(torch.zeros(a_fake_dis.size())).to(device)
                # Discriminator losses
                ##################################################
                a_dis_real_loss = self.MSE(a_real_dis, real_label)
                a_dis_fake_loss = self.MSE(a_fake_dis, fake_label)
                b_dis_real_loss = self.MSE(b_real_dis, real_label)
                b_dis_fake_loss = self.MSE(b_fake_dis, fake_label)

                # Total discriminators losses
                a_dis_loss = (a_dis_real_loss + a_dis_fake_loss) * 0.5
                b_dis_loss = (b_dis_real_loss + b_dis_fake_loss) * 0.5

                # Update discriminators
                ##################################################
                a_dis_loss.backward()
                b_dis_loss.backward()
                self.d_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" %
                      (epoch, i + 1, min(len(a_loader), len(b_loader)),
                       gen_loss, a_dis_loss + b_dis_loss))

                writer.add_scalars('cycleGAN/loss', {'gen_loss': float(gen_loss),
                                                     'disloss': float(a_dis_loss + b_dis_loss)}, iter)
                iter += 1

                # save image
                # print("saveiter<5 and iter%10:",(saveiter<5 and iter%10==0),iter,saveiter)
                if saveiter < 3 and iter % 10 == 0:
                    saveiter += 1
                    pic = (torch.cat([a_real, b_fake, a_recon, b_real, a_fake, b_recon],
                                     dim=0).data + 1) / 2.0
                    torchvision.utils.save_image(pic,
                                                 args.results_dir + '/sample_epoch{}_iter{}.jpg'.format(epoch, iter),
                                                 nrow=args.batch_size)
                    print("save image:/{}/sample_epoch{}_iter{}.jpg".format(args.results_dir, epoch, iter))
            # Override the latest checkpoint
            #######################################################
            # 将所有的保存为一个文件
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                  '%s/latest.ckpt' % (args.checkpoint_dir))
            if epoch % 5 == 0 or epoch % 5 != 0:
                utils.save_checkpoint({'epoch': epoch + 1,
                                       'Da': self.Da.state_dict(),
                                       'Db': self.Db.state_dict(),
                                       'Gab': self.Gab.state_dict(),
                                       'Gba': self.Gba.state_dict(),
                                       'd_optimizer': self.d_optimizer.state_dict(),
                                       'g_optimizer': self.g_optimizer.state_dict()},
                                      '{}/epoch{}.ckpt'.format(args.checkpoint_dir, epoch))
                # torch.save(self.Gab.state_dict(), '{}/GAB_dict_epoch_{}.pt'.format(args.checkpoint_dir, epoch))
                # torch.save(self.Gba.state_dict(), '{}/GBA_dict_epoch_{}.pt'.format(args.checkpoint_dir, epoch))
            # Update learning rates
            ########################
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()
        writer.close()
