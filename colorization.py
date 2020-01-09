import argparse
import os
import time
import matplotlib.pyplot as plt
import pickle
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch
import itertools
import torch.nn as nn
from torch.autograd import Variable
from utils import ReplayBuffer, Gray, print_network, data_load, weights_init_normal
from utils import Logger ,VOC12, ToLabel, Relabel, CrossEntropyLoss2d
import network as net

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='color',  help='project name')
parser.add_argument('--src_data', required=False, default='data/src_data/',  help='src data path')
parser.add_argument('--tgt_data', required=False, default='data/tgt_data/',  help='tgt data path')
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--in_ndc', type=int, default=3, help='input channel for discriminator')
parser.add_argument('--out_ndc', type=int, default=1, help='output channel for discriminator')
parser.add_argument('--batch_size', type=int, default=5, help='batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=32)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--train_epoch', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--resume', default=False, help='resume net path')
args = parser.parse_args()

print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

# results and log save path
if not os.path.isdir(os.path.join(args.name + '_results', 'Colorization')):
    os.makedirs(os.path.join(args.name + '_results', 'Colorization'))
if not os.path.exists(os.path.join(args.name + '_results', 'log')):
    os.makedirs(os.path.join(args.name + '_results', 'log'))

# Loss plot
logger = Logger(os.path.join(args.name + '_results', 'log'))

image_transform = transforms.Compose([
    transforms.CenterCrop(args.input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
 ])

target_transform = transforms.Compose([
    transforms.CenterCrop(args.input_size),
    ToLabel(),
    Relabel(255, 21),

 ])

train_loader_src = DataLoader(VOC12(args.src_data, image_transform, target_transform),num_workers=4, batch_size=args.batch_size, shuffle=True)
train_loader_tgt = DataLoader(VOC12(args.tgt_data, image_transform, target_transform),num_workers=4, batch_size=args.batch_size, shuffle=True)
test_loader_src = data_load('./data/test_data/','test', image_transform, 1, shuffle=True, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

A2BG = net.generator(args.in_ngc, args.out_ngc, args.ngf)
B2AG = net.generator(args.in_ngc, args.out_ngc, args.ngf)
A2Bcolor = net.colorization(args.in_ngc, args.out_ngc, args.ngf)
B2Acolor = net.colorization(args.in_ngc, args.out_ngc, args.ngf)
AD = net.discriminator(args.in_ndc, args.out_ndc, args.ndf)
BD = net.discriminator(args.in_ndc, args.out_ndc, args.ndf)
Segm = net.segmantation(args.in_ngc, args.out_ngc, args.ngf)

print('---------- Networks initialized -------------')
print_network(A2BG)
print_network(A2Bcolor)
print_network(AD)
print_network(Segm)
print('-----------------------------------------------')

vgg16 = models.vgg16(pretrained=True)
vgg16 = net.VGG(vgg16.features[:23]).to(device)

A2BG.to(device)
B2AG.to(device)
A2Bcolor.to(device)
B2Acolor.to(device)
Segm.to(device)
AD.to(device)
BD.to(device)

if not args.resume:
    print('Initializing weights...')
    # initialize
    A2Bcolor.apply(weights_init_normal)
    B2Acolor.apply(weights_init_normal)
    Segm.apply(weights_init_normal)


else:
# load resume network
    print('Loading resume network...')
    A2BG.load_state_dict(torch.load('./color_results/332/A2B_Generator.pkl'))
    B2AG.load_state_dict(torch.load('./color_results/332/B2A_Generator.pkl'))
    A2Bcolor.load_state_dict(torch.load('./color_results/332/A2BColorNet.pkl'))
    B2Acolor.load_state_dict(torch.load('./color_results/332/B2AColorNet.pkl'))
    Segm.load_state_dict(torch.load('./color_results/332/SegmNet.pkl'))
    AD.load_state_dict(torch.load('./color_results/332/A_Discriminator.pkl'))
    BD.load_state_dict(torch.load('./color_results/332/B_Discriminator.pkl'))


A2BG.train()
B2AG.train()
AD.train()
BD.train()
A2Bcolor.train()
B2Acolor.train()
Segm.train()

# loss
criterion_GAN = nn.MSELoss().to(device)
criterion_cycle = nn.L1Loss().to(device)
criterion_identity = nn.L1Loss().to(device)
criterion_color = nn.MSELoss().to(device)
criterion_segm = CrossEntropyLoss2d().to(device)

# Optimizers
A2BG_params = [
    {"params": filter(lambda p: p.requires_grad, A2BG.parameters())},
    {"params": filter(lambda p: p.requires_grad, A2Bcolor.parameters())},
    {"params": filter(lambda p: p.requires_grad, Segm.parameters())}]

B2AG_params = [
    {"params": filter(lambda p: p.requires_grad, B2AG.parameters())},
    {"params": filter(lambda p: p.requires_grad, B2Acolor.parameters())}]

optimizer_G = torch.optim.Adam(itertools.chain(A2BG_params, B2AG_params), lr=args.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(filter(lambda p: p.requires_grad, AD.parameters()), lr=args.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(filter(lambda p: p.requires_grad, BD.parameters()), lr=args.lr, betas=(0.5, 0.999))

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
target_real = Variable(Tensor(args.batch_size, 1).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(args.batch_size, 1).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

torch.backends.cudnn.benchmark = True
train_hist = {}
train_hist['G_loss'] = []
train_hist['G_identity_loss'] = []
train_hist['G_GAN_loss'] = []
train_hist['G_cycle_loss'] = []
train_hist['G_Color_loss'] = []
train_hist['G_seg_loss'] = []
train_hist['D_loss'] = []
train_hist['per_epoch_time'] = []
train_hist['total_time'] = []

################ Training ################

print('Epoch_size:', len(train_loader_src), 'Training start!')
start_time = time.time()
for epoch in range(args.start_epoch, args.train_epoch):
    epoch_start_time = time.time()
    G_losses = []
    G_identity_losses = []
    G_GAN_losses = []
    G_cycle_losses = []
    G_Color_losses = []
    G_Seg_losses = []
    D_losses = []

    for x, y in zip(train_loader_src, train_loader_tgt):

        real_A, label_A = x
        real_B, label_B = y


        real_A = real_A.to(device)
        label_A = label_A.to(device)

        real_B = real_B.to(device)
        label_B = label_B.to(device)

        # train G

        ###### Generators A2B and B2A ######

        optimizer_G.zero_grad()

        #################################### Identity loss

        # G_A2B(B) should equal B if real B is fed
        same_B = A2BG(real_B)
        same_B = A2Bcolor(same_B)
        real_B_feature = vgg16(real_B)
        same_B_feature = vgg16(same_B)
        loss_identity_B = criterion_identity(same_B_feature[2], real_B_feature[2]) * 5.0

        # G_B2A(A) should equal A if real A is fed
        same_A = B2AG(real_A)
        same_A = B2Acolor(same_A)
        real_A_feature = vgg16(real_A)
        same_A_feature = vgg16(same_A)
        loss_identity_A = criterion_identity(same_A_feature[2], real_A_feature[2]) * 5.0


        #################################### GAN loss

        # A2B
        fake_B_G = A2BG(real_A)
        fake_B = A2Bcolor(fake_B_G)
        B_pred_fake = BD(fake_B)
        loss_GAN_A2B = criterion_GAN(B_pred_fake, target_real)


        fake_A_G = B2AG(real_B)
        fake_A = B2Acolor(fake_A_G)
        A_pred_fake = AD(fake_A)
        loss_GAN_B2A = criterion_GAN(A_pred_fake, target_real)

        #################################### Color loss

        # A2B
        RA_Gray = torch.mean(real_A, dim=1, keepdim=True)
        FB_Gray = Gray(fake_B, args.input_size, args.batch_size)
        color_loss_A2B = criterion_color(FB_Gray, RA_Gray) * 10.0

        # B2A
        FA_Gray = torch.mean(fake_A, dim=1, keepdim=True)
        RB_Gray = Gray(real_B, args.input_size, args.batch_size)
        color_loss_B2A = criterion_color(FA_Gray, RB_Gray) * 10.0

        #################################### Cycle loss

        recovered_A = B2AG(fake_B)
        recovered_A = B2Acolor(recovered_A)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

        recovered_B = A2BG(fake_A)
        recovered_B = A2Bcolor(recovered_B)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

        #################################### Seg loss

        # GRAY -> COLOR
        segm_A = Segm(fake_B_G)

        loss_segm_A = criterion_segm(segm_A, label_A[:, 0])

        loss_G_seg = loss_segm_A

        G_Seg_losses.append(loss_G_seg.item())
        train_hist['G_seg_loss'].append(loss_G_seg)

        loss_G_identity = loss_identity_A + loss_identity_B
        G_identity_losses.append(loss_G_identity.item())
        train_hist['G_identity_loss'].append(loss_G_identity.item())

        loss_G_GAN = loss_GAN_A2B + loss_GAN_B2A
        G_GAN_losses.append(loss_G_GAN.item())
        train_hist['G_GAN_loss'].append(loss_G_GAN.item())

        loss_Color = color_loss_B2A + color_loss_A2B
        G_Color_losses.append(loss_Color.item())
        train_hist['G_Color_loss'].append(loss_Color.item())

        loss_G_cycle = loss_cycle_ABA + loss_cycle_BAB
        G_cycle_losses.append(loss_G_cycle.item())
        train_hist['G_cycle_loss'].append(loss_G_cycle.item())

        #################################### Total loss

        loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_identity_A + loss_identity_B\
                 + color_loss_B2A + color_loss_A2B  + loss_G_seg

        loss_G.backward()

        G_losses.append(loss_G.item())
        train_hist['G_loss'].append(loss_G.item())
        optimizer_G.step()

        ####################################

        # train D

        ######### Discriminator A ##########

        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = AD(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = AD(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5

        loss_D_A.backward(retain_graph=True)

        optimizer_D_A.step()

        ####################################

        ######### Discriminator B ##########

        optimizer_D_B.zero_grad()

        # Real loss
        pred_real = BD(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake = BD(fake_B.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5

        loss_D_B.backward(retain_graph=True)

        optimizer_D_B.step()

        loss_D = loss_D_A + loss_D_B

        D_losses.append(loss_D.item())
        train_hist['D_loss'].append(loss_D.item())

        ####################################

    per_epoch_time = time.time() - epoch_start_time
    train_hist['per_epoch_time'].append(per_epoch_time)

    # log
    logger.scalar_summary('loss_G', torch.mean(torch.FloatTensor(G_losses)), epoch + 1)
    logger.scalar_summary('loss_G_identity', torch.mean(torch.FloatTensor(G_identity_losses)), epoch + 1)
    logger.scalar_summary('loss_G_GAN', torch.mean(torch.FloatTensor(G_GAN_losses)), epoch + 1)
    logger.scalar_summary('loss_G_cycle', torch.mean(torch.FloatTensor(G_cycle_losses)), epoch + 1)
    logger.scalar_summary('loss_G_color', torch.mean(torch.FloatTensor(G_Color_losses)), epoch + 1)
    logger.scalar_summary('loss_G_seg', torch.mean(torch.FloatTensor(G_Seg_losses)), epoch + 1)
    logger.scalar_summary('loss_D', torch.mean(torch.FloatTensor(D_losses)), epoch + 1)

    print(
        '[%d/%d] - time: %.2f, G_loss: %.3f, G_identity_loss: %.3f, G_GAN_loss: %.3f, G_cycle_loss: %.3f, D_loss: %.3f' % (
        (epoch + 1), args.train_epoch, per_epoch_time, torch.mean(torch.FloatTensor(G_losses)), torch.mean(torch.FloatTensor(G_identity_losses)),
        torch.mean(torch.FloatTensor(G_GAN_losses)), torch.mean(torch.FloatTensor(G_cycle_losses)), torch.mean(torch.FloatTensor(D_losses))))

    with torch.no_grad():
        A2BG.eval()
        for n, (image, _) in enumerate(test_loader_src):
            image = image.to(device)
            G_recon = A2BG(image)
            G_recon = A2Bcolor(G_recon)
            result = torch.cat((image[0], G_recon[0]), 2)
            path = os.path.join(args.name + '_results', 'Colorization', str(epoch+1) + '_epoch_' + args.name + '_test_' + str(n + 1) + '.png')
            plt.imsave(path, (result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)

    # Save models checkpoints
    if not os.path.isdir(os.path.join(args.name + '_results', str(epoch+1))):
        os.makedirs(os.path.join(args.name + '_results', str(epoch+1)))
        models_path = os.path.join(args.name + '_results', str(epoch+1))
        torch.save(A2BG.state_dict(), os.path.join(models_path, 'A2B_Generator.pkl'))
        torch.save(AD.state_dict(), os.path.join(models_path, 'A_Discriminator.pkl'))
        torch.save(B2AG.state_dict(), os.path.join(models_path, 'B2A_Generator.pkl'))
        torch.save(BD.state_dict(), os.path.join(models_path, 'B_Discriminator.pkl'))
        torch.save(Segm.state_dict(), os.path.join(models_path, 'SegmNet.pkl'))
        torch.save(A2Bcolor.state_dict(), os.path.join(models_path, 'A2BColorNet.pkl'))
        torch.save(B2Acolor.state_dict(), os.path.join(models_path, 'B2AColorNet.pkl'))

total_time = time.time() - start_time
train_hist['total_time'].append(total_time)

print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (torch.mean(torch.FloatTensor(train_hist['per_epoch_time'])), args.train_epoch, total_time))
print("Training finish!... save training results")

torch.save(A2BG.state_dict(), os.path.join(args.name + '_results',  'A2B_Generator_param.pkl'))
torch.save(AD.state_dict(), os.path.join(args.name + '_results',  'A_Discriminator_param.pkl'))
torch.save(B2AG.state_dict(), os.path.join(args.name + '_results',  'B2A_Generator_param.pkl'))
torch.save(BD.state_dict(), os.path.join(args.name + '_results',  'B_Discriminator_param.pkl'))
torch.save(Segm.state_dict(), os.path.join(args.name + '_results', 'SegmNet.pkl'))
torch.save(A2Bcolor.state_dict(), os.path.join(args.name + '_results', 'A2BColorNet.pkl'))
torch.save(B2Acolor.state_dict(), os.path.join(args.name + '_results', 'B2AColorNet.pkl'))
with open(os.path.join(args.name + '_results',  'train_hist.pkl'), 'wb') as f:
    pickle.dump(train_hist, f)