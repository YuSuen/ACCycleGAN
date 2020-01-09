import argparse
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import utils
import network1227 as net
from utils import Colorize
from torchvision.transforms import ToPILImage

parser = argparse.ArgumentParser()
parser.add_argument('--name', required=False, default='1231',  help='project name')
parser.add_argument('--test_data', required=False, default='test_data/',  help='test data path')
parser.add_argument('--in_ngc', type=int, default=3, help='input channel for generator')
parser.add_argument('--out_ngc', type=int, default=3, help='output channel for generator')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--model_path', default='./color_results/', help='model path')
args = parser.parse_args()

print('------------ Options -------------')
for k, v in sorted(vars(args).items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

# results save path
if not os.path.isdir(os.path.join(args.name + '_results', 'test')):
    os.makedirs(os.path.join(args.name + '_results', 'test'))

transform = transforms.Compose([
        transforms.Resize((args.input_size, args.input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
 ])

test_loader_src = utils.data_load(os.path.join('data', args.test_data), 'test', transform, 1, shuffle=True, drop_last=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

A2BG = net.generator(args.in_ngc, args.out_ngc, args.ngf)
A2Bcolor = net.colorization(args.in_ngc, args.out_ngc, args.ngf)

A2BG.load_state_dict(torch.load(args.model_path + 'A2B_Generator.pkl'))
A2Bcolor.load_state_dict(torch.load(args.model_path + 'A2BColorNet.pkl'))

A2BG.to(device)
A2Bcolor.to(device)

color_transform = Colorize()
image_transform = ToPILImage()
with torch.no_grad():
    for n, (x, _) in enumerate(test_loader_src):
        x = x.to(device)
        G_x = A2BG(x)
        G_recon = A2Bcolor(G_x)
        result = torch.cat((x[0], G_recon[0]), 2)
        path = os.path.join(args.name + '_results', 'test',
                        str(n + 1) + args.name + '_test_' + str(n + 1) + '.png')
        plt.imsave(path, (result.cpu().detach().numpy().transpose(1, 2, 0) + 1) / 2)

