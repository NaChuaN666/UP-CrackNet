import torch
from torchvision import transforms
from torch.autograd import Variable
from dataset import DatasetFromFolder
from model import Generator, Discriminator
import utils
import argparse
import os
from logger import Logger
from math import exp
import torch.nn.functional as F
from msgms_loss import MSGMS_Loss
import time
from torch.utils.tensorboard import SummaryWriter 

from PS_loss import StyleLoss, PerceptualLoss

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=False, default='all_crack500_train', help='input dataset')
parser.add_argument('--direction', required=False, default='AtoB', help='input and target image order')
parser.add_argument('--batch_size', type=int, default=8, help='train batch size')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--input_size', type=int, default=256, help='input size')
parser.add_argument('--resize_scale', type=int, default=286, help='resize scale (0 is false)')
parser.add_argument('--crop_size', type=int, default=256, help='crop size (0 is false)')
parser.add_argument('--fliplr', type=bool, default=True, help='random fliplr True of False')
parser.add_argument('--num_epochs', type=int, default=300, help='number of train epochs')
parser.add_argument('--lrG', type=float, default=0.0008, help='learning rate for generator, default=0.0002')
parser.add_argument('--lrD', type=float, default=0.0002, help='learning rate for discriminator, default=0.0002')
parser.add_argument('--lamb', type=float, default=100, help='lambda for L1 loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
params = parser.parse_args()
print(params)

writer = SummaryWriter('./path/log1')

# SSIM:
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


data_dir = './dataset/crack500_train/'
model_dir = './saved-model/'

if not os.path.exists(model_dir):
    os.mkdir(model_dir)

transform = transforms.Compose([transforms.Resize(params.input_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])


train_data = DatasetFromFolder(data_dir, subfolder='train', direction=params.direction, 
                resize_scale=params.resize_scale,  transform=transform, crop_size=params.crop_size, fliplr=params.fliplr)


train_data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=params.batch_size,
                                                shuffle=True, pin_memory=True, num_workers=72, prefetch_factor=20, persistent_workers=True)

test_data = DatasetFromFolder(data_dir, subfolder='validation', direction=params.direction, transform=transform)

test_data_loader = torch.utils.data.DataLoader(dataset=test_data,
                                               batch_size=params.batch_size,
                                               shuffle=False)

# test_input, test_target, test_mask = test_data_loader.__iter__().__next__()


G = Generator(3, params.ngf, 3)
D = Discriminator(6, params.ndf, 1)
G.cuda()
D.cuda()

G.normal_weight_init(mean=0.0, std=0.02)
D.normal_weight_init(mean=0.0, std=0.02)

BCE_loss = torch.nn.BCELoss().cuda()
L1_loss = torch.nn.L1Loss().cuda()
L2_loss = torch.nn.MSELoss().cuda()


perceptual_loss = PerceptualLoss().cuda()
style_loss = StyleLoss().cuda()
msgms_loss = MSGMS_Loss().cuda()

G_optimizer = torch.optim.Adam(G.parameters(), lr=params.lrG, betas=(params.beta1, params.beta2))
D_optimizer = torch.optim.Adam(D.parameters(), lr=params.lrD, betas=(params.beta1, params.beta2))

def adjust_learning_rate1(optimizer, epoch):
    lr = 0.001*(0.99**(epoch))
    print("lr is {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate2(optimizer, epoch):
    lr = 0.004*(0.99**(epoch))
    print("lr is {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

D_avg_losses = []
G_avg_losses = []

step = 0

loss_L1 = False
loss_L1_Style = False
loss_L1_SSIM_GMS = False
loss_L1_SSIM_GMS_Style = True

best_val_loss = 10000000000

for epoch in range(params.num_epochs):
    D_losses = []
    G_losses = []
    adjust_learning_rate1(G_optimizer, epoch)
    adjust_learning_rate2(D_optimizer, epoch)

    for i, (input, target, mask) in enumerate(train_data_loader):

        x_ = Variable(input.cuda())
        y_ = Variable(target.cuda())
        
        D_real_decision = D(x_, y_).squeeze()
        real_ = Variable(torch.ones(D_real_decision.size()).cuda())
        D_real_loss = BCE_loss(D_real_decision, real_)

        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        fake_ = Variable(torch.zeros(D_fake_decision.size()).cuda())
        D_fake_loss = BCE_loss(D_fake_decision, fake_)

        D_loss = (D_real_loss + D_fake_loss) * 0.5
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        gen_image = G(x_)
        D_fake_decision = D(x_, gen_image).squeeze()
        G_fake_loss = BCE_loss(D_fake_decision, real_)
       
        if loss_L1 == True:
            # using pure L1 loss
            l1_loss = params.lamb * L1_loss(gen_image, y_)
        elif loss_L1_SSIM_GMS_Style == True:
            loss_MSGMS = msgms_loss(gen_image, y_)
            loss_SSIM = 1 - ssim(gen_image, y_)
            gen_style_loss = style_loss(gen_image, y_) * 10
            l_rec = gen_style_loss + loss_MSGMS + loss_SSIM + L1_loss(gen_image, y_)
            l1_loss = params.lamb * l_rec

        G_loss = G_fake_loss + l1_loss
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        # loss values
        D_losses.append(D_loss.item())
        G_losses.append(G_loss.item())
        
        print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
              % (epoch+1, params.num_epochs, i+1, len(train_data_loader), D_loss.item(), G_loss.item()))
        step += 1

    writer.add_scalar('G_loss_mean', torch.mean(torch.FloatTensor(G_losses)), epoch)
    writer.add_scalar('D_loss_mean', torch.mean(torch.FloatTensor(D_losses)), epoch)

    if (epoch+1) % 10 == 0: 
        val_losses = 0.00
        # time_start = time.time()
        for i, (input, target, mask) in enumerate(test_data_loader):
            x_ = Variable(input.cuda())
            y_ = Variable(target.cuda())
            
            with torch.no_grad():
                gen_image = G(x_)
            if loss_L1 == True:
            # using pure L1 loss
                l1_loss =  L1_loss(gen_image, y_)
            elif loss_L1_SSIM_GMS_Style == True:
                loss_MSGMS = msgms_loss(gen_image, y_)
                loss_SSIM = 1 - ssim(gen_image, y_)
                gen_style_loss = style_loss(gen_image, y_) * 10
                l_rec = gen_style_loss + loss_MSGMS + loss_SSIM + L1_loss(gen_image, y_)
                l1_loss = params.lamb * l_rec

            loss_all = l1_loss
            val_losses += loss_all

        if val_losses < best_val_loss:
            best_val_loss = min(best_val_loss, val_losses)
            print("best_val_loss is {}".format(best_val_loss))
            torch.save(G.state_dict(), model_dir + 'best_G_param.pkl')
            torch.save(D.state_dict(), model_dir + 'best_D_param.pkl')
            print("the best model is epoch_{}".format(epoch + 1))

    # utils.plot_test_result(test_input, test_target, gen_image, epoch, save=True, save_dir=save_dir)
    if (epoch+1) % 50 == 0:
        torch.save(G.state_dict(), model_dir + '%d'%(epoch +1) +'generator_param.pkl')
        torch.save(D.state_dict(), model_dir + '%d'%(epoch +1) + 'discriminator_param.pkl')
    

