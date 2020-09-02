import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
from dataset import ImageDataset
from model import UnetGenerator, Discriminator

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_number', type=int, default=0)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--model_path', type=str, default='./results/models/')
parser.add_argument('--dataset_dir', type=str, default='./images/')
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--l1_lambda', type=float, default=10)
parser.add_argument('--pose_lambda', type=float, default=1.0)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
args = parser.parse_args()

os.makedirs(args.model_path, exist_ok=True)

device = 'cuda:{}'.format(args.gpu_number) if torch.cuda.is_available() else 'cpu'
# create dataset class
train_dataset = ImageDataset(args.image_size, args.dataset_dir, 'train')
# create data loader class
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

# build model
generator = UnetGenerator(in_channels=3, out_channels=3)
discriminator = Discriminator(in_channels=6, out_channels=1)

# build optimizer
optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

# criterion(loss)
loss_func = nn.BCEWithLogitsLoss()
l1_loss_func = nn.L1Loss()
mse_loss = nn.MSELoss()

# for patch GAN
batch_size = args.batch_size
# ToDo: fix tensor size according to discriminator output shape
ones = torch.ones(batch_size, 1, 4, 4).to(device)
zeros = torch.zeros(batch_size, 1, 4, 4).to(device)

for e in range(args.epoch):
    loss_gene = 0
    loss_disc = 0
    
    for idx, data in enumerate(train_dataloader):
        
        real_rgb, real_pose_rgb, real_pose = data
        batch_len = len(real_rgb)
        
        # adjust data format corresponding to device
        real_rgb = real_rgb.to(device)
        real_pose_rgb = real_pose_rgb.to(device)
        
        # create fake color
        fake_rgb, est_pose = generator(real_pose_rgb)
        fake_rgb_tensor = fake_rgb.detach()  # save fake rgb data temporarily
        
        # Generator Training
        
        disc_input = torch.cat([fake_rgb, real_pose_rgb], dim=1)
        out = discriminator(disc_input)
        # BCE loss
        loss_g_bce = loss_func(out, ones[:batch_len])
        # l1 loss
        loss_g_l1 = l1_loss_func(fake_rgb, real_rgb)
        # pose loss
        loss_pose = mse_loss(est_pose, real_pose)
        
        loss_g = loss_g_bce + args.l1_lambda * loss_g_l1 + args.pose_lambda * loss_pose
        
        
        # back propagation for generator
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        loss_g.backward()
        optimizer_g.step()
        
        
        # Discriminator
        # real part
        disc_real_input = torch.cat([real_rgb, real_rgb_pose], dim=1)
        real_out = discriminator(disc_real_input)
        loss_d_real = loss_func(real_out, ones[:batch_len])
        
        # fake part
        disc_fake_input = torch.cat([fake_rgb_tensor, real_rgb_pose], dim=1)
        fake_out = discriminator(disc_fake_input)
        loss_d_fake = loss_func(fake_out, zeros[:batch_len])
        
        loss_d = loss_d_fake + loss_d_real
        
        # backpropagation for discriminator
        optimizer_g.zero_grad()
        optimizer_d.zero_grad()
        loss_d.backward()
        optimizer_d.step()
        
        loss_gene += loss_g.item()
        loss_disc += loss_d.item()
        
    m_loss_g = loss_gene / len(train_dataloader)
    m_loss_d = loss_disc / len(train_dataloader)
        
    print('=====> Epoch [{}]: Loss D: {:.4f}  Loss G: {:.4f}'.format(e, m_loss_d, m_loss_g))
    
    if e % args.save_freq == 0:
        generator_state_dict = {'epoch': e, 'model': generator.state_dict(), 'optimizer':optimizer_g.state_dict(),
                                'loss_g': loss_g}
        generator_path = args.model_path + 'checkpoint_generator_%05d.pth.tar' % e
        torch.save(generator_state_dict, generator_path)
        
        discriminator_state_dict = {'epoch': e, 'model': discriminator.state_dict(), 'optimizer':optimizer_d.state_dict(),
                                'loss_g': loss_d}
        discriminator_path = args.model_path + 'checkpoint_discriminator_%05d.pth.tar' % e
        torch.save(discriminator_state_dict, discriminator_path)
        