import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                               out_channels=in_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.bn1 = nn.BatchNorm2d(num_features=in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.bn2 = nn.BatchNorm2d(num_features=in_channels)
        self.conv3 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=padding)
        self.bn3 = nn.BatchNorm2d(num_features=out_channels)
        
    def forward(self, x):
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn1(self.conv1(x)))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn2(self.conv2(x)))
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn3(self.conv3(x)))
        
        return x


class UnetGenerator(nn.Module):
    
    def __init__(self, in_channels, out_channels, mode='nearest'):
        super(UnetGenerator, self).__init__()
        self.mode = mode
        # input => (3, 512, 512)
        # encoder
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1)  # (32, 256, 256)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)  # (64, 128, 128)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)  # (128, 64, 64)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)  # (256, 32, 32)
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)  # (512, 16, 16)
        self.bn5 = nn.BatchNorm2d(num_features=512)
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1)  # (1024, 8, 8)
        self.bn6 = nn.BatchNorm2d(num_features=1024)
        
        # fc layers
        self.fc1 = nn.Linear(in_features=1024 * 8 *8, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=256)
        self.fc3 = nn.Linear(in_features=256, out_features=3 * 3 + 3 * 4 + 1)  # 21 + 1(K matrix, Rt matrix, focal length)
        
        # decoder
        if self.mode:
            self.deconv0 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1)  # (512, 8, 8)
            self.debn0 = nn.BatchNorm2d(num_features=512)
            
            self.upsample1 = nn.Upsample(scale_factor=2, mode=self.mode)  # (512, 16, 16)
            self.deconv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1)  # (256, 16, 16)
            self.debn1 = nn.BatchNorm2d(num_features=256)
            
            self.upsample2 = nn.Upsample(scale_factor=2, mode=self.mode)  # (256, 32, 32)
            self.deconv2 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)  # (128, 32, 32)
            self.debn2 = nn.BatchNorm2d(num_features=128)
            
            self.upsample3 = nn.Upsample(scale_factor=2, mode=self.mode)  # (128, 64, 64)
            self.deconv3 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)  # (64, 64, 64)
            self.debn3 = nn.BatchNorm2d(num_features=64)
            
            self.upsample4 = nn.Upsample(scale_factor=2, mode=self.mode)  # (64, 128, 128)
            self.deconv4 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)  # (32, 128, 128)
            self.debn4 = nn.BatchNorm2d(num_features=32)
            
            self.upsample5 = nn.Upsample(scale_factor=2, mode=self.mode)  # (32, 256, 256)
            self.deconv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)  # (32, 256, 256)
            self.debn5 = nn.BatchNorm2d(num_features=32)
            
            self.upsample6 = nn.Upsample(scale_factor=2, mode=self.mode)  # (32, 512, 512)
            self.deconv6 = nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=3, stride=1, padding=1)  # (3, 512, 512)
            
        else:
            self.upconv1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0)  # (512, 16, 16)
            self.convblock1 = ConvBlock(in_channels=1024, out_channels=256, kernel_size=3, stride=1, padding=1)  # (256, 16, 16)
            self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0)  # (256, 32, 32)
            self.convblock2 = ConvBlock(in_channels=512, out_channels=128, kernel_size=3, stride=1, padding=1)  # (128, 32, 32)
            self.upconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0)  # (128, 64, 64)
            self.convblock3 = ConvBlock(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)  # (64, 64, 64)
            self.upconv4 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0)  # (64, 128, 128)
            self.convblock4 = ConvBlock(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1)  # (32, 128, 128)
            self.upconv5 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, padding=0)  # (32, 256, 256)
            self.convblock5 = ConvBlock(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)  # (32, 256, 256)
            self.upconv6 = nn.ConvTranspose2d(in_channels=32, out_channels=out_channels, kernel_size=2, stride=2, padding=0)  # (3, 512, 512)    
        
        
        
    def forward(self, x):
        # encoder
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn1(self.conv1(x)))  # (batch_size, 32, 256, 256)
        x1 = x
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn2(self.conv2(x)))  # (batch_size, 64, 128, 128)
        x2 = x
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn3(self.conv3(x)))  # (batch_size, 128, 64, 64)
        x3 = x
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn4(self.conv4(x)))  # (batch_size, 256, 32, 32)
        x4 = x
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn5(self.conv5(x)))  # (batch_size, 512, 16, 16)
        x5 = x
        x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.bn6(self.conv6(x)))  # (batch_size, 1024, 8, 8)
        
        # split two NN(decoder + fc layers)
        fc_x = x.view(x.size(0), -1)  # (batch_size, 1024 * 8 * 8)
        dec_x = x  # (batch_size, 1024, 8, 8)
        
        # fc_layer
        fc_x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.fc1(fc_x))
        fc_x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.fc2(fc_x))
        fc_x = nn.Tanh()(self.fc3(fc_x))
        print('fully connected layer output:', fc_x)
        
        # decoder
        if self.mode:
            dec_x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.debn0(self.deconv0(dec_x)))  # (batch_size, 512, 8, 8)
            
            dec_x = self.upsample1(dec_x)  # (batch_size, 512, 16, 16)
            dec_x = torch.cat([dec_x, x5], dim=1)  # (batch_size, 1024, 16, 16)
            dec_x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.debn1(self.deconv1(dec_x)))  # (batch_size, 256, 16, 16)
            
            dec_x = self.upsample2(dec_x)  # (batch_size, 256, 32, 32)
            dec_x = torch.cat([dec_x, x4], dim=1)  # (batch_size, 512, 32, 32)
            dec_x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.debn2(self.deconv2(dec_x)))  # (batch_size, 128, 32, 32)
            
            dec_x = self.upsample3(dec_x)  # (batch_size, 128, 64, 64)
            dec_x = torch.cat([dec_x, x3], dim=1)  # (batch_size, 256, 64, 64)
            dec_x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.debn3(self.deconv3(dec_x)))  # (batch_size, 64, 64, 64)
            
            dec_x = self.upsample4(dec_x)  # (batch_size, 64, 128, 128)
            dec_x = torch.cat([dec_x, x2], dim=1)  # (batch_size, 128, 128, 128)
            dec_x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.debn4(self.deconv4(dec_x)))  # (batch_size, 32, 128, 128)
            
            dec_x = self.upsample5(dec_x)  # (batch_size, 32, 256, 256)
            dec_x = torch.cat([dec_x, x1], dim=1)  # (batch_size, 64, 256, 256)
            dec_x = nn.LeakyReLU(negative_slope=0.2, inplace=True)(self.debn5(self.deconv5(dec_x)))  # (batch_size, 32, 256, 256)
            
            dec_x = nn.Tanh()(self.deconv6(self.upsample6(dec_x)))  # (batch_size, 3, 512, 512)
            
        else:
            dec_x = self.upconv1(dec_x)  # (batch_size, 512, 16, 16)
            dec_x = torch.cat([dec_x, x5], dim=1)  # (batch_size, 1024, 16, 16)
            dec_x = self.convblock1(dec_x)  # (batch_size, 256, 16, 16)
            
            dec_x = self.upconv2(dec_x)  # (batch_size, 256, 32, 32)
            dec_x = torch.cat([dec_x, x4], dim=1)  # (batch_size, 512, 32, 32)
            dec_x = self.convblock2(dec_x)  # (batch_size, 128, 32, 32)
            
            dec_x = self.upconv3(dec_x)  # (batch_size, 128, 64, 64)
            dec_x = torch.cat([dec_x, x3], dim=1)  # (batch_size, 256, 64, 64)
            dec_x = self.convblock3(dec_x)  # (batch_size, 64, 64, 64)
            
            dec_x = self.upconv4(dec_x)  # (batch_size, 64, 128, 128)
            dec_x = torch.cat([dec_x, x2], dim=1)  # (batch_size, 128, 128, 128)
            dec_x = self.convblock4(dec_x)  # (batch_size, 32, 128, 128)
            
            dec_x = self.upconv5(dec_x)  # (batch_size, 32, 256, 256)
            dec_x = torch.cat([dec_x, x1], dim=1)  # (batch_size, 64, 256, 256)
            dec_x = self.convblock5(dec_x)  # (batch_size, 32, 256, 256)
            
            dec_x = nn.Tanh()(self.upconv6(dec_x))  # (batch_size, 3, 512, 512)
        
        
        return dec_x, fc_x
    
    
class Discriminator(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super(Discriminator, self).__init__()
        # input => (6, 128, 128)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2, padding=1)  # (32, 64, 64)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)  # (64, 32, 32)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)  # (128, 16, 16)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)  # (256, 8, 8)
        self.bn4 = nn.BatchNorm2d(num_features=256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)  # (512, 4, 4)
        self.bn5 = nn.BatchNorm2d(num_features=512)
        self.out_patch = nn.Conv2d(in_channels=512, out_channels=out_channels, )  
        
    def forward(self, x):
        
        return x  
