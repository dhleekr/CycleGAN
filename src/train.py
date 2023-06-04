from utils.dataset import ImageDataset
from utils.lr_scheduler import LambdaLR
from utils.replaybuffer import ReplayBuffer
from src.models import Generator, Discriminator

import os
import argparse
import itertools
from PIL import Image
import yaml

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm

def load_config(config_name):
    with open(f'./configs/{config_name}.yaml') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets/horse2zebra/')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config')
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print(device)

    config = load_config(args.config)
    print(config)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    target_real = Variable(Tensor(config['batch_size']).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(config['batch_size']).fill_(0.0), requires_grad=False)

    # Dataset loader
    transforms_ = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size']), Image.BICUBIC),
        transforms.RandomCrop((config['cropped_size'], config['cropped_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    dataloader = DataLoader(
        ImageDataset(args.data_path, transforms_=transforms_, unaligned=True),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
    )

    # Models
    g_AB = Generator(config['g_input_channels'],
                     config['g_output_channels'],
                     **config['generator_kwargs']).to(device)
                     
    g_BA = Generator(config['g_output_channels'],
                     config['g_input_channels'],
                     **config['generator_kwargs']).to(device)
    
    d_A = Discriminator(config['d_input_channels'], 
                        **config['discriminator_kwargs']).to(device)
    
    d_B = Discriminator(config['d_output_channels'], 
                        **config['discriminator_kwargs']).to(device)

    # Training setup
    g_optimizer = torch.optim.Adam(itertools.chain(g_AB.parameters(), g_BA.parameters()), 
                                   lr=config['learning_rate'], 
                                   betas=(0.5, 0.999))
    d_A_optimizer = torch.optim.Adam(d_A.parameters(),
                                      lr=config['learning_rate'],
                                      betas=(0.5, 0.999))
    d_B_optimizer = torch.optim.Adam(d_B.parameters(),
                                      lr=config['learning_rate'],
                                      betas=(0.5, 0.999))
    
    g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(g_optimizer, lr_lambda=LambdaLR(config['train_epochs'], 0, config['decay_epoch']).step)
    d_A_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(d_A_optimizer, lr_lambda=LambdaLR(config['train_epochs'], 0, config['decay_epoch']).step)
    d_B_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(d_B_optimizer, lr_lambda=LambdaLR(config['train_epochs'], 0, config['decay_epoch']).step)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()
    
    # Losses
    criterion_GAN = torch.nn.MSELoss().to(device)
    criterion_cycle = torch.nn.L1Loss().to(device)
    criterion_identity = torch.nn.L1Loss().to(device)

    wandb.init(
        project='cyclegan',
        entity='dhleekr',
        sync_tensorboard=True,
    )
    wandb.config.update(config)

    wandb.watch(g_AB)
    wandb.watch(g_BA)
    wandb.watch(d_A)
    wandb.watch(d_B)

    # Training
    for epoch in tqdm(range(config['train_epochs'])):
        for i, batch in tqdm(enumerate(dataloader)):
            real_A = Variable(batch['A'].to(device))
            real_B = Variable(batch['B'].to(device))
            
            # Reconstruction loss
            identity_A = g_BA(real_A)
            identity_B = g_AB(real_B)
            loss_identity = criterion_identity(identity_A, real_A) + criterion_identity(identity_B, real_B)

            # GAN loss
            fake_A = g_BA(real_B)
            disciriminate_fake_A = d_A(fake_A)
            loss_GAN_AB = criterion_GAN(disciriminate_fake_A, target_real)

            fake_B = g_AB(real_A)
            disciriminate_fake_B = d_B(fake_B)
            loss_GAN_BA = criterion_GAN(disciriminate_fake_B, target_real)

            # Cycle loss
            recovered_A = g_BA(fake_B)
            recovered_B = g_AB(fake_A)
            loss_cycle = criterion_cycle(recovered_A, real_A) + criterion_cycle(recovered_B, real_B)

            loss_generator = loss_identity + loss_GAN_AB + loss_GAN_BA + config['lambda']*loss_cycle

            # Update generator
            g_optimizer.zero_grad()
            loss_generator.backward()
            g_optimizer.step()

            
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            fake_B = fake_B_buffer.push_and_pop(fake_B)

            discriminate_real_A = d_A(real_A)
            discriminate_real_B = d_B(real_B)
            disciriminate_fake_A = d_A(fake_A.detach())
            disciriminate_fake_B = d_B(fake_B.detach())

            loss_discriminator_A = 0.5*(criterion_GAN(discriminate_real_A, target_real) + criterion_GAN(disciriminate_fake_A, target_fake))
            loss_discriminator_B = 0.5*(criterion_GAN(discriminate_real_B, target_real) + criterion_GAN(disciriminate_fake_B, target_fake))

            # Update discriminator
            d_A_optimizer.zero_grad()
            d_B_optimizer.zero_grad()
            loss_discriminator_A.backward()
            loss_discriminator_B.backward()
            d_A_optimizer.step()
            d_B_optimizer.step()

            if i % 10 == 0:
                print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss_generator.item()}')

            log_real_A = real_A[0].cpu().detach().numpy().reshape((config['image_size'], config['image_size'], 3))
            log_real_B = real_B[0].cpu().detach().numpy().reshape((config['image_size'], config['image_size'], 3))
            log_fake_A = fake_A[0].cpu().detach().numpy().reshape((config['image_size'], config['image_size'], 3))
            log_fake_B = fake_B[0].cpu().detach().numpy().reshape((config['image_size'], config['image_size'], 3))

            wandb.log({
                'loss_recon': loss_identity.item(),
                'loss_GAN_AB': loss_GAN_AB.item(),
                'loss_GAN_BA': loss_GAN_BA.item(),
                'loss_cycle': loss_cycle.item(),
                'loss_generator': loss_generator.item(),
                'loss_discriminator_A': loss_discriminator_A.item(),
                'loss_discriminator_B': loss_discriminator_B.item(),
                "real_A": wandb.Image(log_real_A),
                "real_B": wandb.Image(log_real_B),
                "fake_A": wandb.Image(log_fake_A),
                "fake_B": wandb.Image(log_fake_B),
            })

        # Update learning rate
        g_lr_scheduler.step()
        d_A_lr_scheduler.step()
        d_B_lr_scheduler.step()

        # Save checkpoints
        if not os.path.exists(f'models_{args.data_path}'):
            os.makedirs(f'models_{args.data_path}')
            
        torch.save(g_AB.state_dict(), f'models_{args.data_path}/g_AB.pth')
        torch.save(g_BA.state_dict(), f'models_{args.data_path}/g_BA.pth')
        torch.save(d_A.state_dict(), f'models_{args.data_path}/d_A.pth')
        torch.save(d_B.state_dict(), f'models_{args.data_path}/d_B.pth')        

    wandb.finish()

if __name__ == '__main__':
    main()