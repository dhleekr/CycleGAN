import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import yaml

from models import Generator
from utils.dataset import ImageDataset

def load_config(config_name):
    with open(f'./configs/{config_name}.yaml') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='datasets/horse2zebra/')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--config', type=str, default='config')
    parser.add_argument('--generator_AB', type=str, default='models/g_AB.pth')
    parser.add_argument('--generator_BA', type=str, default='models/g_BA.pth')
    args = parser.parse_args()
    print(args)

    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print(device)

    config = load_config(args.config)
    print(config)

    g_AB = Generator(config['g_input_channels'],
                     config['g_output_channels'],
                     **config['generator_kwargs']).to(device)
                     
    g_BA = Generator(config['g_output_channels'],
                     config['g_input_channels'],
                     **config['generator_kwargs']).to(device)

    g_AB.load_state_dict(torch.load(args.generator_AB)).eval()
    g_AB.load_state_dict(torch.load(args.generator_BA)).eval()

    # Inputs & targets memory allocation
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    # Dataset loader
    transforms_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    dataloader = DataLoader(
        ImageDataset(args.data_path, transforms_=transforms_, mode='test'), 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
    )

    if not os.path.exists('results/A'):
        os.makedirs('results/A')
    if not os.path.exists('results/B'):
        os.makedirs('results/B')

    for i, batch in enumerate(dataloader):
        real_A = Variable(batch['A'].to(device))
        real_B = Variable(batch['B'].to(device))

        # Generate output
        fake_B = 0.5*(g_AB(real_A).data + 1.0)
        fake_A = 0.5*(g_BA(real_B).data + 1.0)

        # Save image files
        save_image(fake_A, 'results/A/%04d.png' % (i+1))
        save_image(fake_B, 'results/B/%04d.png' % (i+1))

        sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

    sys.stdout.write('\n')


if __name__ == '__main__':
    main()