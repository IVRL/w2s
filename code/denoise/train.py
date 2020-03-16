import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import DnCNN, DnCNN_BUIFD, MemNet, MemNet_BUIFD, RIDNET
from dataset import Dataset
from utils import *
from timeit import default_timer as timer
import pickle
from torchvision.utils import save_image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
parser = argparse.ArgumentParser(description="train_denoiser")
parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be < epochs")

parser.add_argument("--patch_size", type=int, default=64, help="patch size")
parser.add_argument("--stride", type=int, default=32, help="stride used for creating training patches")

parser.add_argument("--net", type=str, default="D", help='DnCNN (D), DnCNN_BUIFD (DF), MemNet (M), MemNet_BUIFD (MF), RIDNet (R)')
parser.add_argument("--img_avg", type=int, default=1, help="Number of images pre-averaged (determines the noise level)")
opt = parser.parse_args()


def main():
    
    # Load dataset
    print('Loading dataset ...\n')
    dataset = Dataset(img_avg=opt.img_avg, patch_size=opt.patch_size, stride=opt.stride)
    loader = DataLoader(dataset=dataset, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    print(f'{len(dataset)} training sample pairs loaded.')

    # Build model
    print(f'** Creating {opt.net} network **\n')
    model_channels = 1

    if opt.net == 'D':
        net = DnCNN(channels=model_channels, num_of_layers=17)
#     elif opt.net == 'DF':
#         net = DnCNN_BUIFD(channels=model_channels, num_of_layers=17)
    elif opt.net == 'M':
        net = MemNet(in_channels=model_channels)
#     elif opt.net == 'MF':
#         net = MemNet_BUIFD(in_channels=model_channels)
    elif opt.net == 'R':
        net = RIDNET(in_channels=model_channels)
    else:
        raise NotImplemented('Network model not implemented.')
    net.apply(weights_init_kaiming)

    
    # Loss metric
    criterion = nn.MSELoss(size_average=False)

    # Move to GPU
    model = nn.DataParallel(net).cuda()
    criterion.cuda()
    print('Trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
            
    
    # Training
    loss_log = np.zeros(opt.epochs)
    loss_batch_log = []
        
    for epoch in range(opt.epochs):
        start_time = timer()
        
        # Learning rate        
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / (10.)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('\nLearning rate = %f' % current_lr)

        # Train
        for idx, (noisy, target) in enumerate(loader):
            
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
        
            # Training step
            noise = noisy - target
            target, noisy = Variable(target.cuda()), Variable(noisy.cuda())
            noise = Variable(noise.cuda())
            
#             if opt.net[-1] != 'F':
            if opt.net == 'R':
                predicted_noise = noisy - model(noisy)
            else:
                predicted_noise = model(noisy)
            loss_noise = criterion(predicted_noise, noise) / (noisy.size()[0]*2)
            loss = loss_noise
                
#             else:
#                 out_train, out_noise_level_train = model(imgn_train)
                
#                 loss_img = criterion(out_train, noise) / (imgn_train.size()[0]*2)
#                 loss_noise_level = criterion(out_noise_level_train, noise_level_train) / (imgn_train.size()[0]*2)
#                 loss = loss_img + loss_noise_level

            loss.backward()
            optimizer.step()
            
            loss_batch_log.append(loss.item())
#             loss_image_log[epoch] += loss_img.item()
#             loss_noise_level_log[epoch] += loss_noise_level.item()
            loss_log[epoch] += loss.item()
            

        # Average out over all batches in the epoch
#         loss_image_log[epoch] = loss_image_log[epoch] / len(loader_train)
#         loss_noise_level_log[epoch] = loss_noise_level_log[epoch] / len(loader_train)
        loss_log[epoch] = loss_log[epoch] / len(loader)

        
        # Save model
        model_name = f'{opt.net}_{opt.img_avg}'
        model_dir = os.path.join('../../net_data/trained_denoisers/', model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model.state_dict(), os.path.join(model_dir, f'epoch_{epoch}.pth') )
        
        
        # Save logs and settings
        if ((epoch+1)%10) == 0:
            log_dict = {'loss_log': loss_log,
                        #'loss_image_log': loss_image_log,
                        #'loss_noise_level_log': loss_noise_level_log,
                        'loss_batch_log': np.asarray(loss_batch_log)}
            fname = os.path.join(model_dir, 'log_dict.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(log_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('wrote', fname)
        
            settings_dict = {'opt': opt}
            fname = os.path.join(model_dir, 'settings_dict.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(settings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            print('wrote', fname)

        
        # Ending-epoch message
        end_time = timer()
        print(f'Epoch {epoch} ({(end_time - start_time)/60.0:.1f} min):    loss={loss_log[epoch]:.4f}')
        
        
    print(f'Training {opt.net} complete for all epochs.')



if __name__ == "__main__":
    main()