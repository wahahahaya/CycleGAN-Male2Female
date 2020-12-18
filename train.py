import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from model import  ResNetGenerator, Discriminator
from data import *
from utils import *

import os
import numpy as np
import math
import itertools

def train():
    # Parameters
    epochs = 200
    epoch = 0
    decay_epoch = 100
    sample_interval = 100
    dataset_name = "horse2zebra"
    img_height = 256
    img_width = 256
    channels = 3

    input_shape = (channels, img_height, img_width)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    #print('GPU State:', device)
    Tensor = torch.cuda.FloatTensor if device=='cuda' else torch.Tensor

    # image transformations
    # resize (100,100) 會間接影響到Discriminator的輸出size 如果要改 model.py 麻煩也要改
    data_process_steps = [
        transforms.Resize((100,100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    # 路徑麻煩你們自己改 然後dataset格式一定要像下面那樣
    # data loader
    # gender
    # |-testA
    # |-testB
    # |-trainA
    # |-trainB

    train_data = DataLoader(
        ImageDataset(
            "C:/Users/a8701/NTUST_dissertation/dataset/horse2zebra/horse2zebra",
            transforms_=data_process_steps,
            unaligned=True,
        ),
        batch_size=1,
        #shuffle=True,
        num_workers=4,
    )


    test_data = DataLoader(
        ImageDataset(
            "C:/Users/a8701/NTUST_dissertation/dataset/horse2zebra/horse2zebra",
            transforms_=data_process_steps,
            unaligned=True,
            mode="test",
        ),
        batch_size=5,
        #shuffle=True,
        num_workers=4,
    )

    # Build Model
    Gen_AB = ResNetGenerator(input_shape).to(device)
    Gen_BA = ResNetGenerator(input_shape).to(device)
    Dis_A = Discriminator(input_shape).to(device)
    Dis_B = Discriminator(input_shape).to(device)

    # Loss Function
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizer
    opt_G = torch.optim.Adam(
        itertools.chain(Gen_AB.parameters(),Gen_BA.parameters()),
        lr=0.00001
    )
    opt_D_A = torch.optim.Adam(
        Dis_A.parameters(),
        lr = 0.00001
    )

    opt_D_B = torch.optim.Adam(
        Dis_B.parameters(),
        lr = 0.00001
    )

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        opt_G, lr_lambda=LambdaLR(epochs, epoch, decay_epoch).step
    )
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        opt_D_A, lr_lambda=LambdaLR(epochs, epoch, decay_epoch).step
    )
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        opt_D_B, lr_lambda=LambdaLR(epochs, epoch, decay_epoch).step
    )

    # Buffers of previously generated samples
    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    #這邊就是輸出的影像 如果要呈現更多影像 改這邊
    def sample_images(batches_done):
        """Saves a generated sample from the test set"""
        imgs = next(iter(test_data))
        Gen_AB.eval()
        Gen_BA.eval()
        real_A = Variable(imgs["A"].type(Tensor))
        fake_B = Gen_AB(real_A)
        real_B = Variable(imgs["B"].type(Tensor))
        fake_A = Gen_BA(real_B)
        # Arange images along x-axis
        real_A = make_grid(real_A, nrow=5, normalize=True)
        real_B = make_grid(real_B, nrow=5, normalize=True)
        fake_A = make_grid(fake_A, nrow=5, normalize=True)
        fake_B = make_grid(fake_B, nrow=5, normalize=True)
        # Arange images along y-axis
        image_grid = torch.cat((real_A, fake_B, real_B, fake_A), 1)
        save_image(image_grid, "Images/%s/%s.png" % (dataset_name, batches_done), normalize=False)

    # train
    G_l=[]
    D_l=[]
    ADV_l=[]
    CYC_l=[]
    ID_l=[]
    
    for epoch in range(epochs):
        G_ll=[]
        D_ll=[]
        ADV_ll=[]
        CYC_ll=[]
        ID_ll=[]
        for i, batch in enumerate(train_data):
            
            # Set model input
            real_A = Variable(batch["A"].type(Tensor))
            real_B = Variable(batch["B"].type(Tensor))

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *Dis_A.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.ones((real_A.size(0), *Dis_A.output_shape))), requires_grad=False)

            #
            # Train Generators
            #

            Gen_AB.train()
            Gen_BA.train()

            opt_G.zero_grad()

            # Identity loss
            loss_id_A = criterion_identity(Gen_BA(real_A), real_A)
            loss_id_B = criterion_identity(Gen_AB(real_B), real_B)

            loss_identity = (loss_id_A + loss_id_B) / 2
            #print(loss_identity)

            # GAN loss
            fake_B = Gen_AB(real_A)
            loss_GAN_AB = criterion_GAN(Dis_B(fake_B), valid)
            fake_A = Gen_BA(real_B)
            loss_GAN_BA = criterion_GAN(Dis_A(fake_A), valid)

            loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

            # Cycle loss
            recov_A = Gen_BA(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = Gen_AB(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)

            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            # Total loss
            loss_G = loss_GAN + 10.0 * loss_cycle + 5.0 * loss_identity
            print(loss_G)

            loss_G.backward()
            opt_G.step()

            #
            # Train Discriminator A
            #

            opt_D_A.zero_grad()

            # Real loss
            loss_real = criterion_GAN(Dis_A(real_A), valid)

            # Fake loss (on batch of previously generated samples)
            fake_A_ = fake_A_buffer.push_and_pop(fake_A)
            loss_fake = criterion_GAN(Dis_A(fake_A_.detach()), fake)

            # Total loss
            loss_D_A = (loss_real + loss_fake) /2

            loss_D_A.backward()
            opt_D_A.step()

            #
            # Train Discrimainator B
            #

            opt_D_B.zero_grad()

            # Real loss
            loss_real = criterion_GAN(Dis_B(real_B), valid)

            # Fake loss (on batch of previously generated samples)
            fake_B_ = fake_B_buffer.push_and_pop(fake_B)
            loss_fake = criterion_GAN(Dis_B(fake_B.detach()), fake)

            # Total loss
            loss_D_B = (loss_real + loss_fake) / 2

            loss_D_B.backward()
            opt_D_B.step()

            loss_D = (loss_D_A + loss_D_B) / 2

            #
            # Log progress
            #
            batches_done = epoch * len(train_data) + i
            G_ll.append(loss_G.item())
            D_ll.append(loss_D.item())
            ADV_ll.append(loss_GAN.item())
            CYC_ll.append(lodd_cucle.item())
            ID_ll.append(loss_identity.item())
            print("Epoch: {}/{}, Batch: {}/{}, D loss: {:.4f}, G loss: {:.4f}, adv loss: {:.4f}, cycle loss: {:.4f}, idenity: {:.4f}".format(epoch,epochs,i,len(train_data),loss_D.item(),loss_G.item(),loss_GAN.item(),loss_cycle.item(),loss_identity.item()))
            
            # If at sample interval save image
            if batches_done % sample_interval == 0:
                sample_images(batches_done)
        
        G_l.append(G_ll.mean())
        D_l.append(D_ll.mean())
        ADV_l.append(ADV_ll.mean())
        CYC_l.append(CYC_ll.mean())
        ID_l.append(ID_ll.mean())
        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        torch.save(Gen_AB.state_dict(), "saved_models/%s/G_AB_%d.pth" % (dataset_name, epoch))
        torch.save(Gen_BA.state_dict(), "saved_models/%s/G_BA_%d.pth" % (dataset_name, epoch))
        torch.save(Dis_A.state_dict(), "saved_models/%s/D_A_%d.pth" % (dataset_name, epoch))
        torch.save(Dis_B.state_dict(), "saved_models/%s/D_B_%d.pth" % (dataset_name, epoch))

if __name__ == "__main__":
    train()
