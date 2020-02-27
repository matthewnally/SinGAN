import SinGAN.functions as functions
import SinGAN.models as models
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import math
import matplotlib.pyplot as plt
from SinGAN.imresize import imresize

def train(opt,Gs,Zs,reals,NoiseAmp):
    real_ = functions.read_images(opt)
    in_s = 0
    scale_num = 0
    real = [imresize(real_[0],opt.scale1,opt), imresize(real_[1],opt.scale1,opt)]
    reals = [functions.creat_reals_pyramid(real[0],reals,opt), functions.creat_reals_pyramid(real[1],reals,opt)]
    nfc_prev = 0

    while scale_num<opt.stop_scale+1:
        opt.nfc = min(opt.nfc_init * pow(2, math.floor(scale_num / 4)), 128)
        opt.min_nfc = min(opt.min_nfc_init * pow(2, math.floor(scale_num / 4)), 128)

        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' % (opt.out_,scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
                pass

        #plt.imsave('%s/in.png' %  (opt.out_), functions.convert_image_np(real), vmin=0, vmax=1)
        #plt.imsave('%s/original.png' %  (opt.out_), functions.convert_image_np(real_), vmin=0, vmax=1)
        plt.imsave('%s/real_scale1.png' %  (opt.outf), functions.convert_image_np(reals[0][scale_num]), vmin=0, vmax=1)
        plt.imsave('%s/real_scale2.png' %  (opt.outf), functions.convert_image_np(reals[1][scale_num]), vmin=0, vmax=1)

        D_curr1,G_curr1 = init_models(opt)
        D_curr2,G_curr2 = init_models(opt)
        D_curr = [D_curr1, D_curr2]
        G_curr = [G_curr1, G_curr2]

        if (nfc_prev==opt.nfc):
            G_curr[0].load_state_dict(torch.load('%s/%d/netG1.pth' % (opt.out_,scale_num-1)))
            D_curr[0].load_state_dict(torch.load('%s/%d/netD1.pth' % (opt.out_,scale_num-1)))
            G_curr[1].load_state_dict(torch.load('%s/%d/netG2.pth' % (opt.out_,scale_num-1)))
            D_curr[1].load_state_dict(torch.load('%s/%d/netD2.pth' % (opt.out_,scale_num-1)))

        z_curr,in_s,G_curr = train_single_scale(D_curr,G_curr,reals,Gs,Zs,in_s,NoiseAmp,opt)

        G_curr[0] = functions.reset_grads(G_curr[0],False)
        G_curr[0].eval()
        D_curr[0] = functions.reset_grads(D_curr[0],False)
        D_curr[0].eval()
        G_curr[1] = functions.reset_grads(G_curr[1],False)
        G_curr[1].eval()
        D_curr[1] = functions.reset_grads(D_curr[1],False)
        D_curr[1].eval()

        Gs.append(G_curr)
        Zs.append(z_curr)
        NoiseAmp.append(opt.noise_amp)

        torch.save(Zs, '%s/Zs.pth' % (opt.out_))
        torch.save(Gs, '%s/Gs.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(NoiseAmp, '%s/NoiseAmp.pth' % (opt.out_))

        scale_num+=1
        nfc_prev = opt.nfc
        del D_curr,G_curr
    return



def train_single_scale(netD,netG,reals,Gs,Zs,in_s,NoiseAmp,opt,centers=None):

    real = [reals[0][len(Gs)], reals[1][len(Gs)]]
    print(len(real))
    print(real[0].shape)
    opt.nzx = real[0].shape[2]#+(opt.ker_size-1)*(opt.num_layer)
    opt.nzy = real[0].shape[3]#+(opt.ker_size-1)*(opt.num_layer)
    opt.receptive_field = opt.ker_size + ((opt.ker_size-1)*(opt.num_layer-1))*opt.stride
    pad_noise = int(((opt.ker_size - 1) * opt.num_layer) / 2)
    pad_image = int(((opt.ker_size - 1) * opt.num_layer) / 2)

    m_noise = nn.ZeroPad2d(int(pad_noise))
    m_image = nn.ZeroPad2d(int(pad_image))

    alpha = opt.alpha

    fixed_noise = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy],device=opt.device)
    z_opt = torch.full(fixed_noise.shape, 0, device=opt.device)
    z_opt = m_noise(z_opt)

    # setup optimizer
    optimizerD = [optim.Adam(netD[i].parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999)) for i in range(2)]
    optimizerG = [optim.Adam(netG[i].parameters(), lr=opt.lr_g, betas=(opt.beta1, 0.999)) for i in range(2)]
    schedulerD = [torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD[i],milestones=[1600],gamma=opt.gamma) for i in range(2)]
    schedulerG = [torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG[i],milestones=[1600],gamma=opt.gamma) for i in range(2)]

    errD2plot = []
    errG2plot = []
    D_real2plot = []
    D_fake2plot = []
    z_opt2plot = []

    for epoch in range(opt.niter):
        if (Gs == []) & (opt.mode != 'SR_train'):
            z_opt = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            z_opt = m_noise(z_opt.expand(1,3,opt.nzx,opt.nzy))
            noise_ = functions.generate_noise([1,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_.expand(1,3,opt.nzx,opt.nzy))
        else:
            noise_ = functions.generate_noise([opt.nc_z,opt.nzx,opt.nzy], device=opt.device)
            noise_ = m_noise(noise_)

        ############################
        # (1) Update D network: maximize D(x) + D(G(z))
        ###########################
        for j in range(opt.Dsteps):
            # train with real
            netD[0].zero_grad()
            netD[1].zero_grad()

            output1 = netD[0](real[1]).to(opt.device) # first discriminator trains vs 2nd image
            output2 = netD[1](real[0]).to(opt.device)
            #D_real_map = output.detach()
            errD_real1 = -output1.mean()#-a
            errD_real2 = -output2.mean()#-a
            errD_real1.backward(retain_graph=True)
            errD_real2.backward(retain_graph=True)
            # D_x1 = -errD_real.item()
            # D_x2 = -errD_real.item()
            # train with fake
            if (j==0) & (epoch == 0):
                if (Gs == []) & (opt.mode != 'SR_train'):
                    prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    in_s = prev
                    prev = m_image(prev)
                    z_prev = torch.full([1,opt.nc_z,opt.nzx,opt.nzy], 0, device=opt.device)
                    z_prev = m_noise(z_prev)
                    opt.noise_amp = 1

                else:
                    prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                    prev = m_image(prev)
                    z_prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rec',m_noise,m_image,opt)
                    criterion = nn.MSELoss()
                    print(real[0].shape)
                    RMSE = torch.sqrt(criterion(real[0], z_prev))
                    opt.noise_amp = opt.noise_amp_init*RMSE
                    z_prev = m_image(z_prev)
            else:
                prev = draw_concat(Gs,Zs,reals,NoiseAmp,in_s,'rand',m_noise,m_image,opt)
                prev = m_image(prev)

            if (Gs == []) & (opt.mode != 'SR_train'):
                noise = noise_
            else:
                noise = opt.noise_amp*noise_+prev

            # the money maker
            fake1 = netG[0](noise.detach(),prev)
            fake2 = netG[1](noise.detach(),fake1)
            output1 = netD[0](fake1.detach())
            output2 = netD[1](fake2.detach())
            errD_fake1 = output1.mean()
            errD_fake2 = output2.mean()
            errD_fake1.backward(retain_graph=True)
            errD_fake2.backward(retain_graph=True)

            # D_G_z = output.mean().item() # for graphing only

            # this is wasserstiens
            gradient_penalty = functions.calc_gradient_penalty(netD, real, [fake1, fake2], opt.lambda_grad, opt.device)
            gradient_penalty.backward()

            #errD = errD_real + errD_fake + gradient_penalty
            optimizerD[0].step()
            optimizerD[1].step()

        #errD2plot.append(errD.detach())

        ############################
        # (2) Update G network: maximize D(G(z))
        ###########################

        for j in range(opt.Gsteps):
            netG[0].zero_grad()
            netG[1].zero_grad()
            output1 = netD[0](fake1)
            output2 = netD[1](fake2)
            #D_fake_map = output.detach()
            errG1 = -output1.mean()
            errG1.backward(retain_graph=True)
            errG2 = -output2.mean()
            errG2.backward(retain_graph=True)
            if alpha!=0:
                loss = nn.MSELoss()
                Z_opt = opt.noise_amp*z_opt+z_prev

                im1 = m_image(netG[0](Z_opt.detach(),z_prev))
                #print(im1.size(), Z_opt.detach().size(), real[0].size())
                rec_loss = alpha*loss(netG[1](im1, z_prev),real[0])
                rec_loss.backward(retain_graph=True)
                rec_loss = rec_loss.detach()

                # Cycle Consistency Loss
            else:
                Z_opt = z_opt
                rec_loss = 0

            optimizerG[0].step()
            optimizerG[1].step()
        # errG2plot.append(errG.detach()+rec_loss)
        # D_real2plot.append(D_x)
        # D_fake2plot.append(D_G_z)x`
        # z_opt2plot.append(rec_loss)

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            print('scale %d:[%d/%d]' % (len(Gs), epoch, opt.niter))

        if epoch % 500 == 0 or epoch == (opt.niter-1):
            # plt.imsave('%s/fake_sample.png' %  (opt.outf), functions.convert_image_np(fake.detach()), vmin=0, vmax=1)
            # plt.imsave('%s/G(z_opt).png'    % (opt.outf),  functions.convert_image_np(netG(Z_opt.detach(), z_prev).detach()), vmin=0, vmax=1)
            #plt.imsave('%s/D_fake.png'   % (opt.outf), functions.convert_image_np(D_fake_map))
            #plt.imsave('%s/D_real.png'   % (opt.outf), functions.convert_image_np(D_real_map))
            #plt.imsave('%s/z_opt.png'    % (opt.outf), functions.convert_image_np(z_opt.detach()), vmin=0, vmax=1)
            #plt.imsave('%s/prev.png'     %  (opt.outf), functions.convert_image_np(prev), vmin=0, vmax=1)
            #plt.imsave('%s/noise.png'    %  (opt.outf), functions.convert_image_np(noise), vmin=0, vmax=1)
            #plt.imsave('%s/z_prev.png'   % (opt.outf), functions.convert_image_np(z_prev), vmin=0, vmax=1)


            torch.save(z_opt, '%s/z_opt.pth' % (opt.outf))

        schedulerD[0].step()
        schedulerG[0].step()
        schedulerD[1].step()
        schedulerG[1].step()

    functions.save_networks(netG,netD,z_opt,opt)
    return z_opt,in_s,netG

def draw_concat(Gs,Zs,reals,NoiseAmp,in_s,mode,m_noise,m_image,opt):
    G_z = in_s
    if len(Gs) > 0:
        if mode == 'rand':
            count = 0
            pad_noise = int(((opt.ker_size-1)*opt.num_layer)/2)
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals[0],reals[0][1:],NoiseAmp):
                if count == 0:
                    z = functions.generate_noise([1, Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                    z = z.expand(1, 3, z.shape[2], z.shape[3])
                else:
                    z = functions.generate_noise([opt.nc_z,Z_opt.shape[2] - 2 * pad_noise, Z_opt.shape[3] - 2 * pad_noise], device=opt.device)
                z = m_noise(z)
                G_z = G_z[:,:,0:real_curr.shape[2],0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*z+G_z
                z_in2= noise_amp*z+m_image(G[0](z_in.detach(),G_z))
                G_z = G[1](z_in2.detach(),G[0](z_in.detach(),G_z))
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                count += 1
        if mode == 'rec':
            count = 0
            for G,Z_opt,real_curr,real_next,noise_amp in zip(Gs,Zs,reals[0],reals[0][1:],NoiseAmp):
                G_z = G_z[:, :, 0:real_curr.shape[2], 0:real_curr.shape[3]]
                G_z = m_image(G_z)
                z_in = noise_amp*Z_opt+G_z
                z_in2 = noise_amp*Z_opt+m_image(G[0](z_in.detach(),G_z))
                G_z = G[1](z_in2.detach(),G[0](z_in2.detach(),G_z))
                G_z = imresize(G_z,1/opt.scale_factor,opt)
                G_z = G_z[:,:,0:real_next.shape[2],0:real_next.shape[3]]
                #if count != (len(Gs)-1):
                #    G_z = m_image(G_z)
                count += 1
    return G_z

def init_models(opt):

    #generator initialization:
    netG = models.GeneratorConcatSkip2CleanAdd(opt).to(opt.device)
    netG.apply(models.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    #discriminator initialization:
    netD = models.WDiscriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    return netD, netG
