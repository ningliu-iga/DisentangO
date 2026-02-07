import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.fft
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial
from itertools import chain

from timeit import default_timer
from utilities3 import *
import os
import sys


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x, y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class MetaFNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, nlayer, task_num, in_width=3, out_width=2):
        super(MetaFNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.nlayer = nlayer

        self.convlayer = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w = nn.Conv2d(self.width, self.width, 1)

        self.fc0 = nn.ModuleList([nn.Linear(in_width, self.width) for _ in range(task_num)])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, out_width)

    def forward(self, x, task_idx):
        batch_size = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        grids = self.get_grid(x.shape).to(x.device)
        x = torch.cat((x, grids), dim=-1)

        x = self.fc0[task_idx](x)
        x = x.permute(0, 3, 1, 2)

        for _ in range(self.nlayer - 1):
            x1 = self.convlayer(x)
            x2 = self.w(x)
            x = F.gelu(x1 + x2) / self.nlayer + x

        x1 = self.convlayer(x)
        x2 = self.w(x)
        x = (x1 + x2) / self.nlayer + x

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        # return torch.cat([x[:, :, :, 0], x[:, :, :, 1]], dim=1)
        return x.permute(0, 3, 1, 2).reshape(batch_size, -1, size_y)

    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)


def read_train_data(s, task_num, data_dir, ntrain_per_task, param_filename):
    # mat_params = np.loadtxt(data_dir + '/' + param_filename)
    mat_params_idx = np.loadtxt(data_dir + '/' + param_filename)[:, 0]
    # mat_params = mat_params[:task_num, 1:]
    # # input material parameters [E,nv,K1/K2,K2,alpha]
    # mat_params[:, 2] = mat_params[:, 2] / mat_params[:, 3]

    f_train = []
    y_train = []
    for fcount in range(task_num):
        idx_file = int(mat_params_idx[fcount])
        filename = 'displacement_neumann%d.txt' % idx_file
        # print(filename)
        f_all = np.loadtxt(data_dir + '/Traction_neumann.txt')
        y_all = np.loadtxt(data_dir + '/' + filename)

        f_all = torch.from_numpy(np.reshape(f_all[:ntrain_per_task], (ntrain_per_task, 1, s))).to(torch.float)
        f_all = f_all.repeat(1, s, 1)

        train_u = np.reshape(y_all[:ntrain_per_task], (ntrain_per_task, s, s, 2))
        train_u_2 = np.concatenate((train_u[:, :, :, 0], train_u[:, :, :, 1]), axis=1)
        y_all = torch.from_numpy(train_u_2).to(torch.float)
        y_train.append(y_all[:ntrain_per_task])
        f_train.append(f_all[:ntrain_per_task])

    f_train_mixed = torch.cat(f_train, dim=0).unsqueeze(-1)
    y_train_mixed = torch.cat(y_train, dim=0)
    return f_train_mixed, y_train_mixed


def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
    return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def main(random_seed, learning_rate, gamma, wd, learning_rate_pt, gamma_pt, wd_pt):
    t0 = default_timer()

    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)

    ################################################################
    # configs
    ################################################################
    normalizer = 'UnitGaussianNormalizer'

    modes = 8
    width = 32
    layer_end = 4
    in_width = 3
    out_width = 2

    s = 41
    train_task_num = 200
    valid_task_num = 10
    test_task_num = 10
    task_num = train_task_num + valid_task_num + test_task_num
    ntrain_per_task = 40
    ntest_per_task = 10
    ntotal_per_task = ntrain_per_task + ntest_per_task

    npretrain = train_task_num * ntotal_per_task

    ntrain = valid_task_num * ntrain_per_task
    ntest = valid_task_num * ntest_per_task

    epochs = 500
    step_size = 100
    batch_size = 1 if ntrain_per_task < 10 else 5

    ################################################################
    # load and normalize data
    ################################################################
    data_dir = '../data/HGO_MINO/Neumann_indis_sensitive_analysis_balance_range_grf_alpha_2'
    param_filename = "HGO_neumann_indistribution_sensitive_analysis_balance_range_grf_alpha_2.txt"
    f_indis_mixed, y_indis_mixed = read_train_data(s,
                                                   task_num,
                                                   data_dir,
                                                   ntrain_per_task + ntest_per_task,
                                                   param_filename)

    # meta-train data is needed in order to have the same normalizers
    f_train_mixed, y_train_mixed = f_indis_mixed[:npretrain], y_indis_mixed[:npretrain]
    f_test_mixed, y_test_mixed = f_indis_mixed[-(ntrain + ntest):], y_indis_mixed[-(ntrain + ntest):]

    # normalize input
    f_normalizer = str_to_class(normalizer)(f_train_mixed)
    # f_train_mixed = f_normalizer.encode(f_train_mixed)
    f_test_mixed = f_normalizer.encode(f_test_mixed)

    # normalize u and v
    y_normalizer = str_to_class(normalizer)(y_train_mixed)
    # y_train_mixed = y_normalizer.encode(y_train_mixed)
    # y_test = y_normalizer.encode(y_test)

    print(f'>> In and out data shape: {f_test_mixed.shape} and {y_test_mixed.shape}')

    train_loader, test_loader = [], []
    for t in range(test_task_num):
        train_loader.append(torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                f_test_mixed[t * ntotal_per_task:t * ntotal_per_task + ntrain_per_task, ...],
                y_test_mixed[t * ntotal_per_task:t * ntotal_per_task + ntrain_per_task, ...]),
            batch_size=batch_size, shuffle=True))

        test_loader.append(torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                f_test_mixed[t * ntotal_per_task + ntrain_per_task:(t + 1) * ntotal_per_task, ...],
                y_test_mixed[t * ntotal_per_task + ntrain_per_task:(t + 1) * ntotal_per_task, ...]),
            batch_size=batch_size, shuffle=False))

    ################################################################
    # train and eval.
    ################################################################
    pretrain_model_dir = './metaNO_HGO_neumann_train_task_%d/shal2deep_ifno' % train_task_num
    base_dir = './metaNO_HGO_neumann_train_task_%d_valid/shal2deep_ifno' % train_task_num
    os.makedirs(base_dir, exist_ok=True)

    res_dir = './res_metaNO_HGO_neumann_valid_task_%d' % train_task_num
    os.makedirs(res_dir, exist_ok=True)
    res_file = "%s/metaNO_n%d_valid.txt" % (res_dir, ntrain)
    if not os.path.isfile(res_file):
        f = open(res_file, "w")
        f.write(f'ntrain, seed, lr, gamma, wd, lr_pt, gamma_pt, wd_pt, '
                f'train_lowest, train_best, test_best, best_epochs, time (hrs)\n')
        f.close()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> Device being used: {device}')
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')
        y_normalizer.cuda()

    myloss = LpLoss(size_average=False)

    nlayer = layer_end - 1
    nb = 2 ** nlayer
    model = MetaFNO2d(modes, modes, width, nb, task_num, in_width=in_width, out_width=out_width).to(device)

    model_filename = '%s/ifno_depth%d_plr%.1e_pgamma%.1f_pwd%.1e_lr%.1e_gamma%.1f_wd%.1e.ckpt' % (base_dir,
                                                                                                  nb,
                                                                                                  learning_rate_pt,
                                                                                                  gamma_pt,
                                                                                                  wd_pt,
                                                                                                  learning_rate,
                                                                                                  gamma,
                                                                                                  wd)
    pretrain_model_filename = '%s/ifno_depth%d_lr%.1e_gamma%.1f_wd%.1e.ckpt' % (pretrain_model_dir,
                                                                                nb,
                                                                                learning_rate_pt,
                                                                                gamma_pt,
                                                                                wd_pt)

    if not os.path.isfile(pretrain_model_filename):
        print('*** Pretrained meta-train model not found. Exiting..')
        return

    print(">> Loading pretrained meta-train model from %s" % pretrain_model_filename)
    pretrained_dict = torch.load(pretrain_model_filename, map_location=device)
    weight_mat = torch.zeros(train_task_num, *pretrained_dict['fc0.%d.weight' % train_task_num].shape)
    bias_mat = torch.zeros(train_task_num, *pretrained_dict['fc0.%d.bias' % train_task_num].shape)

    for i in range(train_task_num):
        weight_mat[i] = pretrained_dict['fc0.%d.weight' % i]
        bias_mat[i] = pretrained_dict['fc0.%d.bias' % i]
    tmp_weight = torch.mean(weight_mat, dim=0)
    tmp_bias = torch.mean(bias_mat, dim=0)

    # initialize lifting layer using mean weights and bias from pretrained model
    for i in range(test_task_num):
        pretrained_dict['fc0.%d.weight' % (train_task_num + i)] = tmp_weight
        pretrained_dict['fc0.%d.bias' % (train_task_num + i)] = tmp_bias
    model.load_state_dict(pretrained_dict)

    # only train lifting layer and freeze others
    all_lifting_parameters = []
    for i in range(test_task_num):
        all_lifting_parameters += list(model.fc0[train_task_num + i].parameters())
    optimizer = torch.optim.Adam(all_lifting_parameters, lr=learning_rate, weight_decay=wd)

    c = 0
    for p in all_lifting_parameters:
        c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))
    print(f'>> Total number of model params: {count_params(model)} ({c} trainable)')

    best_epoch = 0
    train_loss_best = train_loss_lowest = test_loss_best = 1e8
    for ep in range(epochs):
        optimizer = scheduler(optimizer, LR_schedule(learning_rate, ep, step_size, gamma))
        model.train()
        t1 = default_timer()
        train_l2 = 0.0
        for minibatch in zip(*train_loader):
            loss = 0.0
            optimizer.zero_grad()
            for task_idx, (x, y) in enumerate(minibatch):
                x, y = x.to(device), y.to(device)
                this_batch_size = x.shape[0]
                out = model(x, train_task_num + task_idx).reshape(this_batch_size, 2 * s, s)
                out = y_normalizer.decode(out)

                loss += myloss(out.view(this_batch_size, -1), y.view(this_batch_size, -1))
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        train_l2 /= ntrain
        if train_l2 < train_loss_lowest:
            train_loss_lowest = train_l2

        if train_l2 < train_loss_best:
            model.eval()
            test_l2 = 0.0
            with torch.no_grad():
                for minibatch in zip(*test_loader):
                    for task_idx, (x, y) in enumerate(minibatch):
                        x, y = x.to(device), y.to(device)
                        this_batch_size = x.shape[0]
                        out = model(x, train_task_num + task_idx).reshape(this_batch_size, 2 * s, s)
                        out = y_normalizer.decode(out)

                        test_l2 += myloss(out.view(this_batch_size, -1), y.view(this_batch_size, -1))
            test_l2 /= ntest
            if test_l2 < test_loss_best:
                best_epoch = ep
                train_loss_best = train_l2
                test_loss_best = test_l2
                torch.save(model.state_dict(), model_filename)

                t2 = default_timer()
                print(f'>> depth{nb}, epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                      f'runtime: {(t2 - t1):.2f}s, train err: {train_l2:.4f}, test err: {test_l2:.4f}')
            else:
                t2 = default_timer()
                print(f'>> depth{nb}, epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                      f'runtime: {(t2 - t1):.2f}s, train err: {train_l2:.4f} (best: [{best_epoch + 1}], '
                      f'{train_loss_best:.4f}/{test_loss_best:.4f})')
        else:
            t2 = default_timer()
            print(f'>> depth{nb}, epoch [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                  f'runtime: {(t2 - t1):.2f}s, train err: {train_l2:.4f} (best: [{best_epoch + 1}], '
                  f'{train_loss_best:.4f}/{test_loss_best:.4f})')

    t3 = default_timer()

    f = open(res_file, "a")
    f.write(f'{ntrain}, {iseed}, {learning_rate}, {gamma}, {wd}, {learning_rate_pt}, '
            f'{gamma_pt}, {wd_pt}, {train_loss_lowest}, {train_loss_best}, {test_loss_best}, '
            f'{best_epoch}, {(t3 - t0) / 3600:.2f}\n')
    f.close()


if __name__ == '__main__':
    # hyperparameters to tune for lifting layer training
    # 3 * 3 * 8 = 24 * 3
    lrs = [3e-2, 1e-2, 3e-3]
    gammas = [0.5, 0.7, 0.9]
    wds = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7]

    # hyperparameters used in meta-train phase
    # 3 * 3 * 8 = 72
    lrs_pt = [3e-2, 1e-2, 3e-3]
    gammas_pt = [0.5, 0.7, 0.9]
    wds_pt = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7]

    if len(sys.argv) > 2:
        lrs = [lrs[int(sys.argv[1])]]
        wds = [wds[int(sys.argv[2])]]

    seeds = [0]

    icount = 0
    case_total = len(seeds) * len(lrs) * len(gammas) * len(wds) * len(lrs_pt) * len(gammas_pt) * len(wds_pt)
    for iseed in seeds:
        for lr in lrs:
            for gamma in gammas:
                for wd in wds:
                    for lr_pt in lrs_pt:
                        for gamma_pt in gammas_pt:
                            for wd_pt in wds_pt:
                                icount += 1
                                print("-" * 100)
                                print(f'>> running case {icount}/{case_total}: lr={lr}, gamma={gamma}, wd={wd}')
                                print("-" * 100)
                                main(iseed, lr, gamma, wd, lr_pt, gamma_pt, wd_pt)

    print(f'********** Training completed! **********')
