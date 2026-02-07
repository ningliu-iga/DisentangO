import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.fft
# import functorch

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

        self.fc1 = nn.Linear(self.width, 4 * self.width)
        self.fc2 = nn.Linear(4 * self.width, out_width)

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
        return x.permute(0, 3, 1, 2).reshape(batch_size, -1, size_y)

    def get_grid(self, shape):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.linspace(0, 1, size_y)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1)


################################################################
# vae model
################################################################
class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim, in_dim=128, actv_fn=nn.GELU()):
        super().__init__()
        self.latent_dim = latent_dim
        for i in range(latent_dim):
            self.add_module('encoder_%d' % i, nn.Sequential(nn.Linear(in_dim, in_dim * 4),
                                                            actv_fn,
                                                            nn.Linear(in_dim * 4, 1)))

        # NL: why PRelu? weight decay should not be used when learning alpha for good performance.
        # try leaky relu and gelu
        # self.activation = nn.PReLU()

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        mus = [self._modules['encoder_%d' % i](x) for i in range(self.latent_dim)]
        return torch.cat(mus, dim=1)


class Decoder(nn.Module):
    def __init__(self, latent_dim=5, out_dim=128):
        super(Decoder, self).__init__()
        self.out_dim = out_dim
        self.linear1 = nn.Linear(latent_dim, out_dim * 4)
        self.linear2 = nn.Linear(out_dim * 4, out_dim)

    def forward(self, z):
        z = F.gelu(self.linear1(z))
        z = self.linear2(z)
        return z.reshape(-1, self.out_dim)


class VAE(nn.Module):
    def __init__(self, latent_dim, in_dim=128):
        super(VAE, self).__init__()
        self.in_dim = in_dim
        self.encoder = VariationalEncoder(latent_dim, in_dim=in_dim)
        self.decoder = Decoder(latent_dim)

    def forward(self, x):
        x = x.view(x.shape[0], self.in_dim)
        z = self.encoder(x)
        return self.decoder(z), z


class DisentanglementNO(nn.Module):
    def __init__(self, latent_dim, modes1, modes2, width, nlayer, task_num, in_width=3, out_width=2,
                 l_normalizer=None):
        super().__init__()
        self.in_width = in_width
        self.width = width
        self.MetaNO = MetaFNO2d(modes1, modes2, width, nlayer, task_num, in_width=in_width, out_width=out_width)
        self.vae = VAE(latent_dim)
        self.l_normalizer = l_normalizer

    def forward(self, x, task_idx):
        # get lifting layer parameters as vae input
        lifting_params = torch.cat((self.MetaNO.fc0[task_idx].weight.reshape(-1),
                                    self.MetaNO.fc0[task_idx].bias), dim=0).unsqueeze(0)

        # normalize lifting layer parameters
        lifting_params_encoded = self.l_normalizer.encode(lifting_params)
        lifting_params_recons, z = self.vae(lifting_params_encoded)
        lifting_params_recons = self.l_normalizer.decode(lifting_params_recons)

        # reconstruct data
        lifting_weight_recons = lifting_params_recons[0, :self.width * self.in_width].reshape(self.width, self.in_width)
        lifting_bias_recons = lifting_params_recons[0, self.width * self.in_width:]

        batch_size, size_x, size_y = x.shape[:3]

        grids = self.MetaNO.get_grid(x.shape).to(x.device)
        x = torch.cat((x, grids), dim=-1)

        x = torch.matmul(x, lifting_weight_recons.T) + lifting_bias_recons
        x = x.permute(0, 3, 1, 2)

        for _ in range(self.MetaNO.nlayer - 1):
            x1 = self.MetaNO.convlayer(x)
            x2 = self.MetaNO.w(x)
            x = F.gelu(x1 + x2) / self.MetaNO.nlayer + x

        x1 = self.MetaNO.convlayer(x)
        x2 = self.MetaNO.w(x)
        x = (x1 + x2) / self.MetaNO.nlayer + x

        x = x.permute(0, 2, 3, 1)
        x = self.MetaNO.fc1(x)
        x = F.gelu(x)
        x = self.MetaNO.fc2(x)
        x = x.permute(0, 3, 1, 2).reshape(batch_size, -1, size_y)

        return x, z, lifting_params_recons, lifting_params


def read_train_data(s, task_num, data_dir, ntrain_per_task, param_filename):
    mat_params = np.loadtxt(data_dir + '/' + param_filename)[:task_num]
    mat_params_idx = mat_params[:, 0]
    mat_params = mat_params[:, 1:]
    # input material parameters [E, nv, K1/K2, K2, alpha]
    mat_params[:, 2] = mat_params[:, 2] / mat_params[:, 3]
    mat_params = torch.from_numpy(mat_params)

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
    return f_train_mixed, y_train_mixed, mat_params


def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
    return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def main(iseed, learning_rate, gamma, wd):
    t0 = default_timer()

    torch.manual_seed(iseed)
    torch.cuda.manual_seed(iseed)
    np.random.seed(iseed)

    ################################################################
    # configs
    ################################################################
    normalizer = 'UnitGaussianNormalizer'
    # model_dir_suffix = '' if normalizer == 'UnitGaussianNormalizer' else '_gaussian'

    # select pretrained MetaNO models by meta-train and meta-test hyperparameters
    if normalizer == 'UnitGaussianNormalizer':
        learning_rate_pt, gamma_pt, wd_pt = 3e-2, 0.7, 1e-6
        learning_rate_valid, gamma_valid, wd_valid = 3e-2, 0.5, 1e-4
        model_dir_suffix = ''
    else:
        # GaussianNormalizer
        learning_rate_pt, gamma_pt, wd_pt = 1e-2, 0.7, 3e-6
        learning_rate_valid, gamma_valid, wd_valid = 1e-2, 0.5, 1e-6
        model_dir_suffix = '_gaussian'

    modes = 8
    width = 32
    layer_end = 4
    in_width = 3
    out_width = 2
    latent_dim = 5

    s = 41
    train_task_num = 200
    valid_task_num = 10
    test_task_num = 10
    task_num = train_task_num + valid_task_num + test_task_num

    ntrain_per_task = 50

    ntrain = train_task_num * ntrain_per_task
    nvalid = valid_task_num * ntrain_per_task
    ntest = test_task_num * ntrain_per_task

    epochs = 500
    step_size = 100
    batch_size = 1 if ntrain_per_task < 10 else 5

    ################################################################
    # load and normalize data
    ################################################################
    data_dir = '../data/HGO_MINO/Neumann_indis_sensitive_analysis_balance_range_grf_alpha_2'
    param_filename = "HGO_neumann_indistribution_sensitive_analysis_balance_range_grf_alpha_2.txt"
    f_indis_mixed, y_indis_mixed, mat_params = read_train_data(s,
                                                               task_num,
                                                               data_dir,
                                                               ntrain_per_task,
                                                               param_filename)

    validate_assumption4 = True
    if validate_assumption4:
        vs = []
        for i in range(2 * latent_dim):
            vi = torch.zeros(1, 2 * latent_dim)
            vi[0, :latent_dim] = mat_params[0] - mat_params[i + 1]
            vs.append(vi)
        vs = torch.cat(vs, dim=0)
        rank_vs = torch.linalg.matrix_rank(vs)
        print(f'>> The computed matrix rank is {rank_vs.item()}')

    # meta-train data is needed in order to have the same normalizers
    f_train_mixed = f_indis_mixed[:ntrain]
    y_train_mixed = y_indis_mixed[:ntrain]
    mat_params_train = mat_params[:train_task_num]

    f_valid_mixed = f_indis_mixed[-nvalid:]
    y_valid_mixed = y_indis_mixed[-nvalid:]
    mat_params_valid = mat_params[-valid_task_num:]

    f_test_mixed = f_indis_mixed[-ntest - nvalid:-nvalid]
    y_test_mixed = y_indis_mixed[-ntest - nvalid:-nvalid]
    mat_params_test = mat_params[-test_task_num - valid_task_num:-valid_task_num]

    # normalize input
    f_normalizer = str_to_class(normalizer)(f_train_mixed)
    f_train_mixed = f_normalizer.encode(f_train_mixed)
    f_valid_mixed = f_normalizer.encode(f_valid_mixed)
    f_test_mixed = f_normalizer.encode(f_test_mixed)

    # normalize u and v
    y_normalizer = str_to_class(normalizer)(y_train_mixed)

    print(f'>> train input, output, mat param data shape: '
          f'{f_train_mixed.shape}, {y_train_mixed.shape}, {mat_params_train.shape}')
    print(f'>> test input, output, mat param data shape: '
          f'{f_test_mixed.shape}, {y_test_mixed.shape}, {mat_params_test.shape}')

    train_loader = []
    for t in range(train_task_num):
        train_loader.append(torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(f_train_mixed[t * ntrain_per_task:(t + 1) * ntrain_per_task, :],
                                           y_train_mixed[t * ntrain_per_task:(t + 1) * ntrain_per_task, :]),
            batch_size=batch_size, shuffle=True))

    valid_loader = []
    for t in range(valid_task_num):
        valid_loader.append(torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(f_valid_mixed[t * ntrain_per_task:(t + 1) * ntrain_per_task, :],
                                           y_valid_mixed[t * ntrain_per_task:(t + 1) * ntrain_per_task, :]),
            batch_size=batch_size, shuffle=False))

    test_loader = []
    for t in range(test_task_num):
        test_loader.append(torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(f_test_mixed[t * ntrain_per_task:(t + 1) * ntrain_per_task, :],
                                           y_test_mixed[t * ntrain_per_task:(t + 1) * ntrain_per_task, :]),
            batch_size=batch_size, shuffle=False))

    ################################################################
    # train and eval.
    ################################################################
    pretrain_model_dir = './metaNO_HGO_neumann_train_task_%d_test%s/shal2deep_ifno' % (train_task_num,
                                                                                       model_dir_suffix)
    base_dir = './DNO_freeze_MetaNO%s' % model_dir_suffix
    os.makedirs(base_dir, exist_ok=True)

    res_dir = './res_DNO_freeze_MetaNO%s' % model_dir_suffix
    os.makedirs(res_dir, exist_ok=True)
    res_file = "%s/DNO_HGO_n%d%s.txt" % (res_dir, ntrain, model_dir_suffix)
    if not os.path.isfile(res_file):
        f = open(res_file, "w")
        f.write(f'ntrain, seed, lr, gamma, wd, train_lowest, train_best, '
                f'train_abs_data, train_abs_recons, train_abs_z, train_l2_data, train_l2_recons, train_l2_z, '
                f'valid_best, valid_abs_data, valid_abs_recons, valid_abs_z, '
                f'valid_l2_data, valid_l2_recons, valid_mse_z, '
                f'valid_mse_z1, valid_mse_z2, valid_mse_z3, valid_mse_z4, valid_mse_z5, '
                f'test, test_abs_data, test_abs_recons, test_abs_z, test_l2_data, test_l2_recons, test_mse_z, '
                f'test_mse_z1, test_mse_z2, test_mse_z3, test_mse_z4, test_mse_z5, best_epoch, time (hrs)\n')
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
    model = DisentanglementNO(latent_dim, modes, modes, width, nb, task_num, in_width=in_width,
                              out_width=out_width).to(device)

    model_filename = '%s/DNO_lr%.1e_gamma%.1f_wd%.1e.ckpt' % (base_dir, learning_rate, gamma, wd)

    pretrained_MetaNO_filename = '%s/ifno_depth%d_plr%.1e_pgamma%.1f_pwd%.1e_lr%.1e_gamma%.1f_wd%.1e.ckpt' % (
        pretrain_model_dir, nb, learning_rate_pt, gamma_pt, wd_pt, learning_rate_valid, gamma_valid, wd_valid)

    print(f'Loading pretrained MetaNO model from: {pretrained_MetaNO_filename}')
    if not os.path.isfile(pretrained_MetaNO_filename):
        print('*** WARNING: pretrained MetaNO model not found. Exiting..')
        return

    # load pretrained MetaNO model
    pretrained_MetaNO = torch.load(pretrained_MetaNO_filename, map_location=device)
    state0 = model.MetaNO.state_dict()
    state0.update(pretrained_MetaNO)
    model.MetaNO.load_state_dict(state0)

    # normalize lifting layer parameters
    lifting_params_train = torch.cat([torch.cat((state0['fc0.' + str(task_idx) + '.weight'].reshape(-1),
                                                 state0['fc0.' + str(task_idx) + '.bias']), dim=0).unsqueeze(0)
                                      for task_idx in range(train_task_num)], dim=0)

    l_normalizer = str_to_class(normalizer)(lifting_params_train)

    # normalize material parameters
    m_normalizer = str_to_class(normalizer)(mat_params_train)

    if torch.cuda.is_available():
        l_normalizer.cuda()
        m_normalizer.cuda()

    model.l_normalizer = l_normalizer

    # only train VAE and freeze others
    optimizer = torch.optim.Adam(model.vae.parameters(), lr=learning_rate, weight_decay=wd)

    print(f'>> Total number of model params: {count_params(model)} ({count_params(model.vae)} trainable)')

    best_epoch = 0
    train_loss_best = train_loss_lowest = valid_loss_best = 1e8
    train_l2_data_best = train_l2_recons_best = train_l2_latent_z_best = 1e8
    valid_l2_data_best = valid_l2_recons_best = valid_l2_latent_z_best = 1e8
    train_abs_data_best = train_abs_recons_best = train_abs_latent_z_best = 1e8
    valid_abs_data_best = valid_abs_recons_best = valid_abs_latent_z_best = 1e8
    valid_mse_z_best = []
    for ep in range(epochs):
        optimizer = scheduler(optimizer, LR_schedule(learning_rate, ep, step_size, gamma))
        model.train()
        t1 = default_timer()
        train_l2 = 0.0
        train_l2_data = train_l2_recons = train_l2_latent_z = 0.0
        train_abs_data = train_abs_recons = train_abs_latent_z = 0.0
        for minibatch in zip(*train_loader):
            loss = 0.0
            optimizer.zero_grad()
            for task_idx, (x, y) in enumerate(minibatch):
                x, y, m = x.to(device), y.to(device), mat_params_train[task_idx].to(device)
                this_batch_size = x.shape[0]
                out, z, l_recons, l_gt = model(x, task_idx)
                out = y_normalizer.decode(out.reshape(this_batch_size, 2 * s, s))
                z = m_normalizer.decode(z)

                loss_data = torch.linalg.norm(out.view(this_batch_size, -1) - y.view(this_batch_size, -1)) ** 2
                loss_recons = this_batch_size * torch.linalg.norm(l_recons.view(1, -1) - l_gt.view(1, -1)) ** 2
                loss_latent_z = this_batch_size * torch.linalg.norm(z.view(1, -1) - m.view(1, -1)) ** 2

                loss_l2_data = myloss(out.view(this_batch_size, -1), y.view(this_batch_size, -1))
                loss_l2_recons = this_batch_size * myloss(l_recons.view(1, -1), l_gt.view(1, -1))
                loss_l2_latent_z = this_batch_size * sum([myloss(z[0, i_ld].view(1, -1), m[i_ld].view(1, -1))
                                                          for i_ld in range(latent_dim)]) / latent_dim

                train_l2_data += loss_l2_data.item()
                train_l2_recons += loss_l2_recons.item()
                train_l2_latent_z += loss_l2_latent_z.item()

                train_abs_data += loss_data.item()
                train_abs_recons += loss_recons.item()
                train_abs_latent_z += loss_latent_z.item()
                loss += loss_data + loss_recons + loss_l2_latent_z
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        train_abs_data /= ntrain
        train_abs_recons /= ntrain
        train_abs_latent_z /= ntrain
        train_l2_data /= ntrain
        train_l2_recons /= ntrain
        train_l2_latent_z /= ntrain
        train_l2 /= ntrain
        if train_l2 < train_loss_lowest:
            train_loss_lowest = train_l2

        if train_l2 < train_loss_best:
            model.eval()
            valid_l2 = 0.0
            valid_l2_data = valid_l2_recons = valid_l2_latent_z = 0.0
            valid_abs_data = valid_abs_recons = valid_abs_latent_z = 0.0
            valid_mse_z = [0.0] * latent_dim
            with torch.no_grad():
                for minibatch in zip(*valid_loader):
                    for task_idx, (x, y) in enumerate(minibatch):
                        x, y, m = x.to(device), y.to(device), mat_params_valid[task_idx].to(device)
                        this_batch_size = x.shape[0]
                        out, z, l_recons, l_gt = model(x, train_task_num + task_idx)
                        out = y_normalizer.decode(out.reshape(this_batch_size, 2 * s, s))
                        z = m_normalizer.decode(z)

                        loss_data = torch.linalg.norm(out.view(this_batch_size, -1) - y.view(this_batch_size, -1)) ** 2
                        loss_recons = this_batch_size * torch.linalg.norm(l_recons.view(1, -1) - l_gt.view(1, -1)) ** 2
                        loss_latent_z = this_batch_size * torch.linalg.norm(z.view(1, -1) - m.view(1, -1)) ** 2

                        loss_l2_data = myloss(out.view(this_batch_size, -1), y.view(this_batch_size, -1))
                        loss_l2_recons = this_batch_size * myloss(l_recons.view(1, -1), l_gt.view(1, -1))
                        loss_l2_latent_z = this_batch_size * sum([myloss(z[0, i_ld].view(1, -1), m[i_ld].view(1, -1))
                                                                  for i_ld in range(latent_dim)]) / latent_dim

                        for i_ld in range(latent_dim):
                            valid_mse_z[i_ld] += this_batch_size * myloss(z[0, i_ld].view(1, -1),
                                                                          m[i_ld].view(1, -1)).item()

                        valid_l2_data += loss_l2_data.item()
                        valid_l2_recons += loss_l2_recons.item()
                        valid_l2_latent_z += loss_l2_latent_z.item()

                        valid_abs_data += loss_data.item()
                        valid_abs_recons += loss_recons.item()
                        valid_abs_latent_z += loss_latent_z.item()
                        valid_l2 += loss_data.item() + loss_recons.item() + loss_l2_latent_z.item()
            valid_l2_data /= nvalid
            valid_l2_recons /= nvalid
            valid_l2_latent_z /= nvalid

            valid_abs_data /= nvalid
            valid_abs_recons /= nvalid
            valid_abs_latent_z /= nvalid
            valid_l2 /= nvalid
            valid_mse_z = [i_elem / nvalid for i_elem in valid_mse_z]
            if valid_l2 < valid_loss_best:
                best_epoch = ep
                train_loss_best = train_l2
                train_l2_data_best = train_l2_data
                train_l2_recons_best = train_l2_recons
                train_l2_latent_z_best = train_l2_latent_z
                train_abs_data_best = train_abs_data
                train_abs_recons_best = train_abs_recons
                train_abs_latent_z_best = train_abs_latent_z

                valid_loss_best = valid_l2
                valid_l2_data_best = valid_l2_data
                valid_l2_recons_best = valid_l2_recons
                valid_l2_latent_z_best = valid_l2_latent_z
                valid_abs_data_best = valid_abs_data
                valid_abs_recons_best = valid_abs_recons
                valid_abs_latent_z_best = valid_abs_latent_z
                valid_mse_z_best = valid_mse_z

                torch.save(model.state_dict(), model_filename)

                t2 = default_timer()
                print(f'>> ep [{(ep + 1): >{len(str(epochs))}d}/{epochs}], runtime: {(t2 - t1):.1f}s, '
                      f'train err: {train_l2:.4f} '
                      f'(dlz: {train_abs_data:.3f}/{train_abs_recons:.3f}/{train_l2_latent_z:.3f}), '
                      f'valid err: {valid_l2:.4f} '
                      f'({valid_abs_data:.3f}/{valid_abs_recons:.3f}/{valid_l2_latent_z:.3f})')
                print(f'*** rel. l2 loss for dlz: '
                      f'train ({train_l2_data:.3f}/{train_l2_recons:.3f}/{train_l2_latent_z:.3f}), '
                      f'valid ({valid_l2_data:.3f}/{valid_l2_recons:.3f}/{valid_l2_latent_z:.3f}) ***')
                print(f'*** Individual MSE of validation z: {valid_mse_z_best} ***')
            else:
                t2 = default_timer()
                print(f'>> ep [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                      f'runtime: {(t2 - t1):.1f}s, train err: {train_l2:.4f} (best: [{best_epoch + 1}], '
                      f'{train_loss_best:.4f}/{valid_loss_best:.4f})')
        else:
            t2 = default_timer()
            print(f'>> ep [{(ep + 1): >{len(str(epochs))}d}/{epochs}], '
                  f'runtime: {(t2 - t1):.1f}s, train err: {train_l2:.4f} (best: [{best_epoch + 1}], '
                  f'{train_loss_best:.4f}/{valid_loss_best:.4f})')

    # test
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
    test_l2 = 0.0
    test_l2_data = test_l2_recons = test_l2_latent_z = 0.0
    test_abs_data = test_abs_recons = test_abs_latent_z = 0.0
    test_mse_z = [0.0] * latent_dim
    with torch.no_grad():
        for minibatch in zip(*test_loader):
            for task_idx, (x, y) in enumerate(minibatch):
                x, y, m = x.to(device), y.to(device), mat_params_test[task_idx].to(device)
                this_batch_size = x.shape[0]
                out, z, l_recons, l_gt = model(x, train_task_num + valid_task_num + task_idx)
                out = y_normalizer.decode(out.reshape(this_batch_size, 2 * s, s))
                z = m_normalizer.decode(z)

                loss_data = torch.linalg.norm(out.view(this_batch_size, -1) - y.view(this_batch_size, -1)) ** 2
                loss_recons = this_batch_size * torch.linalg.norm(l_recons.view(1, -1) - l_gt.view(1, -1)) ** 2
                loss_latent_z = this_batch_size * torch.linalg.norm(z.view(1, -1) - m.view(1, -1)) ** 2

                loss_l2_data = myloss(out.view(this_batch_size, -1), y.view(this_batch_size, -1))
                loss_l2_recons = this_batch_size * myloss(l_recons.view(1, -1), l_gt.view(1, -1))
                loss_l2_latent_z = this_batch_size * sum([myloss(z[0, i_ld].view(1, -1), m[i_ld].view(1, -1))
                                                          for i_ld in range(latent_dim)]) / latent_dim

                for i_ld in range(latent_dim):
                    test_mse_z[i_ld] += this_batch_size * myloss(z[0, i_ld].view(1, -1), m[i_ld].view(1, -1)).item()

                test_l2_data += loss_l2_data.item()
                test_l2_recons += loss_l2_recons.item()
                test_l2_latent_z += loss_l2_latent_z.item()

                test_abs_data += loss_data.item()
                test_abs_recons += loss_recons.item()
                test_abs_latent_z += loss_latent_z.item()
                test_l2 += loss_data.item() + loss_recons.item() + loss_l2_latent_z.item()

    test_l2_data /= ntest
    test_l2_recons /= ntest
    test_l2_latent_z /= ntest
    test_abs_data /= ntest
    test_abs_recons /= ntest
    test_abs_latent_z /= ntest
    test_l2 /= ntest
    test_mse_z = [i_elem / ntest for i_elem in test_mse_z]

    t3 = default_timer()

    f = open(res_file, "a")
    f.write(f'{ntrain}, {iseed}, {learning_rate}, {gamma}, {wd}, {train_loss_lowest}, {train_loss_best}, '
            f'{train_abs_data_best}, {train_abs_recons_best}, {train_abs_latent_z_best}, '
            f'{train_l2_data_best}, {train_l2_recons_best}, {train_l2_latent_z_best}, '
            f'{valid_loss_best}, {valid_abs_data_best}, {valid_abs_recons_best}, {valid_abs_latent_z_best}, '
            f'{valid_l2_data_best}, {valid_l2_recons_best}, {valid_l2_latent_z_best}, {valid_mse_z_best}, '
            f'{test_l2}, {test_abs_data}, {test_abs_recons}, {test_abs_latent_z}, '
            f'{test_l2_data}, {test_l2_recons}, {test_l2_latent_z}, {test_mse_z}, '
            f'{best_epoch}, {(t3 - t0) / 3600:.2f}\n')
    f.close()


if __name__ == '__main__':
    # 3 * 3 * 8 = 24 * 3
    lrs = [1e-2, 3e-3, 3e-2]
    gammas = [0.9, 0.7, 0.5]
    wds = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 1e-7]

    if len(sys.argv) > 2:
        lrs = [lrs[int(sys.argv[1])]]
        wds = [wds[int(sys.argv[2])]]

    seeds = [0]

    icount = 0
    case_total = len(seeds) * len(lrs) * len(gammas) * len(wds)
    for iseed in seeds:
        for lr in lrs:
            for gamma in gammas:
                for wd in wds:
                    icount += 1
                    print("-" * 100)
                    print(f'>> running case {icount}/{case_total}: lr={lr}, gamma={gamma}, wd={wd}')
                    print("-" * 100)
                    main(iseed, lr, gamma, wd)

    print(f'********** Training completed! **********')
