import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.fft
from timeit import default_timer
from utilities3 import *
import os
import sys

# from disent.metrics import metric_unsupervised
# from disent.dataset import DisentDataset
# from sklearn.metrics import *
import matplotlib.pyplot as plt


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
    def __init__(self, modes1, modes2, width, nlayer, task_num, in_width=4, out_width=2):
        super(MetaFNO2d, self).__init__()

        """
        The overall network. It contains n layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0.
        2. n layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=in_width)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=out_width)
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
    def __init__(self, latent_dim, nparam, actv_fn=nn.GELU()):
        super().__init__()
        self.latent_dim = latent_dim
        for i in range(latent_dim):
            self.add_module('encoder_%d' % i, nn.Sequential(nn.Linear(nparam, nparam * 4),
                                                            actv_fn,
                                                            nn.Linear(nparam * 4, 1)))

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        mus = [self._modules['encoder_%d' % i](x) for i in range(self.latent_dim)]
        return torch.cat(mus, dim=1)


class Decoder(nn.Module):
    def __init__(self, latent_dim, nparam, actv_fn=nn.GELU()):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, nparam * 4)
        self.fc2 = nn.Linear(nparam * 4, nparam)
        self.actv_fn = actv_fn
        self.nparam = nparam

    def forward(self, z):
        z = self.actv_fn(self.fc1(z))
        z = self.fc2(z)
        return z.reshape(-1, self.nparam)


class VAE(nn.Module):
    def __init__(self, latent_dim, nparam, class_dim=10):
        super().__init__()
        self.nparam = nparam
        self.encoder = VariationalEncoder(latent_dim, nparam)
        self.decoder = Decoder(latent_dim, nparam)
        self.linear_classifier = nn.Linear(latent_dim, class_dim)

    def forward(self, x):
        x = x.view(x.shape[0], self.nparam)
        z = self.encoder(x)
        return z, self.decoder(z), self.linear_classifier(z)


class DisentanglementNO(nn.Module):
    def __init__(self, nparam, latent_dim, modes1, modes2, width, nlayer, task_num, in_width=4, out_width=2,
                 l_normalizer=None):
        super().__init__()
        self.in_width = in_width
        self.width = width
        self.MetaNO = MetaFNO2d(modes1, modes2, width, nlayer, task_num, in_width=in_width, out_width=out_width)
        self.vae = VAE(latent_dim, nparam)
        self.l_normalizer = l_normalizer

    def forward(self, x, task_idx):
        # get lifting layer parameters as vae input
        lifting_params = torch.cat((self.MetaNO.fc0[task_idx].weight.reshape(-1),
                                    self.MetaNO.fc0[task_idx].bias), dim=0).unsqueeze(0)

        # normalize lifting layer parameters
        lifting_params_encoded = self.l_normalizer.encode(lifting_params)
        z, lifting_params_recons, classify_logits = self.vae(lifting_params_encoded)
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

        return x, z, classify_logits, lifting_params_recons, lifting_params


def accuracy(out, labels):
    _, pred = torch.max(out, dim=1)
    return torch.sum(pred == labels).item()


def scheduler(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def LR_schedule(learning_rate, steps, scheduler_step, scheduler_gamma):
    return learning_rate * np.power(scheduler_gamma, (steps // scheduler_step))


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def main(iseed, learning_rate, gamma, wd, latent_dim, cls_loss_coeff):
    t0 = default_timer()

    torch.manual_seed(iseed)
    torch.cuda.manual_seed(iseed)
    np.random.seed(iseed)

    ################################################################
    # configs
    ################################################################
    normalizer = 'UnitGaussianNormalizer'
    lifting_normalizer = 'GaussianNormalizer'

    # select pretrained MetaNO models by meta-train and meta-test hyperparameters
    if normalizer == 'UnitGaussianNormalizer':
        learning_rate_pt, gamma_pt, wd_pt = 6e-2, 0.5, 1e-3
        learning_rate_valid, gamma_valid, wd_valid = 1e-2, 0.5, 1e-7
        model_dir_suffix = ''
    else:
        # GaussianNormalizer
        learning_rate_pt, gamma_pt, wd_pt = 1e-2, 0.7, 3e-6
        learning_rate_valid, gamma_valid, wd_valid = 1e-2, 0.5, 1e-6
        model_dir_suffix = '_gaussian'

    modes = 8
    width = 32
    layer_end = 5
    in_width = 4
    out_width = 2
    nparam = width * 5  # nparam = (nparam_lift_weight + nparam_lift_bias) * 1 model for both u and v
    n_digits = 10

    S = 29
    train_task_num = 420
    valid_task_num = 40
    test_task_num = 40
    task_num_total = train_task_num + test_task_num + valid_task_num
    sample_per_task = 200

    ntrain = train_task_num * sample_per_task
    nvalid = valid_task_num * sample_per_task
    ntest = test_task_num * sample_per_task

    epochs = 500
    step_size = 100
    batch_size = 10

    ################################################################
    # load and normalize data
    ################################################################
    # data_dir = '../../NAO/Meta_pulse_example/src/models/sythntic_clean/DATA_u_b_chi_100000x29x29xd.mat'
    data_dir = '../../nao/DATA_u_b_chi_100000x29x29xd.mat'
    # data_dir = './data/DATA_u_b_chi_100000x29x29xd.mat'
    train_indexes = np.arange(0, 100000)
    train_list = list(train_indexes)[:ntrain]
    valid_list = list(train_indexes)[-(ntest + nvalid):-ntest]
    test_list = list(train_indexes)[-ntest:]

    indis_index = []
    valid_index = []
    test_index = []
    img_per_digit = 50
    train_images_per_digit = train_task_num // n_digits
    valid_images_per_digit = valid_task_num // n_digits
    test_images_per_digit = test_task_num // n_digits
    for i in range(n_digits):
        indis_index.extend(list(range(i * img_per_digit, i * img_per_digit + train_images_per_digit)))
        valid_index.extend(
            list(range(i * img_per_digit + train_images_per_digit, (i + 1) * img_per_digit - test_images_per_digit)))
        test_index.extend(list(range((i + 1) * img_per_digit - test_images_per_digit, (i + 1) * img_per_digit)))
    indis_index.extend(valid_index)
    indis_index.extend(test_index)

    class_labels = torch.zeros(task_num_total * sample_per_task, n_digits)
    for i in range(n_digits):
        # train
        idx_start = i * train_images_per_digit * sample_per_task
        idx_end = (i + 1) * train_images_per_digit * sample_per_task
        class_labels[idx_start: idx_end, i] = 1.0
        # valid
        idx_start = train_task_num * sample_per_task + i * valid_images_per_digit * sample_per_task
        idx_end = train_task_num * sample_per_task + (i + 1) * valid_images_per_digit * sample_per_task
        class_labels[idx_start: idx_end, i] = 1.0
        # test
        idx_start = (train_task_num + valid_task_num) * sample_per_task + i * test_images_per_digit * sample_per_task
        idx_end = (train_task_num + valid_task_num) * sample_per_task + (
                    i + 1) * test_images_per_digit * sample_per_task
        class_labels[idx_start: idx_end, i] = 1.0

    reader = MatReader(data_dir)
    sol_total = reader.read_field('u').view(
        task_num_total * sample_per_task, S, S, 2).reshape(
        task_num_total, sample_per_task, S, S, 2)[indis_index, ...].reshape(
        task_num_total * sample_per_task, S, S, 2)

    i_check_digits = 0
    if i_check_digits == 1:
        chi_total = reader.read_field('chi').view(
            task_num_total * sample_per_task, S, S, 1).reshape(
            task_num_total, sample_per_task, S, S, 1)[indis_index, ...].reshape(
            task_num_total * sample_per_task, S, S)

        fig, ax = plt.subplots(1, 3, figsize=(8, 8))
        interp_coeff = 'none'

        img0 = ax[0].imshow(torch.flip(chi_total[8400 * 4 + 200 * 33 + 3, :, :].T, [0]), interpolation=interp_coeff)
        fig.colorbar(img0, ax=ax[0], shrink=0.4)
        img1 = ax[1].imshow(torch.flip(chi_total[8400 * 7 + 200 * 33 + 3, :, :].T, [0]), interpolation=interp_coeff)
        fig.colorbar(img1, ax=ax[1], shrink=0.4)
        img2 = ax[2].imshow(torch.flip(chi_total[8400 * 9 + 200 * 33 + 3, :, :].T, [0]), interpolation=interp_coeff)
        fig.colorbar(img2, ax=ax[2], shrink=0.4)

        ax[0].title.set_text('0')
        ax[1].title.set_text('1')
        ax[2].title.set_text('2')

        plt.show()

    f_total = reader.read_field('b').view(
        task_num_total * sample_per_task, S, S, 2).reshape(
        task_num_total, sample_per_task, S, S, 2)[indis_index, ...].reshape(
        task_num_total * sample_per_task, S, S, 2)

    sol_train = sol_total[train_list, :].view(ntrain, S, S, 2).permute(0, 3, 1, 2).reshape(ntrain, -1, S)
    f_train = f_total[train_list, :].view(ntrain, S, S, 2)

    sol_valid = sol_total[valid_list, :].view(nvalid, S, S, 2).permute(0, 3, 1, 2).reshape(nvalid, -1, S)
    f_valid = f_total[valid_list, :].view(nvalid, S, S, 2)

    sol_test = sol_total[test_list, :].view(ntest, S, S, 2).permute(0, 3, 1, 2).reshape(ntest, -1, S)
    f_test = f_total[test_list, :].view(ntest, S, S, 2)

    # normalize input
    f_normalizer = str_to_class(normalizer)(f_train)
    f_train = f_normalizer.encode(f_train)
    f_valid = f_normalizer.encode(f_valid)
    f_test = f_normalizer.encode(f_test)

    # normalize u and v
    y_normalizer = str_to_class(normalizer)(sol_train)
    # sol_train = y_normalizer.encode(sol_train)
    # sol_valid = y_normalizer.encode(sol_valid)
    # sol_test = y_normalizer.encode(sol_test)

    print(f'>> x train and test data shape: {f_train.shape} and {f_test.shape}')
    print(f'>> y train and test data shape: {sol_train.shape} and {sol_test.shape}')

    train_loader = []
    for t in range(train_task_num):
        train_loader.append(torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(f_train[t * sample_per_task:(t + 1) * sample_per_task, :],
                                           sol_train[t * sample_per_task:(t + 1) * sample_per_task, :],
                                           class_labels[t * sample_per_task:(t + 1) * sample_per_task, :]),
            batch_size=batch_size, shuffle=True))

    valid_loader = []
    for t in range(valid_task_num):
        t_skip = train_task_num * sample_per_task
        valid_loader.append(torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(f_valid[t * sample_per_task:(t + 1) * sample_per_task, :],
                                           sol_valid[t * sample_per_task:(t + 1) * sample_per_task, :],
                                           class_labels[t_skip + t * sample_per_task:t_skip + (t + 1) * sample_per_task,
                                           :]),
            batch_size=batch_size, shuffle=False))

    test_loader = []
    for t in range(test_task_num):
        t_skip = (train_task_num + valid_task_num) * sample_per_task
        test_loader.append(torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(f_test[t * sample_per_task:(t + 1) * sample_per_task, :],
                                           sol_test[t * sample_per_task:(t + 1) * sample_per_task, :],
                                           class_labels[t_skip + t * sample_per_task:t_skip + (t + 1) * sample_per_task,
                                           :]),
            batch_size=batch_size, shuffle=False))

    ################################################################
    # train and eval.
    ################################################################
    pretrain_model_dir = './metaNO_MMNIST_train32_task_%d_test/shal2deep_ifno' % train_task_num
    base_dir = './DNO_MMNIST_n%d_l%d' % (ntrain, latent_dim)
    os.makedirs(base_dir, exist_ok=True)

    res_dir = './res_DNO_MMNIST%s' % model_dir_suffix
    os.makedirs(res_dir, exist_ok=True)
    res_file = "%s/DNO_MMNIST_n%d%s_l%d.txt" % (res_dir, ntrain, model_dir_suffix, latent_dim)
    if not os.path.isfile(res_file):
        f = open(res_file, "w")
        f.write(f'ntrain, seed, lr, gamma, wd, train_lowest, train_best, train_abs_data, train_abs_recons, train_kl, '
                f'train_classify, train_l2_data, train_l2_recons, valid_best, valid_abs_data, valid_abs_recons, '
                f'valid_kl, valid_classify, valid_l2_data, valid_l2_recons, '
                f'test, test_abs_data, test_abs_recons, test_kl, test_classify, test_l2_data, test_l2_recons, '
                f'best_epoch, time (hrs)\n')
        f.close()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print(f'>> Device being used: {device}')
    else:
        print(f'>> Device being used: {device} ({torch.cuda.get_device_name(0)})')
        y_normalizer.cuda()

    myloss = LpLoss(size_average=False)
    ce_loss = nn.CrossEntropyLoss()

    nlayer = layer_end - 1
    nb = 2 ** nlayer
    model = DisentanglementNO(nparam, latent_dim, modes, modes, width, nb, task_num_total, in_width=in_width,
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

    l_normalizer = str_to_class(lifting_normalizer)(lifting_params_train)

    if torch.cuda.is_available():
        l_normalizer.cuda()

    model.l_normalizer = l_normalizer

    # only train VAE and freeze others
    optimizer = torch.optim.Adam(model.vae.parameters(), lr=learning_rate, weight_decay=wd)

    print(f'>> Total number of model params: {count_params(model)} ({count_params(model.vae)} trainable)')

    best_epoch = 0
    train_loss_best = train_loss_lowest = valid_loss_best = 1e8
    train_abs_data_best = train_abs_recons_best = train_l2_data_best = train_l2_recons_best = train_kl_best = train_classify_best = 1e8
    valid_abs_data_best = valid_abs_recons_best = valid_l2_data_best = valid_l2_recons_best = valid_kl_best = valid_classify_best = 1e8
    for ep in range(epochs):
        optimizer = scheduler(optimizer, LR_schedule(learning_rate, ep, step_size, gamma))
        model.train()
        t1 = default_timer()
        train_l2 = 0.0
        train_abs_data = train_l2_data = train_abs_recons = train_l2_recons = train_kl = train_classify = 0.0
        for minibatch in zip(*train_loader):
            loss = 0.0
            optimizer.zero_grad()
            for task_idx, (x, y, label) in enumerate(minibatch):
                x, y, label = x.to(device), y.to(device), label[:1].to(device)
                this_batch_size = x.shape[0]
                out, z, classify_logits, l_recons, l_gt = model(x, task_idx)
                out = y_normalizer.decode(out.reshape(this_batch_size, 2 * S, S))

                loss_data = torch.linalg.norm(out.view(this_batch_size, -1) - y.view(this_batch_size, -1)) ** 2
                loss_recons = this_batch_size * torch.linalg.norm(l_recons.view(1, -1) - l_gt.view(1, -1)) ** 2
                loss_kl = this_batch_size * torch.linalg.norm(z) ** 2
                loss_classify = this_batch_size * ce_loss(classify_logits, label)

                train_l2_data += myloss(out.view(this_batch_size, -1), y.view(this_batch_size, -1)).item()
                train_l2_recons += this_batch_size * myloss(l_recons.view(1, -1), l_gt.view(1, -1)).item()
                train_abs_data += loss_data.item()
                train_abs_recons += loss_recons.item()
                train_kl += loss_kl.item()
                train_classify += loss_classify.item()

                loss += loss_data + loss_recons + loss_kl + cls_loss_coeff * loss_classify
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        train_l2_data /= ntrain
        train_l2_recons /= ntrain
        train_abs_data /= ntrain
        train_abs_recons /= ntrain
        train_kl /= ntrain
        train_classify /= ntrain
        train_l2 /= ntrain
        if train_l2 < train_loss_lowest:
            train_loss_lowest = train_l2

        if train_l2 < train_loss_best:
            model.eval()
            valid_l2 = 0.0
            valid_abs_data = valid_abs_recons = valid_l2_data = valid_l2_recons = valid_kl = valid_classify = 0.0
            with torch.no_grad():
                for minibatch in zip(*valid_loader):
                    for task_idx, (x, y, label) in enumerate(minibatch):
                        x, y, label = x.to(device), y.to(device), label[:1].to(device)
                        this_batch_size = x.shape[0]
                        out, z, classify_logits, l_recons, l_gt = model(x, train_task_num + task_idx)
                        out = y_normalizer.decode(out.reshape(this_batch_size, 2 * S, S))

                        loss_data = torch.linalg.norm(out.view(this_batch_size, -1) - y.view(this_batch_size, -1)) ** 2
                        loss_recons = this_batch_size * torch.linalg.norm(l_recons.view(1, -1) - l_gt.view(1, -1)) ** 2
                        loss_kl = this_batch_size * torch.linalg.norm(z) ** 2
                        loss_classify = this_batch_size * ce_loss(classify_logits, label)

                        valid_l2_data += myloss(out.view(this_batch_size, -1), y.view(this_batch_size, -1)).item()
                        valid_l2_recons += this_batch_size * myloss(l_recons.view(1, -1), l_gt.view(1, -1)).item()
                        valid_abs_data += loss_data.item()
                        valid_abs_recons += loss_recons.item()
                        valid_kl += loss_kl.item()
                        valid_classify += loss_classify.item()

                        valid_l2 += loss_data + loss_recons + loss_kl + cls_loss_coeff * loss_classify
            valid_l2_data /= nvalid
            valid_l2_recons /= nvalid
            valid_abs_data /= nvalid
            valid_abs_recons /= nvalid
            valid_kl /= nvalid
            valid_classify /= nvalid
            valid_l2 = valid_l2.item() / nvalid
            if valid_l2 < valid_loss_best:
                best_epoch = ep
                train_loss_best = train_l2
                train_l2_data_best = train_l2_data
                train_l2_recons_best = train_l2_recons
                train_abs_data_best = train_abs_data
                train_abs_recons_best = train_abs_recons
                train_kl_best = train_kl
                train_classify_best = train_classify

                valid_loss_best = valid_l2
                valid_l2_data_best = valid_l2_data
                valid_l2_recons_best = valid_l2_recons
                valid_abs_data_best = valid_abs_data
                valid_abs_recons_best = valid_abs_recons
                valid_kl_best = valid_kl
                valid_classify_best = valid_classify

                torch.save(model.state_dict(), model_filename)

                t2 = default_timer()
                print(f'>> ep [{(ep + 1): >{len(str(epochs))}d}/{epochs}], runtime: {(t2 - t1):.1f}s, '
                      f'train err: {train_l2:.4f} '
                      f'(drkc: {train_abs_data:.3f}/{train_abs_recons:.3f}/{train_kl:.3f}/{train_classify:.3f}), '
                      f'valid err: {valid_l2:.4f} '
                      f'({valid_abs_data:.3f}/{valid_abs_recons:.3f}/{valid_kl:.3f}/{valid_classify:.3f})')
                print(f'*** rel. l2 loss for data and lifting: train ({train_l2_data:.3f}/{train_l2_recons:.3f}), '
                      f'valid ({valid_l2_data:.3f}/{valid_l2_recons:.3f}) ***')
            else:
                t2 = default_timer()
                print(f'>> ep [{(ep + 1): >{len(str(epochs))}d}/{epochs}], runtime: {(t2 - t1):.1f}s, '
                      f'train err: {train_l2:.4f} (best: [{best_epoch + 1}], '
                      f'{train_loss_best:.4f}/{valid_loss_best:.4f})')
        else:
            t2 = default_timer()
            print(f'>> ep [{(ep + 1): >{len(str(epochs))}d}/{epochs}], runtime: {(t2 - t1):.1f}s, '
                  f'train err: {train_l2:.4f} (best: [{best_epoch + 1}], '
                  f'{train_loss_best:.4f}/{valid_loss_best:.4f})')

    # test
    model.load_state_dict(torch.load(model_filename, map_location=device))
    model.eval()
    test_l2 = 0.0
    test_abs_data = test_abs_recons = test_l2_data = test_l2_recons = test_kl = test_classify = 0.0
    with torch.no_grad():
        for minibatch in zip(*test_loader):
            for task_idx, (x, y, label) in enumerate(minibatch):
                x, y, label = x.to(device), y.to(device), label[:1].to(device)
                this_batch_size = x.shape[0]
                out, z, classify_logits, l_recons, l_gt = model(x, train_task_num + valid_task_num + task_idx)
                out = y_normalizer.decode(out.reshape(this_batch_size, 2 * S, S))

                loss_data = torch.linalg.norm(out.view(this_batch_size, -1) - y.view(this_batch_size, -1)) ** 2
                loss_recons = this_batch_size * torch.linalg.norm(l_recons.view(1, -1) - l_gt.view(1, -1)) ** 2
                loss_kl = this_batch_size * torch.linalg.norm(z) ** 2
                loss_classify = this_batch_size * ce_loss(classify_logits, label)

                test_l2_data += myloss(out.view(this_batch_size, -1), y.view(this_batch_size, -1)).item()
                test_l2_recons += this_batch_size * myloss(l_recons.view(1, -1), l_gt.view(1, -1)).item()
                test_abs_data += loss_data.item()
                test_abs_recons += loss_recons.item()
                test_kl += loss_kl.item()
                test_classify += loss_classify.item()

                test_l2 += loss_data + loss_recons + loss_kl + cls_loss_coeff * loss_classify
    test_abs_data /= ntest
    test_abs_recons /= ntest
    test_l2_data /= ntest
    test_l2_recons /= ntest
    test_kl /= ntest
    test_classify /= ntest
    test_l2 = test_l2.item() / ntest

    t3 = default_timer()

    f = open(res_file, "a")
    f.write(f'{ntrain}, {iseed}, {learning_rate}, {gamma}, {wd}, {train_loss_lowest}, {train_loss_best}, '
            f'{train_abs_data_best}, {train_abs_recons_best}, {train_kl_best}, '
            f'{train_classify_best}, {train_l2_data_best}, {train_l2_recons_best}, '
            f'{valid_loss_best}, {valid_abs_data_best}, {valid_abs_recons_best}, {valid_kl_best}, '
            f'{valid_classify_best}, {valid_l2_data_best}, {valid_l2_recons_best}, '
            f'{test_l2}, {test_abs_data}, {test_abs_recons}, {test_kl}, '
            f'{test_classify}, {test_l2_data}, {test_l2_recons}, '
            f'{best_epoch}, {(t3 - t0) / 3600:.2f}\n')
    f.close()


if __name__ == '__main__':
    # 4 * 3 * 5 = 20 * 3
    lrs = [3e-3, 2e-3, 1e-3, 5e-4]
    gammas = [0.9, 0.7, 0.5]
    wds = [1e-5, 3e-6, 1e-6, 3e-7, 1e-7]

    latent_dim = int(sys.argv[1])  # pick from [2, 5, 10, 15]
    # latent_dim = 10
    classify_loss_coeff = 1.0

    if len(sys.argv) > 2:
        lrs = [lrs[int(sys.argv[2])]]
        wds = [wds[int(sys.argv[3])]]

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
                    main(iseed, lr, gamma, wd, latent_dim, classify_loss_coeff)

    print(f'********** Training completed! **********')
