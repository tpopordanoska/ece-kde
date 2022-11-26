import torch
from torch import nn


def get_bandwidth(f, device):
    """
    Select a bandwidth for the kernel based on maximizing the leave-one-out likelihood (LOO MLE).

    :param f: The vector containing the probability scores, shape [num_samples, num_classes]
    :param device: The device type: 'cpu' or 'cuda'

    :return: The bandwidth of the kernel
    """
    bandwidths = torch.cat((torch.logspace(start=-5, end=-1, steps=15), torch.linspace(0.2, 1, steps=5)))
    max_b = -1
    max_l = 0
    n = len(f)
    for b in bandwidths:
        log_kern = get_kernel(f, b, device)
        log_fhat = torch.logsumexp(log_kern, 1) - torch.log((n-1)*b)
        l = torch.sum(log_fhat)
        if l > max_l:
            max_l = l
            max_b = b

    return max_b


def get_ece_kde(f, y, bandwidth, p, mc_type, device):
    """
    Calculate an estimate of Lp calibration error.

    :param f: The vector containing the probability scores, shape [num_samples, num_classes]
    :param y: The vector containing the labels, shape [num_samples]
    :param bandwidth: The bandwidth of the kernel
    :param p: The p-norm. Typically, p=1 or p=2
    :param mc_type: The type of multiclass calibration: canonical, marginal or top_label
    :param device: The device type: 'cpu' or 'cuda'

    :return: An estimate of Lp calibration error
    """
    check_input(f, bandwidth, mc_type)
    if f.shape[1] == 1:
        return 2 * get_ratio_binary(f, y, bandwidth, p, device)
    else:
        if mc_type == 'canonical':
            return get_ratio_canonical(f, y, bandwidth, p, device)
        elif mc_type == 'marginal':
            return get_ratio_marginal_vect(f, y, bandwidth, p, device)
        elif mc_type == 'top_label':
            return get_ratio_toplabel(f, y, bandwidth, p, device)


def get_ratio_binary(f, y, bandwidth, p, device):
    assert f.shape[1] == 1

    log_kern = get_kernel(f, bandwidth, device)

    return get_kde_for_ece(f, y, log_kern, p)


def get_ratio_canonical(f, y, bandwidth, p, device):
    if f.shape[1] > 60:
        # Slower but more numerically stable implementation for larger number of classes
        return get_ratio_canonical_log(f, y, bandwidth, p, device)

    log_kern = get_kernel(f, bandwidth, device)
    kern = torch.exp(log_kern)

    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
    kern_y = torch.matmul(kern, y_onehot)
    den = torch.sum(kern, dim=1)
    # to avoid division by 0
    den = torch.clamp(den, min=1e-10)

    ratio = kern_y / den.unsqueeze(-1)
    ratio = torch.sum(torch.abs(ratio - f)**p, dim=1)

    return torch.mean(ratio)


# Note for training: Make sure there are at least two examples for every class present in the batch, otherwise
# LogsumexpBackward returns nans.
def get_ratio_canonical_log(f, y, bandwidth, p, device='cpu'):
    log_kern = get_kernel(f, bandwidth, device)
    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
    log_y = torch.log(y_onehot)
    log_den = torch.logsumexp(log_kern, dim=1)
    final_ratio = 0
    for k in range(f.shape[1]):
        log_kern_y = log_kern + (torch.ones([f.shape[0], 1]) * log_y[:, k].unsqueeze(0))
        log_inner_ratio = torch.logsumexp(log_kern_y, dim=1) - log_den
        inner_ratio = torch.exp(log_inner_ratio)
        inner_diff = torch.abs(inner_ratio - f[:, k])**p
        final_ratio += inner_diff

    return torch.mean(final_ratio)


def get_ratio_marginal_vect(f, y, bandwidth, p, device):
    y_onehot = nn.functional.one_hot(y, num_classes=f.shape[1]).to(torch.float32)
    log_kern_vect = beta_kernel(f, f, bandwidth).squeeze()
    log_kern_diag = torch.diag(torch.finfo(torch.float).min * torch.ones(len(f))).to(device)
    # Multiclass case
    log_kern_diag_repeated = f.shape[1] * [log_kern_diag]
    log_kern_diag_repeated = torch.stack(log_kern_diag_repeated, dim=2)
    log_kern_vect = log_kern_vect + log_kern_diag_repeated

    return get_kde_for_ece_vect(f, y_onehot, log_kern_vect, p)


def get_ratio_toplabel(f, y, bandwidth, p, device):
    f_max, indices = torch.max(f, 1)
    f_max = f_max.unsqueeze(-1)
    y_max = (y == indices).to(torch.int)

    return get_ratio_binary(f_max, y_max, bandwidth, p, device)


def get_kde_for_ece_vect(f, y, log_kern, p):
    log_kern_y = log_kern * y
    # Trick: -inf instead of 0 in log space
    log_kern_y[log_kern_y == 0] = torch.finfo(torch.float).min

    log_num = torch.logsumexp(log_kern_y, dim=1)
    log_den = torch.logsumexp(log_kern, dim=1)

    log_ratio = log_num - log_den
    ratio = torch.exp(log_ratio)
    ratio = torch.abs(ratio - f)**p

    return torch.sum(torch.mean(ratio, dim=0))


def get_kde_for_ece(f, y, log_kern, p):
    f = f.squeeze()
    N = len(f)
    # Select the entries where y = 1
    idx = torch.where(y == 1)[0]
    if not idx.numel():
        return torch.sum((torch.abs(-f))**p) / N

    if idx.numel() == 1:
        # because of -inf in the vector
        log_kern = torch.cat((log_kern[:idx], log_kern[idx+1:]))
        f_one = f[idx]
        f = torch.cat((f[:idx], f[idx+1:]))

    log_kern_y = torch.index_select(log_kern, 1, idx)

    log_num = torch.logsumexp(log_kern_y, dim=1)
    log_den = torch.logsumexp(log_kern, dim=1)

    log_ratio = log_num - log_den
    ratio = torch.exp(log_ratio)
    ratio = torch.abs(ratio - f)**p

    if idx.numel() == 1:
        return (ratio.sum() + f_one ** p)/N

    return torch.mean(ratio)


def get_kernel(f, bandwidth, device):
    # if num_classes == 1
    if f.shape[1] == 1:
        log_kern = beta_kernel(f, f, bandwidth).squeeze()
    else:
        log_kern = dirichlet_kernel(f, bandwidth).squeeze()
    # Trick: -inf on the diagonal
    return log_kern + torch.diag(torch.finfo(torch.float).min * torch.ones(len(f))).to(device)


def beta_kernel(z, zi, bandwidth=0.1):
    p = zi / bandwidth + 1
    q = (1-zi) / bandwidth + 1
    z = z.unsqueeze(-2)

    log_beta = torch.lgamma(p) + torch.lgamma(q) - torch.lgamma(p + q)
    log_num = (p-1) * torch.log(z) + (q-1) * torch.log(1-z)
    log_beta_pdf = log_num - log_beta

    return log_beta_pdf


def dirichlet_kernel(z, bandwidth=0.1):
    alphas = z / bandwidth + 1

    log_beta = (torch.sum((torch.lgamma(alphas)), dim=1) - torch.lgamma(torch.sum(alphas, dim=1)))
    log_num = torch.matmul(torch.log(z), (alphas-1).T)
    log_dir_pdf = log_num - log_beta

    return log_dir_pdf


def check_input(f, bandwidth, mc_type):
    assert not isnan(f)
    assert len(f.shape) == 2
    assert bandwidth > 0
    assert torch.min(f) >= 0
    assert torch.max(f) <= 1


def isnan(a):
    return torch.any(torch.isnan(a))
