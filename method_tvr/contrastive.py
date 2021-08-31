import torch
import math
import torch.nn.functional as F


def log_sum_exp(x, axis=None):
    """
    Log sum exp function
    Args:
        x: Input.
        axis: Axis over which to perform sum.
    Returns:
        torch.Tensor: log sum exp
    """
    x_max = torch.max(x, axis)[0]
    y = torch.log((torch.exp(x - x_max)).sum(axis)) + x_max
    return y


def get_positive_expectation(p_samples, measure='JSD', average=True):
    """
    Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)
    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(-p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = torch.ones_like(p_samples) - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        raise ValueError('Unknown measurement {}'.format(measure))
    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(q_samples, measure='JSD', average=True):
    """
    Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = math.log(2.)
    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'DV':
        Eq = log_sum_exp(q_samples, 0) - math.log(q_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples
    else:
        raise ValueError('Unknown measurement {}'.format(measure))
    if average:
        return Eq.mean()
    else:
        return Eq


def batch_video_query_loss(video, query, match_labels, mask, measure='JSD'):
    """
        QV-CL module
        Computing the Contrastive Loss between the video and query.
        :param video: video rep (bsz, Lv, dim)
        :param query: query rep (bsz, dim)
        :param match_labels: match labels (bsz, Lv)
        :param mask: mask (bsz, Lv)
        :param measure: estimator of the mutual information
        :return: L_{qv}
    """
    # generate mask
    pos_mask = match_labels.type(torch.float32)  # (bsz, Lv)
    neg_mask = (torch.ones_like(pos_mask) - pos_mask) * mask  # (bsz, Lv)

    # compute scores
    query = query.unsqueeze(2)  # (bsz, dim, 1)
    res = torch.matmul(video, query).squeeze(2)  # (bsz, Lv)

    # computing expectation for the MI between the target moment (positive samples) and query.
    E_pos = get_positive_expectation(res * pos_mask, measure, average=False)
    E_pos = torch.sum(E_pos * pos_mask, dim=1) / (torch.sum(pos_mask, dim=1) + 1e-12)  # (bsz, )

    # computing expectation for the MI between clips except target moment (negative samples) and query.
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False)
    E_neg = torch.sum(E_neg * neg_mask, dim=1) / (torch.sum(neg_mask, dim=1) + 1e-12)  # (bsz, )

    E = E_neg - E_pos  # (bsz, )
    return torch.mean(E)


def batch_video_video_loss(video, st_ed_indices, match_labels, mask, measure='JSD'):
    """
        VV-CL module
        Computing the Contrastive loss between the start/end clips and the video
        :param video: video rep (bsz, Lv, dim)
        :param st_ed_indices: (bsz, 2)
        :param match_labels: match labels (bsz, Lv)
        :param mask: mask (bsz, Lv)
        :param measure: estimator of the mutual information
        :return: L_{vv}
    """
    # generate mask
    pos_mask = match_labels.type(torch.float32)  # (bsz, Lv)
    neg_mask = (torch.ones_like(pos_mask) - pos_mask) * mask  # (bsz, Lv)

    # select start and end indices features
    st_indices, ed_indices = st_ed_indices[:, 0], st_ed_indices[:, 1]  # (bsz, )
    batch_indices = torch.arange(0, video.shape[0]).long()  # (bsz, )
    video_s = video[batch_indices, st_indices, :]  # (bsz, dim)
    video_e = video[batch_indices, ed_indices, :]  # (bsz, dim)

    # compute scores
    video_s = video_s.unsqueeze(2)  # (bsz, dim, 1)
    res_s = torch.matmul(video, video_s).squeeze(2)  # (bsz, Lv), fusion between the start clips and the video
    video_e = video_e.unsqueeze(2)  # (bsz, dim, 1)
    res_e = torch.matmul(video, video_e).squeeze(2)  # (bsz, Lv), fusion between the end clips and the video

    # start clips: MI expectation for all positive samples
    E_s_pos = get_positive_expectation(res_s * pos_mask, measure, average=False)
    E_s_pos = torch.sum(E_s_pos * pos_mask, dim=1) / (torch.sum(pos_mask, dim=1) + 1e-12)  # (bsz, )
    # end clips: MI expectation for all positive samples
    E_e_pos = get_positive_expectation(res_e * pos_mask, measure, average=False)
    E_e_pos = torch.sum(E_e_pos * pos_mask, dim=1) / (torch.sum(pos_mask, dim=1) + 1e-12)
    E_pos = E_s_pos + E_e_pos

    # start clips: MI expectation for all negative samples
    E_s_neg = get_negative_expectation(res_s * neg_mask, measure, average=False)
    E_s_neg = torch.sum(E_s_neg * neg_mask, dim=1) / (torch.sum(neg_mask, dim=1) + 1e-12)

    # end clips: MI expectation for all negative samples
    E_e_neg = get_negative_expectation(res_e * neg_mask, measure, average=False)
    E_e_neg = torch.sum(E_e_neg * neg_mask, dim=1) / (torch.sum(neg_mask, dim=1) + 1e-12)
    E_neg = E_s_neg + E_e_neg

    E = E_neg - E_pos  # (bsz, )
    return torch.mean(E)
