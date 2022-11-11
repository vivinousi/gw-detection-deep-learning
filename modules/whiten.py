import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.fft import rfft, rfftfreq, irfft


def torch_inverse_spectrum_truncation(psd, max_filter_len, low_frequency_cutoff=9., delta_f=1.,
                                      trunc_method='hann'):
    N = (psd.size(1) - 1) * 2
    inv_asd = torch.zeros_like(psd)
    kmin = int(low_frequency_cutoff / delta_f)
    inv_asd[0, kmin:N//2] = (1. / psd[0, kmin:N//2]) ** 0.5

    q = irfft(inv_asd, n=N, norm='forward')
    trunc_start = max_filter_len // 2
    trunc_end = N - max_filter_len // 2
    if trunc_method == 'hann':
        trunc_window = torch.hann_window(max_filter_len, dtype=torch.float64).to(psd.device)
        q[0, 0:trunc_start] *= trunc_window[-trunc_start:]
        q[0, trunc_end:] *= trunc_window[:max_filter_len // 2]
    if trunc_start < trunc_end:
        q[0, trunc_start:trunc_end] = 0
    psd_trunc = rfft(q, n=N, norm='forward')
    psd_trunc *= psd_trunc.conj()

    psd = 1 / torch.abs(psd_trunc)
    return psd / 2


class Whiten(nn.Module):
    def __init__(self, delta_t, low_frequency_cutoff=15., m=1.25, max_filter_len=1., legacy=True):
        super().__init__()
        # store psd estimate
        self.max_filter_len = max_filter_len
        self.legacy = legacy
        self.delta_t = delta_t
        self.delta_f = 1 / m
        m /= delta_t
        self.m = int(m)
        self.d = int(m / 2)
        self.psd_est = None
        self.norm = nn.Parameter(torch.zeros(2, 1281), requires_grad=False)
        self.frequencies = None
        self.low_frequency_cutoff = low_frequency_cutoff
        self.frequencies = rfftfreq(self.m, d=self.delta_t)

    def initialize(self, noise_t):
        if noise_t.dim() == 2:
            noise_t = noise_t.unsqueeze(0).unsqueeze(2)
        n_channels = noise_t.size(1)
        psds = []
        for c in range(n_channels):
            psd = self.estimate_psd(noise_t[:, c, :, :].unsqueeze(1))
            psds.append(psd)
        self.psd_est = torch.cat(psds, dim=0)
        if self.legacy:
            self.norm.data = torch.sqrt(self.psd_est / self.delta_f)
            idx = int(self.low_frequency_cutoff / self.delta_f)
            self.norm.data[:, :idx] = self.norm.data[:, idx].view(-1, 1)
            self.norm.data[:, -1:] = self.norm.data[:, -2].view(-1, 1)
        else:
            self.norm.data = self.psd_est ** 0.5

    def estimate_psd(self, noise_t):
        """
        noise in (1, C, 1, D) Tensor format
        """
        # step 1: split signal into L segments of M length with D overlap
        m = self.m
        d = self.d
        segments = F.unfold(noise_t, kernel_size=(1, m), stride=(1, d)).double()
        n_segments = segments.size(2)

        # step 2: apply hann window over all segments
        w_hann = torch.hann_window(segments.size(1), periodic=True, dtype=torch.float64).to(segments.device)
        segments_w = segments * w_hann.unsqueeze_(1)

        # step 3: compute FFT for all segments
        segments_fft = rfft(segments_w, dim=1, norm="forward")

        segments_sq_mag = torch.abs(segments_fft * segments_fft.conj())
        segments_sq_mag[0, 0, :] /= 2
        segments_sq_mag[0, -1, :] /= 2

        # step 4: aggregate (we use the mean, but pycbc uses median by default)
        t_psd = torch.mean(segments_sq_mag, dim=2)
        t_psd *= 2 * self.delta_f * m / (w_hann * w_hann).sum()

        # final step: interpolate if needed and inverse spectrum truncation
        if t_psd.size(1) != 1281:
            t_psd = F.interpolate(t_psd.unsqueeze(1), 1281).squeeze(1)
            self.frequencies = rfftfreq(int(1.25 * 2048), d=self.delta_t)
        t_psd = torch_inverse_spectrum_truncation(t_psd, int(self.max_filter_len / self.delta_t),
                                                  low_frequency_cutoff=15,
                                                  delta_f=self.delta_f)
        return t_psd

    def forward(self, signal):
        # then whiten signal with self.psd_est and return the whitened version
        return self.whiten(signal)

    def whiten(self, signal):
        signal_f = rfft(signal.double(), dim=2, norm='forward')
        # this works when we have as many psds as we do channels
        signal_t = irfft(signal_f / self.norm, norm='forward', n=signal.size(2))
        # this works in every case
        # norms = torch.split(self.norm.double(), 2, dim=0)
        # signal_t = torch.cat([irfft(signal_f / norm, norm='forward', n=signal.size(2)) for norm in norms], dim=1)
        return signal_t.float()


class CropWhitenNet(nn.Module):
    def __init__(self, net=None, norm=None, deploy=False, m=0.625, l=0.5, f=15.):
        super(CropWhitenNet, self).__init__()
        self.net = net
        self.norm = norm
        self.whiten = Whiten(1/2048, low_frequency_cutoff=f, m=m, max_filter_len=l, legacy=False)
        self.deploy = deploy
        self.step = 0.1

    def forward(self, x, inj_times=None):
        segments_wh = []
        n_batch = x.size(0)
        slice_len = x.size(2)
        if inj_times is not None:
            # inj_times contains the times of coalescence for each sample, this is only available during training and
            # validation
            for i, sample in enumerate(x):
                with torch.no_grad():
                    self.whiten.initialize(sample)
                if inj_times[i] < 0:
                    # negative inj_time means negative sample, crop random 1.25s
                    if sample.size(1) > 2560:
                        crop_idx = np.random.randint(sample.size(1) - 2560)
                    else:
                        crop_idx = 0
                    segment = sample[:, crop_idx:crop_idx + 2560].unsqueeze_(0)
                else:
                    # positive sample, crop around inj_time
                    int_s = int(inj_times[i] // 1)
                    crop_idx = int_s * 2048
                    segment = sample[:, crop_idx:crop_idx + 2560].unsqueeze_(0)
                # crop 0.125s from the beginning and from the end of the whitened samples
                segments_wh.append(self.whiten(segment)[:, :, 256:-256])
            segments_wh = torch.cat(segments_wh)
        else:
            # inj_times not available during testing
            segments_wh = []
            with torch.no_grad():
                c = x.size(1)
                for i, sample in enumerate(x):
                    # compute psd for each sample in batch
                    self.whiten.initialize(sample)
                    sample = sample.unsqueeze(0).unsqueeze(2)
                    # split each sample into sub-batches every 0.1s (or 204 indices)
                    x_segments = F.unfold(sample, kernel_size=(1, 2560), stride=(1, 204)).contiguous()
                    n = sample.size(0)
                    l = x_segments.size(2)
                    x_segments = x_segments.view(n, c, -1, l).permute(3, 1, 2, 0).squeeze_(3)
                    segments_wh.append(self.whiten(x_segments)[:, :, 256:-256])
                segments_wh = torch.cat(segments_wh)

        if self.norm is not None:
            segments_wh = self.norm(segments_wh)

        if self.deploy:
            # we need to add 0.1s to each sample in the batch
            added_time = torch.arange(0, slice_len / 2048. - 1.25 + self.step, self.step).repeat(n_batch)
            return self.net(segments_wh), added_time
        else:
            return self.net(segments_wh)
