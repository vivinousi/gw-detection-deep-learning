"""
This file uses the injections HDF file to gather the parameters of the injections and generate the corresponding
waveforms. The result is saved in a .npy file which is then used by the SlicerDataset classes.
"""

import numpy as np
from tqdm import tqdm

from pycbc.waveform import get_td_waveform
from pycbc.inject import InjectionSet
from pycbc.inject import projector

from utils.eval_utils import find_injection_times


def make_strain_from_params(inj, delta_t, detector_name,
                            distance_scale=1., f_lower=None):
    # compute the waveform time series
    f_lower = inj.f_lower if f_lower is None else f_lower
    hp, hc = get_td_waveform(inj, delta_t=delta_t, f_lower=f_lower)
    return projector(detector_name,
                     inj, hp, hc, distance_scale=distance_scale)


hdf_filename = 'dataset-4/v2/val_background_s24w6d1_1.hdf'
injections_file = 'dataset-2/v2/val_injections_s24w6d1_1.hdf'

injector = InjectionSet(injections_file)
injtable = injector.table
dur, idxs = find_injection_times([hdf_filename], injections_file)
injtable = injtable[idxs]
n_injections = len(injtable)
print(n_injections)

waveforms = np.zeros((n_injections, 2, 2560), dtype=np.float32)
print(f'GBs required for {n_injections} waveforms: {waveforms.nbytes / (1024 ** 3)}')

for i in tqdm(range(n_injections)):
    inj = injtable[i]
    w_h1 = make_strain_from_params(inj, 1/2048., 'H1', f_lower=18)
    w_l1 = make_strain_from_params(inj, 1/2048., 'L1', f_lower=18)
    w_h1.append_zeros(512)
    w_h1.prepend_zeros(1024)
    w_l1.append_zeros(512)
    w_l1.prepend_zeros(1024)
    # crop waveforms to 1.25s total duration with tc at 1s
    w_h1 = w_h1.time_slice(inj.tc - 1, inj.tc + 0.25, mode='nearest')
    w_l1 = w_l1.time_slice(inj.tc - 1, inj.tc + 0.25, mode='nearest')
    waveforms[i, 0, :] = np.float32(w_h1.numpy())
    waveforms[i, 1, :] = np.float32(w_l1.numpy())

np.save('dataset-2/v2/val_injections_s24w6d1_1.25s.npy', waveforms)
