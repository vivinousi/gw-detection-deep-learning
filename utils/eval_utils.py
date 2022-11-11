import torch
import numpy as np
import h5py
import logging
from tqdm import tqdm
from pycbc.types import TimeSeries


def find_injection_times(fgfiles, injfile, padding_start=0, padding_end=0):
    """Determine injections which are contained in the file.

    Arguments
    ---------
    fgfiles : list of str
        Paths to the files containing the foreground data (noise +
        injections).
    injfile : str
        Path to the file containing information on the injections in the
        foreground files.
    padding_start : {float, 0}
        The amount of time (in seconds) at the start of each segment
        where no injections are present.
    padding_end : {float, 0}
        The amount of time (in seconds) at the end of each segment
        where no injections are present.

    Returns
    -------
    duration:
        A float representing the total duration (in seconds) of all
        foreground files.
    bool-indices:
        A 1D array containing bools that specify which injections are
        contained in the provided foreground files.
    """
    duration = 0
    times = []
    for fpath in fgfiles:
        with h5py.File(fpath, 'r') as fp:
            det = list(fp.keys())[0]

            for key in fp[det].keys():
                ds = fp[f'{det}/{key}']
                start = ds.attrs['start_time']
                end = start + len(ds) * ds.attrs['delta_t']
                duration += end - start
                start += padding_start
                end -= padding_end
                if end > start:
                    times.append([start, end])

    with h5py.File(injfile, 'r') as fp:
        injtimes = fp['tc'][()]

    ret = np.full((len(times), len(injtimes)), False)
    for i, (start, end) in enumerate(times):
        ret[i] = np.logical_and(start <= injtimes, injtimes <= end)

    return duration, np.any(ret, axis=0)


def mchirp(mass1, mass2):
    return (mass1 * mass2) ** (3. / 5.) / (mass1 + mass2) ** (1. / 5.)


# code or part of coded adapted from https://github.com/gwastro/ml-mock-data-challenge-1
class Slicer(object):
    """Class that is used to slice and iterate over a single input data
    file.

    Arguments
    ---------
    infile : open file object
        The open HDF5 file from which the data should be read.
    step_size : {float, 0.1}
        The step size (in seconds) for slicing the data.
    peak_offset : {float, 0.6}
        The time (in seconds) from the start of each window where the
        peak is expected to be on average.
    slice_length : {int, 2048}
        The length of the output slice in samples.
    detectors : {None or list of datasets}
        The datasets that should be read from the infile. If set to None
        all datasets listed in the attribute `detectors` will be read.
    """

    def __init__(self, infile, step_size=0.1, peak_offset=0.6,
                 slice_length=2048, detectors=None, whiten=True):
        self.infile = infile
        self.step_size = step_size  # this is the approximate one passed as an argument, the exact one is defined in the __next__ method
        self.peak_offset = peak_offset
        self.slice_length = slice_length
        self.detectors = detectors
        self.whiten = whiten
        # self.detectors = ['H1', 'L1']
        # if self.detectors is None:
        #     self.detectors = [self.infile[key] for key in list(self.infile.attrs['detectors'])]
        self.detector_names = list(self.infile.keys())
        self.detectors = [self.infile[key] for key in self.detector_names]
        self.keys = sorted(list(self.detectors[0].keys()),
                           key=lambda inp: int(inp))
        self.determine_n_slices()
        return

    def determine_n_slices(self):
        self.n_slices = {}
        start = 0
        for ds_key in self.keys:
            ds = self.detectors[0][ds_key]
            dt = ds.attrs['delta_t']
            index_step_size = int(self.step_size / dt)

            if self.whiten:
                nsteps = int((len(ds) - self.slice_length - 512) // \
                             index_step_size)
            else:
                nsteps = int((len(ds) - self.slice_length) // \
                             index_step_size)

            self.n_slices[ds_key] = {'start': start,
                                     'stop': start + nsteps,
                                     'len': nsteps}
            start += nsteps

    def __len__(self):
        return sum([val['len'] for val in self.n_slices.values()])

    def _generate_access_indices(self, index):
        assert index.step is None or index.step == 1, 'Slice with step is not supported'
        ret = {}
        start = index.start
        stop = index.stop
        for key in self.keys:
            cstart = self.n_slices[key]['start']
            cstop = self.n_slices[key]['stop']
            if cstart <= start and start < cstop:
                ret[key] = slice(start, min(stop, cstop))
                start = ret[key].stop
        return ret

    def generate_data(self, key, index):
        # Ideally set dt = self.detectors[0][key].attrs['delta_t']
        # Due to numerical limitations this may be off by a single sample
        dt = 1. / 2048  # This definition limits the scope of this object
        index_step_size = int(self.step_size / dt)
        sidx = (index.start - self.n_slices[key]['start']) * index_step_size
        eidx = (index.stop - self.n_slices[key]['start']) * index_step_size + self.slice_length + 512
        rawdata = [det[key][sidx:eidx] for det in self.detectors]
        times = (self.detectors[0][key].attrs['start_time'] + sidx * dt) + index_step_size * dt * np.arange(
            index.stop - index.start) + self.peak_offset

        data = np.zeros((index.stop - index.start, len(rawdata), self.slice_length))
        for detnum, rawdat in enumerate(rawdata):
            for i in range(index.stop - index.start):
                sidx = i * index_step_size
                if self.whiten:
                    eidx = sidx + self.slice_length + 512
                    ts = TimeSeries(rawdat[sidx:eidx], delta_t=dt)
                    ts = ts.whiten(0.5, 0.25, low_frequency_cutoff=18.)
                else:
                    eidx = sidx + self.slice_length
                    ts = TimeSeries(rawdat[sidx:eidx], delta_t=dt)
                data[i, detnum, :] = ts.numpy()
        return data, times

    def __getitem__(self, index):
        is_single = False
        if isinstance(index, int):
            is_single = True
            if index < 0:
                index = len(self) + index
            index = slice(index, index + 1)
        access_slices = self._generate_access_indices(index)

        data = []
        times = []
        for key, idxs in access_slices.items():
            dat, t = self.generate_data(key, idxs)
            data.append(dat)
            times.append(t)
        data = np.concatenate(data)
        times = np.concatenate(times)

        if is_single:
            return data[0], times[0]
        else:
            return data, times

# code or part of coded adapted from https://github.com/gwastro/ml-mock-data-challenge-1
class TorchSlicer(Slicer, torch.utils.data.Dataset):
    def __init__(self, *args, **kwargs):
        torch.utils.data.Dataset.__init__(self)
        Slicer.__init__(self, *args, **kwargs)

    def __getitem__(self, index):
        next_slice, next_time = Slicer.__getitem__(self, index)
        return torch.from_numpy(next_slice), torch.tensor(next_time)

# code or part of coded adapted from https://github.com/gwastro/ml-mock-data-challenge-1
def get_clusters(triggers, cluster_threshold=0.35, var=0.2):
    """Cluster a set of triggers into candidate detections.

    Arguments
    ---------
    triggers : list of triggers
        A list of triggers.  A trigger is a list of length two, where
        the first entry represents the trigger time and the second value
        represents the accompanying output value from the network.
    cluster_threshold : {float, 0.35}
        Cluster triggers together which are no more than this amount of
        time away from the boundaries of the corresponding cluster.

    Returns
    cluster_times :
        A numpy array containing the single times associated to each
        cluster.
    cluster_values :
        A numpy array containing the trigger values at the corresponing
        cluster_times.
    cluster_timevars :
        The timing certainty for each cluster. Injections must be within
        the given value for the cluster to be counted as true positive.
    """
    clusters = []
    for trigger in triggers:
        new_trigger_time = trigger[0]
        if len(clusters) == 0:
            start_new_cluster = True
        else:
            last_cluster = clusters[-1]
            last_trigger_time = last_cluster[-1][0]
            start_new_cluster = (new_trigger_time - last_trigger_time) > cluster_threshold
        if start_new_cluster:
            clusters.append([trigger])
        else:
            last_cluster.append(trigger)

    logging.info(
        "Clustering has resulted in %i independent triggers. Centering triggers at their maxima." % len(clusters))

    cluster_times = []
    cluster_values = []
    cluster_timevars = []

    ### Determine maxima of clusters and the corresponding times and append them to the cluster_* lists
    for cluster in clusters:
        times = [trig[0] for trig in cluster]
        values = np.array([trig[1] for trig in cluster])
        max_index = np.argmax(values)
        cluster_times.append(times[max_index])
        cluster_values.append(values[max_index])
        cluster_timevars.append(var)

    cluster_times = np.array(cluster_times)
    cluster_values = np.array(cluster_values)
    cluster_timevars = np.array(cluster_timevars)

    return cluster_times, cluster_values, cluster_timevars

# code or part of coded adapted from https://github.com/gwastro/ml-mock-data-challenge-1
def get_triggers(Network, inputfile, step_size=0.1,
                 trigger_threshold=0.2, device='cpu',
                 verbose=False, dtype=torch.float32,
                 batch_size=512, slicer_cls=TorchSlicer,
                 num_workers=8, whiten=True, slice_length=2048):
    """Use a network to generate a list of triggers, where the network
    outputs a value above a given threshold.

    Arguments
    ---------
    Network : network as returned by get_network
        The network to use during the evaluation.
    inputfile : str
        Path to the input data file.
    step_size : {float, 0.1}
        The step size (in seconds) to use for slicing the data.
    trigger_threshold : {float, 0.2}
        The value to use as a threshold on the network output to create
        triggers.
    device : {str, `cpu`}
        The device on which the calculations are carried out.
    verbose : {bool, False}
        Print update messages.

    Returns
    -------
    triggers:
        A list of of triggers. A trigger is a list of length two, where
        the first entry represents the trigger time and the second value
        represents the accompanying output value from the network.
    """
    Network.to(dtype=dtype, device=device)
    with h5py.File(inputfile, 'r') as infile:
        slicer = slicer_cls(infile, step_size=step_size, whiten=whiten, slice_length=slice_length)
        triggers = []
        data_loader = torch.utils.data.DataLoader(slicer,
                                                  batch_size=batch_size,
                                                  shuffle=False, num_workers=num_workers,
                                                  pin_memory=True if 'cuda' in device else False)
        ### Gradually apply network to all samples and if output exceeds the trigger threshold, save the time and the output value
        iterable = tqdm(data_loader, desc="Iterating over dataset") if verbose else data_loader
        for slice_batch, slice_times in iterable:
            with torch.no_grad():
                output_values = Network(slice_batch.to(device=device))
                if isinstance(output_values, tuple):
                    output_added_times = output_values[1]
                    output_values = output_values[0]
                else:
                    output_added_times = None
                output_values = output_values[:, 0]
                trigger_bools = torch.gt(output_values, trigger_threshold)
                if output_added_times is not None:
                    n_slices = output_added_times.size(0) // slice_batch.size(0)
                    slice_times = slice_times.repeat_interleave(n_slices)
                    slice_times += output_added_times
                    for slice_time, trigger_bool, output_value in zip(slice_times, trigger_bools, output_values):
                        if trigger_bool.clone().cpu().item():
                            triggers.append(
                                [slice_time.clone().cpu().item() + 0.125, output_value.clone().cpu().item()])
                else:
                    for slice_time, trigger_bool, output_value in zip(slice_times, trigger_bools, output_values):
                        if trigger_bool.clone().cpu().item():
                            triggers.append(
                                [slice_time.clone().cpu().item() + 0.125, output_value.clone().cpu().item()])
        logging.info("A total of %i slices have exceeded the threshold of %f." % (len(triggers), trigger_threshold))
    return triggers
