import multiprocessing as mp
import queue
import time
import warnings
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

from utils.eval_utils import find_injection_times, mchirp


class Slicer(object):
    """Generator to access slices of an input file as output by generate_data.py
    for the MLGWSC-1.

    Arguments
    ---------
    filepath : str
        Path to the file that should be sliced.
    step_size : {int, 204}
        The stride of the slicer in samples.
    window_size : {int, 2048}
        The size of the window in samples for which processing is done.
    workers : {None or int > 0, None}
        How many processes to start for data reading and processing.
    prefetch : {int, 0}
        How many samples to pre-calculate. Can improve performance at
        at the cost of memory efficiency.
    timeout : {float, 0.01}
        How long to wait when trying to read from or write to a prallel
        queue.
    batch_size : {int, 1}
        The number of samples to accumulate for each call to __next__.

    Notes
    -----
    +To apply processing to the data, sub-class this class and overwrite
     the methods
     -process_slice
     -format_return
     The process_slice method runs in parallel, if multiple workers were
     requested. Any heavy processing should be put into this method.
     The format_return method runs sequentially and should only do light
     re-formatting of the output.
    +Usage:
     # >>> gen = Slicer(filepath, workers=2, prefetch=4)
     # >>> with gen:
     # >>>    results = list(gen)
    """

    def __init__(self, filepath, step_size=204, window_size=2048,
                 workers=None, prefetch=0, timeout=0.01, batch_size=1):
        self.filepath = filepath
        self.step_size = int(step_size)
        self.window_size = int(window_size)
        self.workers = workers
        self.prefetch = prefetch
        self.timeout = timeout
        self.batch_size = batch_size
        self.entered = False
        self._init_file_vars()
        self.determine_n_slices()
        self.reset()

    def _init_file_vars(self):
        with h5py.File(self.filepath, 'r') as fp:
            self.detectors = list(fp.attrs['detectors'])
            self.sample_rate = fp.attrs['sample_rate']
            self.flow = fp.attrs['low_frequency_cutoff']
            self.keys = sorted(fp[self.detectors[0]].keys(),
                               key=lambda inp: int(inp))

    def determine_n_slices(self):
        self.n_slices = {}
        start = 0
        with h5py.File(self.filepath, 'r') as fp:
            for ds_key in self.keys:
                ds = fp[self.detectors[0]][ds_key]

                nsteps = int((len(ds) - self.window_size) // self.step_size)

                self.n_slices[ds_key] = {'start': start,
                                         'stop': start + nsteps,
                                         'len': nsteps}
                start += nsteps

    @property
    def n_samples(self):
        if not hasattr(self, 'n_slices'):
            self.determine_n_slices()
        return sum([val['len'] for val in self.n_slices.values()])

    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))

    def empty_queues(self):
        while True:
            try:
                self.fetched.get(timeout=0.01)
            except (queue.Empty, AttributeError):
                break
        while True:
            try:
                self.index_queue.get(timeout=0.01)
            except (queue.Empty, AttributeError):
                break

    def reset(self):
        self.index = 0
        self.empty_queues()
        self.last_index_put = -1
        if hasattr(self, 'last_fetched'):
            self.last_fetched.value = -1

    def _access_index(self, index):
        for ds, dic in self.n_slices.items():
            if dic['start'] <= index and index < dic['stop']:
                return (ds, index - dic['start'])
        else:
            raise IndexError('Index not found')

    def __iter__(self):
        return self

    def __getitem__(self, index):
        if self.index >= len(self):
            raise IndexError
        if self.workers is None \
                or self.prefetch < 1 \
                or not self.entered:  # Single process
            if self.workers is not None \
                    and self.workers > 0 \
                    and self.prefetch > 0:
                warnings.warn(("Multiple workers were requested but the "
                               "generator was not entered. Remember to use "
                               "the generator as a context manager. Running "
                               "sequentially."), RuntimeWarning)
            batch_idxs = list(range(index * self.batch_size,
                                    min(self.n_samples,
                                        (index + 1) * self.batch_size)))
            ret = [[] for _ in self.detectors]
            for idx in batch_idxs:
                ds, dsidx = self._access_index(idx)
                start = dsidx * self.step_size
                stop = start + self.window_size
                with h5py.File(self.filepath, 'r') as fp:
                    for i, det in enumerate(self.detectors):
                        data = fp[det][ds][start:stop]
                        ret[i].append(self.process_slice(data, det))
        else:  # Multiprocessing
            upper = min(index + self.prefetch, len(self))
            if upper > self.last_index_put:
                for i in range(self.last_index_put + 1, upper):
                    self.index_queue.put(i)
                    self.last_index_put = i
                if len(self) <= upper:
                    self.last_index_put = len(self)
            while True:
                try:
                    ret = self.fetched.get(timeout=self.timeout)
                    break
                except queue.Empty:
                    continue

        # self.index += 1
        ret = [np.stack(pt) for pt in ret]
        return self.format_return(ret)

    def __next__(self):
        if self.index >= len(self):
            raise StopIteration
        if self.workers is None \
                or self.prefetch < 1 \
                or not self.entered:  # Single process
            if self.workers is not None \
                    and self.workers > 0 \
                    and self.prefetch > 0:
                warnings.warn(("Multiple workers were requested but the "
                               "generator was not entered. Remember to use "
                               "the generator as a context manager. Running "
                               "sequentially."), RuntimeWarning)
            batch_idxs = list(range(self.index * self.batch_size,
                                    min(self.n_samples,
                                        (self.index + 1) * self.batch_size)))
            ret = [[] for _ in self.detectors]
            for idx in batch_idxs:
                ds, dsidx = self._access_index(idx)
                start = dsidx * self.step_size
                stop = start + self.window_size
                with h5py.File(self.filepath, 'r') as fp:
                    for i, det in enumerate(self.detectors):
                        data = fp[det][ds][start:stop]
                        ret[i].append(self.process_slice(data, det))
        else:  # Multiprocessing
            upper = min(self.index + self.prefetch, len(self))
            if upper > self.last_index_put:
                for i in range(self.last_index_put + 1, upper):
                    self.index_queue.put(i)
                    self.last_index_put = i
                if len(self) <= upper:
                    self.last_index_put = len(self)
            while True:
                try:
                    ret = self.fetched.get(timeout=self.timeout)
                    break
                except queue.Empty:
                    continue

        self.index += 1
        ret = [np.stack(pt) for pt in ret]
        return self.format_return(ret)

    def _fetch_func(self, pidx, index_pipe, output_pipe, event):
        ret = None
        index = None
        with h5py.File(self.filepath, 'r') as fp:
            while not event.is_set():
                if ret is None:
                    try:
                        index = index_pipe.get(timeout=self.timeout)
                        batch_idxs = list(range(index * self.batch_size,
                                                min(self.n_samples,
                                                    (index + 1) * self.batch_size)))
                        ret = [[] for _ in self.detectors]
                        for idx in batch_idxs:
                            ds, dsidx = self._access_index(idx)
                            start = dsidx * self.step_size
                            stop = start + self.window_size
                            for i, det in enumerate(self.detectors):
                                data = fp[det][ds][start:stop]
                                ret[i].append(self.process_slice(data, det))
                    except queue.Empty:
                        continue
                try:
                    if self.last_fetched.value + 1 != index:
                        time.sleep(self.timeout)
                    else:
                        output_pipe.put(ret, timeout=self.timeout)
                        self.last_fetched.value = index
                        ret = None
                except queue.Full:
                    continue

    def __enter__(self):
        if self.workers is not None and self.workers > 0 and self.prefetch > 0:
            self.event = mp.Event()
            self.processes = []
            self.fetched = mp.Queue(maxsize=2 * self.prefetch)
            self.index_queue = mp.Queue(maxsize=2 * self.prefetch)
            self.last_fetched = mp.Value('i', -1)
            for i in range(self.workers):
                process = mp.Process(target=self._fetch_func,
                                     args=(i,
                                           self.index_queue,
                                           self.fetched,
                                           self.event))
                self.processes.append(process)
                process.start()
            self.entered = True

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if hasattr(self, 'event'):
            self.event.set()
            self.empty_queues()
            if hasattr(self, 'processes'):
                self.empty_queues()
                while len(self.processes) > 0:
                    process = self.processes.pop(0)
                    process.join()
            self.event = None
            self.entered = False

    def process_slice(self, data, detector):
        """Applies processing to the raw data from one detector as read
        from the file.

        Arguments
        ---------
        data : numpy.array
            1 dimensional array of length self.window_size with dtype
            specified by the read file (usually numpy.float32).
        detector : str
            The string specifying the detector.

        Returns
        -------
        data
            Processed data that can be pickled.
        """
        return np.float32(data[np.newaxis, :])

    def format_return(self, data):
        """Formats the return value of the generator for a single slice
        and both detectors.

        Arguments
        ---------
        data : list of self.process_slice return values
            A list containing the processed single detector slices.

        Returns
        -------
        data
            The formatted data that can be understood by the consumer.
        """
        return np.concatenate(data, axis=1)


class SlicerDataset(Dataset):
    """
    Given a background hdf file and injections npy file, this class slices the background file and uses the segments
    once as negative samples and once as positive samples, after injection.
    """
    def __init__(self, background_hdf, injections_npy, slice_len=int(3.25 * 2048), slice_stride=int(2.5 * 2048),
                 max_seg_idx=3, return_time=True):
        self.slicer = Slicer(background_hdf, step_size=slice_stride, window_size=slice_len)
        self.n_slices = len(self.slicer)
        self.waves = np.load(injections_npy, mmap_mode='r')
        self.n_waves = self.waves.shape[0]
        self.wave_indices = np.random.choice(self.n_waves, self.n_slices, replace=True)
        self.return_time = return_time

        print(f'Using {self.n_slices} background segments and {self.n_waves} waveforms...')
        self.n_pos = self.n_slices
        self.n_neg = self.n_slices
        print(f'Total n_samples: {self.n_pos + self.n_neg}')

        self.rel_inj_t = np.round(np.random.uniform(0.5, 0.7, self.n_pos), 3) + 0.125
        self.injection_times = np.random.randint(0, max_seg_idx, self.n_pos) + self.rel_inj_t

    def __len__(self):
        return self.n_pos + self.n_neg

    def __getitem__(self, item):
        if item < self.n_slices:
            noise = self.slicer[item][0, :, :]
            wave = self.waves[self.wave_indices[item]]

            # shift to (0.5, 0.7)
            inj_time = self.rel_inj_t[item]
            idx_shift = 2048 - int(inj_time * 2048)

            abs_inj_time = self.injection_times[item]
            start_idx = int((abs_inj_time - inj_time) * 2048)
            end_idx = start_idx + (2560 - idx_shift)

            # inject
            sample = noise.copy()
            sample[:, start_idx:end_idx] += wave[:, idx_shift:]

            # label
            label = np.array([1, 0])

        else:
            noise = self.slicer[item - self.n_slices][0, :, :]
            sample = noise.copy()
            label = np.array([0, 1])
            abs_inj_time = -1
            inj_time = -1

        if self.return_time:
            return torch.from_numpy(sample).to(dtype=torch.float32), \
                   torch.from_numpy(label).to(dtype=torch.float32), \
                   abs_inj_time
        else:
            return torch.from_numpy(sample).to(dtype=torch.float32), \
                   torch.from_numpy(label).to(dtype=torch.float32),


class SlicerDatasetSNR(Dataset):
    """
    This class builds upon the SlicerDataset class and expands it using injection parameters to estimate the SNR of the
    resulting waveforms.
    """
    def __init__(self, background_hdf, injections_npy, slice_len=int(3.25 * 2048), slice_stride=int(2.5 * 2048),
                 max_seg_idx=3, return_time=True, injections_hdf=None,
                 min_snr=0, max_snr=None, p_augment=0.1):
        self.slicer = Slicer(background_hdf, step_size=slice_stride, window_size=slice_len)
        self.n_slices = len(self.slicer)
        self.waves = np.load(injections_npy, mmap_mode='r')
        self.n_waves = self.waves.shape[0]

        self.return_time = return_time
        self.p_augment = p_augment

        self.n_pos = int(1 * self.n_slices)
        self.n_neg = int(1 * self.n_slices)

        print(f'Using {self.n_slices} background segments and {self.n_waves} waveforms...')
        print(f'Total n_samples: {self.n_pos + self.n_neg}')

        self.wave_indices = np.random.choice(self.n_waves, self.n_pos, replace=True)
        self.segment_indices = np.random.randint(0, max_seg_idx, self.n_pos)
        self.injection_times = np.random.uniform(0.5, 0.7, self.n_pos) + 0.125

        if injections_hdf is not None:
            padding_start, padding_end = 0, 0
            dur, idxs = find_injection_times([background_hdf],
                                                  injections_hdf,
                                                  padding_start=padding_start,
                                                  padding_end=padding_end)
            inj_params = {}
            with h5py.File(injections_hdf, 'r') as fp:
                inj_params['tc'] = fp['tc'][()][idxs]
                inj_params['distance'] = fp['distance'][()][idxs]
                inj_params['mass1'] = fp['mass1'][()][idxs]
                inj_params['mass2'] = fp['mass2'][()][idxs]
                inj_params['inclination'] = fp['inclination'][()][idxs]
            self.inj_params = inj_params
        else:
            self.inj_params = None

        if max_snr is not None and self.inj_params is not None:
            # step 1: compute ~SNR for all waveforms
            Mchirp = mchirp(self.inj_params['mass1'], self.inj_params['mass2'])
            ds = np.float32(self.inj_params['distance'])
            inc = np.float32(self.inj_params['inclination'])
            s = (2626.556 / ds) * (Mchirp ** (5 / 7.1))
            snr = s + (s * (0.3 * np.cos(2 * inc) - 0.3))
            self.snr = snr

            # step 2: find where snr <= max_snr and >= min_snr
            valid_indices = np.where((snr <= max_snr) & (snr >= min_snr))[0]
            assert np.all(valid_indices >= 0) and np.all(
                valid_indices <= self.n_waves), "self.rad has higher values than self.n_waves"
            self.wave_indices = np.random.choice(valid_indices, self.n_pos, replace=True)

    def set_snr_range(self, min_snr, max_snr):
        print(f'Updating snr range to ({min_snr}, {max_snr})')
        valid_indices = np.where((self.snr <= max_snr) &
                                 (self.snr >= min_snr))[0]
        assert np.all(valid_indices >= 0) and np.all(valid_indices <= self.n_waves),\
            "self.snr has more values than self.n_waves"
        self.wave_indices = np.random.choice(valid_indices, self.n_pos, replace=True)
        assert self.wave_indices.shape[0] == self.n_pos, f"could not get {self.n_pos} waveforms"

    def __len__(self):
        return self.n_pos + self.n_neg

    def __getitem__(self, item):
        p_aug = np.random.uniform()
        rand_item = np.random.randint(0, self.n_slices)

        if item < self.n_pos:
            noise = self.slicer[item][0, :, :]
            if p_aug <= self.p_augment:
                noise[1, :] = self.slicer[rand_item][0, 1, :]
            wave = self.waves[self.wave_indices[item]]

            # shift to (0.5, 0.7)
            inj_time = self.injection_times[item]
            idx_shift = 2048 - int(inj_time * 2048)

            seg_idx = self.segment_indices[item]
            inj_time += seg_idx
            start_idx = seg_idx * 2048
            end_idx = start_idx + (2560 - idx_shift)

            # inject
            sample = noise.copy()
            sample[:, start_idx:end_idx] += wave[:, idx_shift:]

            # label
            label = np.array([1, 0])
        else:
            sample = self.slicer[item - self.n_slices][0, :, :]
            if p_aug <= self.p_augment:
                sample[1, :] = self.slicer[rand_item][0, 1, :]
            label = np.array([0, 1])
            inj_time = -1
        if self.return_time:
            return torch.from_numpy(sample).to(dtype=torch.float32), \
                   torch.from_numpy(label).to(dtype=torch.float32), \
                   inj_time
        else:
            return torch.from_numpy(sample).to(dtype=torch.float32), \
                   torch.from_numpy(label).to(dtype=torch.float32)

