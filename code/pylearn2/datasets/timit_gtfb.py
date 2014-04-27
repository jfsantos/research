"""
Pylearn2 wrapper for the TIMIT dataset
"""
__authors__ = ["Vincent Dumoulin"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Laurent Dinh", "Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Vincent Dumoulin"
__email__ = "dumouliv@iro"

import os.path
import functools
import numpy
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.datasets.dataset import Dataset
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace
from pylearn2.utils import serial
from pylearn2.utils import safe_zip
from research.code.scripts.segmentaxis import segment_axis
from research.code.pylearn2.utils.iteration import FiniteDatasetIterator
import scipy.stats
import itertools
from brian import hertz, khertz
from brian.hears import Sound, erbspace, Gammatone

class TIMITGTFB(Dataset):
    """
    Frame-based TIMIT dataset
    """
    _default_seed = (17, 2, 946)
    
    # Mean and standard deviation of the acoustic samples from the whole
    # dataset (train, valid, test).
    _mean = 2639012.3033423889
    _std = 23348287.541279223

    def __init__(self, which_set, frame_length, overlap=0,
                 n_channels=64, frames_per_example=1, start=0,
                 stop=None, audio_only=False, n_prev_phones=0,
                 n_next_phones=0, samples_to_predict=1,
                 filter_fn=None, rng=_default_seed, gtfb_data_path='/home/jfsantos/data/pylearn2data/timit/readable'):
        """
        Parameters
        ----------
        which_set : str
            Either "train", "valid" or "test"
        frame_length : int
            Number of acoustic samples contained in a frame
        overlap : int, optional
            Number of overlapping acoustic samples for two consecutive frames.
            Defaults to 0, meaning frames don't overlap.
        frames_per_example : int, optional
            Number of frames in a training example. Defaults to 1.
        start : int, optional
            Starting index of the sequences to use. Defaults to 0.
        stop : int, optional
            Ending index of the sequences to use. Defaults to `None`, meaning
            sequences are selected all the way to the end of the array.
        audio_only : bool, optional
            Whether to load only the raw audio and no auxiliary information.
            Defaults to `False`.
        rng : object, optional
            A random number generator used for picking random indices into the
            design matrix when choosing minibatches.
        """
        self.frame_length = frame_length
        self.overlap = overlap
        self.frames_per_example = frames_per_example
        self.offset = self.frame_length - self.overlap
        self.audio_only = audio_only
        self.n_prev_phones = n_prev_phones
        self.n_next_phones = n_next_phones
        self.samples_to_predict = samples_to_predict
        self.n_channels = n_channels
        # RNG initialization
        if hasattr(rng, 'random_integers'):
            self.rng = rng
        else:
            self.rng = numpy.random.RandomState(rng)

        self.fc = erbspace(80*hertz, 5*khertz, self.n_channels)

        # Load data from disk
        self._load_data(which_set, gtfb_data_path)
        # Standardize data
        for i, sequence in enumerate(self.raw_wav):
            self.raw_wav[i] = (sequence - TIMITGTFB._mean) / TIMITGTFB._std

        if filter_fn is not None:
            filter_fn = eval(filter_fn)
            indexes = filter_fn(self.speaker_info_list[self.speaker_id])
            self.raw_wav = self.raw_wav[indexes]
            if not self.audio_only:
                self.phones = self.phones[indexes]

        # Slice data
        if stop is not None:
            self.raw_wav = self.raw_wav[start:stop]
            if not self.audio_only:
                self.phones = self.phones[start:stop]
        else:
            self.raw_wav = self.raw_wav[start:]
            if not self.audio_only:
                self.phones = self.phones[start:]

        examples_per_sequence = [0]
        self.phone_rel_dur = []

        for sequence_id, samples_sequence in enumerate(self.raw_wav):
            if not self.audio_only:
                tot_n_frames = samples_sequence.shape[0]
                # Phones segmentation
                phones_sequence = self.phones[sequence_id]
                phone_list = numpy.asarray([k for k, g in itertools.groupby(phones_sequence)])
                phone_duration = [len(list(g)) for k, g in itertools.groupby(phones_sequence)]
                phone_position = numpy.cumsum(phone_duration)
                frame_position = numpy.arange(0, tot_n_frames*self.overlap, self.overlap)
                seq_phones = numpy.empty((tot_n_frames, 1+self.n_prev_phones+self.n_next_phones), dtype=int)
                phone_rel_dur = numpy.empty(tot_n_frames, dtype=float)
                for frame in range(tot_n_frames):
                    cur_phone_idx = (frame_position[frame] < phone_position).argmax()
                    if cur_phone_idx == 0:
                        phone_rel_dur[frame] = frame_position[frame]/float(phone_duration[cur_phone_idx])
                    else:
                        phone_rel_dur[frame] = (frame_position[frame] - phone_position[cur_phone_idx-1])/float(phone_duration[cur_phone_idx])
                    if self.n_prev_phones > 0:
                        if cur_phone_idx - self.n_prev_phones < 0:
                            seq_phones[frame,0:self.n_prev_phones] = 5 # code for silent frame
                        else:
                            seq_phones[frame,0:self.n_prev_phones] = phone_list[cur_phone_idx-self.n_prev_phones:cur_phone_idx] # prev phones
                    if self.n_next_phones > 0:
                        if cur_phone_idx + self.n_next_phones >= len(phone_list):
                            seq_phones[frame,-self.n_next_phones:] = 5 # code for silent frame
                        else:
                            seq_phones[frame,-self.n_next_phones] = phone_list[cur_phone_idx+1:cur_phone_idx+self.n_next_phones+1] #next phones
                    seq_phones[frame,self.n_prev_phones] = phone_list[cur_phone_idx]
                self.phone_rel_dur.append(phone_rel_dur)
                self.phones[sequence_id] = seq_phones

            # TODO: look at this, does it force copying the data?
            # Sequence segmentation
            # s = Sound(samples_sequence, samplerate=16*khertz)
            # fb = Gammatone(s, self.fc)
            # y = fb.process()
            # channel_energy = []
            # Compute energy per channel
            # for ch in range(self.n_channels):
            #     y_ch = segment_axis(y[:,ch], frame_length, overlap)
            #     y_energy = numpy.sum(y_ch**2, axis=1)
            #     channel_energy.append(y_energy)
            
            # channel_energy = numpy.vstack(channel_energy).T
            # channel_energy = samples_sequence
            # if self.n_next_phones == 0:
            #     self.raw_wav[sequence_id] = channel_energy[self.n_prev_phones:]
            # else:
            #     self.raw_wav[sequence_id] = channel_energy[self.n_prev_phones:-self.n_next_phones]

            # TODO: change me
            # Generate features/targets/phones/phonemes/words map
            num_frames = samples_sequence.shape[0]-(self.n_prev_phones+self.n_next_phones)
            num_examples = num_frames - self.frames_per_example
            examples_per_sequence.append(num_examples)

        self.cumulative_example_indexes = numpy.cumsum(examples_per_sequence)
        self.samples_sequences = self.raw_wav
#        numpy.save('%s_gtfb_%sch.npy'%(which_set, str(self.n_channels)), self.samples_sequences)
        if not self.audio_only:
            self.phones_sequences = self.phones
        self.num_examples = self.cumulative_example_indexes[-1]

        # DataSpecs
        features_space = VectorSpace(
            dim=self.n_channels * self.frames_per_example
        )
        features_source = 'features'
        def features_map_fn(indexes):
            rval = []
            for sequence_index, example_index in self._fetch_index(indexes):
                rval.append(self.samples_sequences[sequence_index][example_index:example_index + self.frames_per_example].ravel())
            return rval

        targets_space = VectorSpace(dim=self.n_channels)
        targets_source = 'targets'
        def targets_map_fn(indexes):
            rval = []
            for sequence_index, example_index in self._fetch_index(indexes):
                rval.append(self.samples_sequences[sequence_index][example_index + self.frames_per_example])
            return rval

        space_components = [features_space, targets_space]
        source_components = [features_source, targets_source]
        map_fn_components = [features_map_fn, targets_map_fn]
        batch_components = [None, None]

        if not self.audio_only:
            num_phones = numpy.max([numpy.max(sequence) for sequence
                                    in self.phones]) + 1
            phones_space = IndexSpace(max_labels=num_phones, dim=1+self.n_prev_phones+self.n_next_phones,
                                      dtype=str(self.phones_sequences[0].dtype))
            phones_source = 'phones'
            def phones_map_fn(indexes):
                rval = []
                for sequence_index, example_index in self._fetch_index(indexes):
                    rval.append(self.phones_sequences[sequence_index][example_index].ravel())
                return rval

            phone_rel_dur_space = VectorSpace(dim=1)
            phone_rel_dur_source = 'phone_rel_dur'
            def phone_rel_dur_map_fn(indexes):
                rval = []
                for sequence_index, example_index in self._fetch_index(indexes):
                    rval.append(self.phone_rel_dur[sequence_index][example_index])
                return rval

            space_components.extend([phones_space, phone_rel_dur_space])
            source_components.extend([phones_source, phone_rel_dur_source])
            map_fn_components.extend([phones_map_fn, phone_rel_dur_map_fn])
            batch_components.extend([None, None])

        space = CompositeSpace(space_components)
        source = tuple(source_components)
        self.data_specs = (space, source)
        self.map_functions = tuple(map_fn_components)
        self.batch_buffers = batch_components

        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('shuffled_sequential')
        self._iter_data_specs = (CompositeSpace((features_space,
                                                 targets_space)),
                                 (features_source, targets_source))

    def _fetch_index(self, indexes):
        digit = numpy.digitize(indexes, self.cumulative_example_indexes) - 1
        return zip(digit,
                   numpy.array(indexes) - self.cumulative_example_indexes[digit])

    def _load_data(self, which_set, gtfb_data_path):
        """
        Load the TIMIT data from disk.

        Parameters
        ----------
        which_set : str
            Subset of the dataset to use (either "train", "valid" or "test")
        """
        # Check which_set
        if which_set not in ['train', 'valid', 'test']:
            raise ValueError(which_set + " is not a recognized value. " +
                             "Valid values are ['train', 'valid', 'test'].")

        # Create file paths
        timit_base_path = os.path.join(os.environ["PYLEARN2_DATA_PATH"],
                                       "timit/readable")
        speaker_info_list_path = os.path.join(timit_base_path, "spkrinfo.npy")
        speaker_features_list_path = os.path.join(timit_base_path,
                                                  "spkr_feature_names.pkl")
        speaker_id_list_path = os.path.join(timit_base_path,
                                            "speakers_ids.pkl")
        raw_wav_path = os.path.join(gtfb_data_path, which_set + "_gtfb_64ch.npy")
        phones_path = os.path.join(timit_base_path,
                                     which_set + "_x_phones.npy")
        speaker_path = os.path.join(timit_base_path,
                                    which_set + "_spkr.npy")

        # Load data. For now most of it is not used, as only the acoustic
        # samples are provided, but this is bound to change eventually.
        # Global data
        if not self.audio_only:
            self.speaker_info_list = serial.load(
                speaker_info_list_path
            ).tolist().toarray()
            self.speaker_id_list = serial.load(speaker_id_list_path)
            self.speaker_features_list = serial.load(speaker_features_list_path)
        # Set-related data
        self.raw_wav = serial.load(raw_wav_path)
        if not self.audio_only:
            self.phones = serial.load(phones_path)
            self.speaker_id = numpy.asarray(serial.load(speaker_path), 'int')

    def _validate_source(self, source):
        """
        Verify that all sources in the source tuple are provided by the
        dataset. Raise an error if some requested source is not available.

        Parameters
        ----------
        source : `tuple` of `str`
            Requested sources
        """
        for s in source:
            try:
                self.data_specs[1].index(s)
            except ValueError:
                raise ValueError("the requested source named '" + s + "' " +
                                 "is not provided by the dataset")

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.

        .. note::

            Once again, this is very hacky, as the data is not stored that way
            internally. However, the data that's returned by `TIMIT.get()`
            _does_ respect those data specs.
        """
        return self.data_specs

    def get(self, source, indexes):
        """
        .. todo::

            WRITEME
        """
        if type(indexes) is slice:
            indexes = numpy.arange(indexes.start, indexes.stop)
        self._validate_source(source)
        rval = []
        for so in source:
            batch = self.map_functions[self.data_specs[1].index(so)](indexes)
            batch_buffer = self.batch_buffers[self.data_specs[1].index(so)]
            dim = self.data_specs[0].components[self.data_specs[1].index(so)].dim
            if batch_buffer is None or batch_buffer.shape != (len(batch), dim):
                batch_buffer = numpy.zeros((len(batch), dim),
                                           dtype=batch[0].dtype)
            for i, example in enumerate(batch):
                batch_buffer[i] = example
            rval.append(batch_buffer)
        return tuple(rval)

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):
        """
        .. todo::

            WRITEME
        """
        if data_specs is None:
            data_specs = self._iter_data_specs

        # If there is a view_converter, we have to use it to convert
        # the stored data for "features" into one that the iterator
        # can return.
        space, source = data_specs
        if isinstance(space, CompositeSpace):
            sub_spaces = space.components
            sub_sources = source
        else:
            sub_spaces = (space,)
            sub_sources = (source,)

        convert = []
        for sp, src in safe_zip(sub_spaces, sub_sources):
            convert.append(None)

        # TODO: Refactor
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        if num_batches is None:
            num_batches = getattr(self, '_iter_num_batches', None)
        if rng is None and mode.stochastic:
            rng = self.rng
        return FiniteDatasetIterator(self,
                                     mode(self.num_examples, batch_size,
                                          num_batches, rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)

def male_speakers(spkrinfo):
    return spkrinfo[:,24] == 1
        
def female_speakers(spkrinfo):
    return spkrinfo[:,25] == 1

def dialect_region(spkrinfo, dr):
    return spkrinfo[:,dr] == 1

if __name__ == "__main__":
    from sys import argv

    valid_timit = TIMITGTFB(argv[1], frame_length=240, overlap=120,
                            frames_per_example=1, audio_only=False, n_next_phones=1, n_prev_phones=1)
