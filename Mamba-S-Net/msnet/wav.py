from collections import OrderedDict
import hashlib
import math
import json
import os
from pathlib import Path
import tqdm

import julius
import torch as th
import torchaudio as ta
from torch.nn import functional as F

from .utils import convert_audio_channels
from accelerate import Accelerator

accelerator = Accelerator()

MIXTURE = "mixture"
EXT = ".wav"

def _track_metadata(track, sources, normalize=True, ext=EXT):
    track_length = None
    track_samplerate = None
    mean = 0
    std = 1
    for source in sources + [MIXTURE]:
        file = track / f"{source}{ext}"
        try:
            info = ta.info(str(file))
        except RuntimeError:
            print(file)
            raise
        length = info.num_frames
        if track_length is None:
            track_length = length
            track_samplerate = info.sample_rate
        elif track_length != length:
            raise ValueError(
                f"Invalid length for file {file}: "
                f"expecting {track_length} but got {length}.")
        elif info.sample_rate != track_samplerate:
            raise ValueError(
                f"Invalid sample rate for file {file}: "
                f"expecting {track_samplerate} but got {info.sample_rate}.")
        if source == MIXTURE and normalize:
            try:
                wav, _ = ta.load(str(file))
            except RuntimeError:
                print(file)
                raise
            wav = wav.mean(0)
            mean = wav.mean().item()
            std = wav.std().item()

    return {"length": length, "mean": mean, "std": std, "samplerate": track_samplerate}

def build_metadata(path, sources, normalize=True, ext=EXT):
    meta = {}
    path = Path(path)
    pendings = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as pool:
        for root, folders, files in os.walk(path, followlinks=True):
            root = Path(root)
            if root.name.startswith('.') or folders or root == path:
                continue
            name = str(root.relative_to(path))
            pendings.append((name, pool.submit(_track_metadata, root, sources, normalize, ext)))
        for name, pending in tqdm.tqdm(pendings, ncols=120):
            meta[name] = pending.result()
    return meta

class Wavset:
    def __init__(
            self,
            root, metadata, sources,
            segment=None, shift=None, normalize=True,
            samplerate=44100, channels=2, ext=EXT):

        self.root = Path(root)
        self.metadata = OrderedDict(metadata)
        self.segment = segment
        self.shift = shift or segment
        self.normalize = normalize
        self.sources = sources
        self.channels = channels
        self.samplerate = samplerate
        self.ext = ext
        self.num_examples = []
        for name, meta in self.metadata.items():
            track_duration = meta['length'] / meta['samplerate']
            if segment is None or track_duration < segment:
                examples = 1
            else:
                examples = int(math.ceil((track_duration - self.segment) / self.shift) + 1)
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def get_file(self, name, source):
        return self.root / name / f"{source}{self.ext}"

    def __getitem__(self, index):
        for name, examples in zip(self.metadata, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            meta = self.metadata[name]
            num_frames = -1
            offset = 0
            if self.segment is not None:
                offset = int(meta['samplerate'] * self.shift * index)
                num_frames = int(math.ceil(meta['samplerate'] * self.segment))
            wavs = []
            for source in self.sources:
                file = self.get_file(name, source)
                wav, _ = ta.load(str(file), frame_offset=offset, num_frames=num_frames)
                wav = convert_audio_channels(wav, self.channels)
                wavs.append(wav)

            example = th.stack(wavs)
            example = julius.resample_frac(example, meta['samplerate'], self.samplerate)
            if self.normalize:
                example = (example - meta['mean']) / meta['std']
            if self.segment:
                length = int(self.segment * self.samplerate)
                example = example[..., :length]
                example = F.pad(example, (0, length - example.shape[-1]))
            return example

def get_wav_datasets(args):
    sig = hashlib.sha1(str(args.wav).encode()).hexdigest()[:8]
    metadata_file = Path(args.metadata) / ('wav_' + sig + ".json")
    train_path = Path(args.wav) / "train"
    valid_path = Path(args.wav) / "test"
    if not metadata_file.is_file() and accelerator.is_main_process:
        metadata_file.parent.mkdir(exist_ok=True, parents=True)
        train = build_metadata(train_path, args.sources)
        valid = build_metadata(valid_path, args.sources)
        json.dump([train, valid], open(metadata_file, "w"))
    accelerator.wait_for_everyone()

    train, valid = json.load(open(metadata_file))
    kw_cv = {}

    train_set = Wavset(train_path, train, args.sources,
                       segment=args.segment, shift=args.shift,
                       samplerate=args.samplerate, channels=args.channels,
                       normalize=args.normalize)
    valid_set = Wavset(valid_path, valid, [MIXTURE] + list(args.sources),
                       samplerate=args.samplerate, channels=args.channels,
                       normalize=args.normalize, **kw_cv)
    return train_set, valid_set



