from collections import defaultdict
from contextlib import contextmanager
import os
import tempfile
import typing as tp
import torch
import julius
from pathlib import Path
from contextlib import contextmanager

def convert_audio_channels(wav, channels=2):
    if wav.ndim == 1:
        src_channels = 1
    else:
        src_channels = wav.shape[-2]

    if src_channels == channels:
        pass
    elif channels == 1:
        if src_channels > 1:
            wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        wav = wav.expand(-1, channels, -1)
    elif src_channels >= channels:
        wav = wav[..., :channels, :]
    else:
        raise ValueError('Input audio channel count is less than target and not mono, cannot convert.')
    return wav

def convert_audio(wav, from_samplerate, to_samplerate, channels):
    wav = convert_audio_channels(wav, channels)

    if from_samplerate != to_samplerate:
        wav = julius.resample_frac(wav, from_samplerate, to_samplerate)
    return wav

def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}

@contextmanager
def swap_state(model, state):
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state, strict=False)
    try:
        yield
    finally:
        model.load_state_dict(old_state)

@contextmanager
def temp_filenames(count: int, delete=True):
    names = []
    try:
        for _ in range(count):
            names.append(tempfile.NamedTemporaryFile(delete=False).name)
        yield names
    finally:
        if delete:
            for name in names:
                os.unlink(name)

def center_trim(tensor: torch.Tensor, reference: tp.Union[torch.Tensor, int]):
    ref_size: int
    if isinstance(reference, torch.Tensor):
        ref_size = reference.size(-1)
    else:
        ref_size = reference
    delta = tensor.size(-1) - ref_size
    if delta < 0:
        raise ValueError("输入张量必须大于参考张量。")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor

def EMA(beta: float = 1):
    fix: tp.Dict[str, float] = defaultdict(float)
    total: tp.Dict[str, float] = defaultdict(float)

    def _update(metrics: dict, weight: float = 1) -> dict:
        nonlocal total, fix
        for key, value in metrics.items():
            total[key] = total[key] * beta + weight * float(value)
            fix[key] = fix[key] * beta + weight
        return {key: tot / fix[key] for key, tot in total.items()}
    return _update

class DummyPoolExecutor:
    class DummyResult:
        def __init__(self, func, *args, **kwargs):
            self.func = func
            self.args = args
            self.kwargs = kwargs

        def result(self):
            return self.func(*self.args, **self.kwargs)

    def __init__(self, workers=0):
        pass

    def submit(self, func, *args, **kwargs):
        return DummyPoolExecutor.DummyResult(func, *args, **kwargs)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return

def new_sdr(references, estimates):
    assert references.dim() == 4
    assert estimates.dim() == 4
    delta = 1e-7
    num = torch.sum(torch.square(references), dim=(2, 3))
    den = torch.sum(torch.square(references - estimates), dim=(2, 3))
    num += delta
    den += delta
    scores = 10 * torch.log10(num / den)
    return scores

def load_model(model, checkpoint_path):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint file not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    except Exception as e:
        raise RuntimeError(f"Error loading checkpoint file: {e}")

    if checkpoint is None:
        raise ValueError(f"Checkpoint file {checkpoint_path} is empty after loading")

    state_dict = None
    if isinstance(checkpoint, dict):
        for key in ['best_state', 'state', 'state_dict']:
            if key in checkpoint and checkpoint[key] is not None:
                state_dict = checkpoint[key]
                print(f"Using '{key}' key from checkpoint")
                break
        if state_dict is None:
            state_dict = checkpoint
            print("Using entire checkpoint as state dict")
    else:
        state_dict = checkpoint
        print("Checkpoint is not dict type, using directly as state dict")

    if state_dict is None:
        raise ValueError("Cannot extract valid state dict from checkpoint")

    new_state_dict = {}
    try:
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
    except Exception as e:
        raise RuntimeError(f"Error processing state dict: {e}")

    try:
        model.load_state_dict(new_state_dict)
        print("Model weights loaded successfully")
    except Exception as e:
        print(f"Error loading state dict: {e}")
        print("Available keys in checkpoint:", list(state_dict.keys()))
        print("Required keys by model:", list(model.state_dict().keys()))
        raise

    return model


