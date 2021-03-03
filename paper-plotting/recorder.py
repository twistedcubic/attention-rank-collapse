from contextlib import contextmanager
import torch


@contextmanager
def recorder():
    assert '_record' not in locals() or '_record' not in globals()
    global _record
    _record = {}
    yield _record
    del _record

def record_variable(name, x, *, ignore_if_not_recording = False):
    if '_record' not in globals():
        if ignore_if_not_recording:
            return
        raise Exception("`record_variable` should be used within a `recorder` context.")
    global _record
    if name not in _record:
        _record[name] = []

    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()

    _record[name].append(x)

