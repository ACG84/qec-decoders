"""Shot-level parallelism: shots are independent, so a batch splits across processes."""
from multiprocessing import Pool
import numpy as np

_DEC = None


def _init(dec):
    global _DEC
    _DEC = dec


def _work(chunk):
    return _DEC.decode_batch(chunk)


def parallel_decode(decoder, syndromes, workers=2, chunks_per_worker=4):
    """Decode a batch with `workers` processes. Returns the same array as decoder.decode_batch."""
    parts = np.array_split(syndromes, workers * chunks_per_worker)
    with Pool(workers, initializer=_init, initargs=(decoder,)) as pool:
        return np.concatenate(pool.map(_work, parts))
