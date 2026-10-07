"""
One-time repack of the per-sample .npz latent tree into a single uncompressed
memmap, to kill the ~52ms/file zip-decompression that makes training data-bound.

Source: <SRC>/**/*.npz  each holding latent_dist (2,8,32,32) fp32 + cls
Output: <DST>/latents.npy  (N,8,32,32) fp32 memmap   (only the [0] aug slice, which
                            is the one the dataset actually uses)
        <DST>/labels.npy   (N,) int64
        <DST>/files.txt    the source file order (for traceability)

Reads/writes disjoint index ranges from parallel workers straight into the memmap
(safe: no overlapping writes). ~55 min for 1.28M files on 20 cores.
"""
import os, glob, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial

SRC = "/run/media/hoang/ssd/imagenet/latent/"
DST = "/run/media/hoang/ssd/imagenet/latent_memmap/"
NWORK = 20


def worker(rng, lat_path, lab_path, files):
    lo, hi = rng
    lat = np.load(lat_path, mmap_mode="r+")
    lab = np.load(lab_path, mmap_mode="r+")
    for i in range(lo, hi):
        d = np.load(files[i])
        lat[i] = d["latent_dist"][0]                       # (8,32,32)
        lab[i] = int(np.asarray(d["cls"]).ravel()[0])
    lat.flush(); lab.flush()
    return hi - lo


def main():
    os.makedirs(DST, exist_ok=True)
    files = sorted(glob.glob(os.path.join(SRC, "**/*.npz"), recursive=True))
    N = len(files)
    assert N > 0, f"no .npz found under {SRC}"
    # infer per-sample shape from the first file
    shp = np.load(files[0])["latent_dist"][0].shape
    print(f"repacking {N} files -> memmap (N,{','.join(map(str, shp))}) fp32", flush=True)

    with open(os.path.join(DST, "files.txt"), "w") as f:
        f.write("\n".join(files))

    lat_path = os.path.join(DST, "latents.npy")
    lab_path = os.path.join(DST, "labels.npy")
    np.lib.format.open_memmap(lat_path, mode="w+", dtype=np.float32, shape=(N, *shp))
    np.lib.format.open_memmap(lab_path, mode="w+", dtype=np.int64, shape=(N,))

    # more chunks than workers -> better load balance
    nchunks = NWORK * 8
    chunk = (N + nchunks - 1) // nchunks
    ranges = [(i, min(i + chunk, N)) for i in range(0, N, chunk)]

    done = 0
    with ProcessPoolExecutor(max_workers=NWORK) as ex:
        for c in ex.map(partial(worker, lat_path=lat_path, lab_path=lab_path, files=files), ranges):
            done += c
            print(f"repacked {done}/{N} ({100*done/N:.1f}%)", flush=True)
    print(f"DONE: {lat_path} ({N} samples). You can delete the .npz tree once verified.", flush=True)


if __name__ == "__main__":
    main()
