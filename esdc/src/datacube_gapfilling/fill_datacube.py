"""Fill gaps in a preprocessed ESDC zarr using a trained MaskFeatGapFiller checkpoint.

The script iterates through the data cube one zarr chunk at a time, applies the
pre-trained ViViT model (gap_only_masking=True) to reconstruct missing LST values,
and writes a gap-free cube to no_gaps_<start>_<end>.zarr in the same directory.

Dataset parameters (variable name, block size, value range, …) are read from
meta.json in the data directory, which is written by create_esdc_dataset.py.

--time_span controls both what gets gap-filled and, if preprocessing is needed,
what time window is downloaded and preprocessed. preprocessed_<start>_<end>.zarr
is created automatically when absent; otherwise it is reused as-is. If
--time_span is omitted, the full time range of the single existing
preprocessed_*.zarr is used.

Multiple calls with different --time_span values produce separate output zarrs
that coexist in the data directory.

Run from project root:
    PYTHONPATH=. python esdc/src/datacube_gapfilling/fill_datacube.py \\
        --time_span 2005-06-01:2005-07-31 \\
        --device cpu \\
        --batch_size 4
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Optional

import numpy as np
import torch
import xarray as xr

repo_root = os.environ.get(
    'GAP_AWARE_ROOT',
    os.path.join(os.path.dirname(__file__), '..', '..', '..'),
)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from model_gap_aware.maskfeat import MaskFeatGapFiller
from patchfill.completion import fill_dataset
from patchfill.preprocessing import preprocess

_DEFAULT_DATA_DIR = os.environ.get(
    'ESDC_DATA_DIR',
    os.path.join(repo_root, 'esdc', 'data', 'lst_no_syn_gaps'),
)
_DEFAULT_LOG_DIR = os.path.join(repo_root, 'esdc', 'lightning_logs', 'pretraining')


# --- Metadata ----------------------------------------------------------------

def load_meta(data_dir: str) -> dict:
    meta_path = data_dir + 'meta.json'
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f'meta.json not found in {data_dir}. '
            'Run create_esdc_dataset.py to generate it.'
        )
    with open(meta_path) as f:
        return json.load(f)


# --- Checkpoint discovery ----------------------------------------------------

def find_checkpoint(log_dir: str, checkpoint: Optional[str] = None) -> str:
    if checkpoint:
        return checkpoint
    pattern = os.path.join(log_dir, '*', 'checkpoints', '*.ckpt')
    ckpt_files = glob.glob(pattern)
    if not ckpt_files:
        raise FileNotFoundError(f'No checkpoints found under {log_dir}')

    def _mae_val(path: str) -> float:
        m = re.search(r'MAE_val=([0-9]+(?:\.[0-9]+)?)', os.path.basename(path))
        return float(m.group(1)) if m else float('inf')

    return min(ckpt_files, key=_mae_val)


# --- Preprocessing guard -----------------------------------------------------

def _find_preprocessed(data_dir: str, start: str, end: str) -> Optional[str]:
    """Return the filename of a preprocessed zarr that covers [start, end], or None.

    Checks the exact-match name first, then scans for any preprocessed_*.zarr
    whose date range (parsed from the filename) fully contains [start, end].
    """
    exact = f'preprocessed_{start}_{end}.zarr'
    if os.path.exists(data_dir + exact):
        return exact

    req_start = np.datetime64(start, 'D')
    req_end   = np.datetime64(end,   'D')
    pattern   = re.compile(r'^preprocessed_(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})\.zarr$')
    for candidate in sorted(glob.glob(data_dir + 'preprocessed_*.zarr')):
        m = pattern.match(os.path.basename(candidate))
        if not m:
            continue
        ex_start = np.datetime64(m.group(1), 'D')
        ex_end   = np.datetime64(m.group(2), 'D')
        if ex_start <= req_start and ex_end >= req_end:
            return os.path.basename(candidate)
    return None


# --- Model wrapping ----------------------------------------------------------

def make_predict_fn(model: MaskFeatGapFiller, device: torch.device):
    """Return predict_fn(batch_np) -> pred_np for use with fill_dataset.

    batch_np: (B, T, H, W, C) channels-last float32
    pred_np:  (B, C_out, H, W) channels-first float32
    """
    model.eval()

    @torch.no_grad()
    def predict_fn(batch_np: np.ndarray) -> np.ndarray:
        # (B, T, H, W, C) -> (B, T, C, H, W) for ViViT
        x = torch.from_numpy(
            np.ascontiguousarray(batch_np.transpose(0, 1, 4, 2, 3)).astype(np.float32)
        ).to(device)
        pred, _ = model(x, gap_only_masking=True)  # (B, C_out, H, W)
        return pred.cpu().numpy()

    return predict_fn


# --- CLI ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Fill ESDC gaps with a ViViT checkpoint')
    p.add_argument(
        '--data_dir', default=_DEFAULT_DATA_DIR,
        help='Dataset directory containing meta.json and preprocessed_*.zarr',
    )
    p.add_argument(
        '--log_dir', default=_DEFAULT_LOG_DIR,
        help='Lightning logs directory for checkpoint auto-discovery',
    )
    p.add_argument(
        '--checkpoint', default=None,
        help='Explicit checkpoint path (overrides auto-discovery via --log_dir)',
    )
    p.add_argument(
        '--time_span', default=None,
        help="Time span for gap-filling, e.g. '2005-06-01:2005-07-31'. "
             "preprocessed_<start>_<end>.zarr is created automatically if absent "
             "or if it does not fully cover the requested range. "
             "If omitted, the full time range of the single existing "
             "preprocessed_*.zarr is used.",
    )
    p.add_argument(
        '--device', default='cpu',
        help='PyTorch device string: cpu, cuda, cuda:0, mps, ...',
    )
    p.add_argument(
        '--batch_size', type=int, default=32,
        help='Number of spatiotemporal patches per model call',
    )
    p.add_argument(
        '--no_denorm', action='store_true',
        help='Skip reverse normalisation (output stays in [0, 1])',
    )
    return p.parse_args()


# --- Entry point -------------------------------------------------------------

def main() -> None:
    args = parse_args()

    data_dir = args.data_dir
    if not data_dir.endswith('/'):
        data_dir += '/'

    meta        = load_meta(data_dir)
    var         = meta['var']
    aux_vars    = meta['aux_vars']
    block_size  = tuple(meta['block_size'])
    patch_overlap = meta['patch_overlap']

    if args.time_span is None:
        candidates = sorted(glob.glob(data_dir + 'preprocessed_*.zarr'))
        if len(candidates) == 0:
            raise FileNotFoundError(
                'No preprocessed_*.zarr found. Provide --time_span to download and preprocess data.'
            )
        if len(candidates) > 1:
            names = [os.path.basename(c) for c in candidates]
            raise ValueError(
                f'Multiple preprocessed zarrs found: {names}. '
                'Provide --time_span to select a specific one.'
            )
        preprocessed_name = os.path.basename(candidates[0])
        preprocessed_path = candidates[0]
        ds = xr.open_zarr(preprocessed_path)
        times = ds.time.values
        start = str(np.datetime64(times[0], 'D'))
        end   = str(np.datetime64(times[-1], 'D'))
        time_slice = None
        print(f'No --time_span provided; using full {preprocessed_name} range: {start} – {end}')
    else:
        start, end = args.time_span.split(':')
        time_slice = slice(start, end)
        preprocessed_name = _find_preprocessed(data_dir, start, end)
        if preprocessed_name is None:
            preprocessed_name = f'preprocessed_{start}_{end}.zarr'
            print(f'Preprocessing {var} for {start} – {end} ...')
            _norm_range = meta.get('valid_range') or meta['computed_range']
            preprocess(
                var=var,
                output_path=data_dir,
                norm_range=_norm_range,
                block_size=block_size,
                patch_overlap=patch_overlap,
                aux_vars=aux_vars,
                time_span=slice(start, end),
                valid_range=meta.get('valid_range'),
                preprocessed_name=preprocessed_name,
            )
        else:
            print(f'{preprocessed_name} covers requested range — skipping preprocessing.')

    ckpt_path = find_checkpoint(args.log_dir, args.checkpoint)
    print(f'Loading checkpoint: {ckpt_path}')

    device = torch.device(args.device)
    model = MaskFeatGapFiller.load_from_checkpoint(ckpt_path, map_location=device)
    model.to(device)

    predict_fn = make_predict_fn(model, device)

    output_name = f'no_gaps_{start}_{end}.zarr'
    _norm_range = meta.get('valid_range') or meta['computed_range']
    denorm_range = None if args.no_denorm else tuple(_norm_range)
    print(f'Starting gap-filling from {preprocessed_name}')
    print(f'  time_span     : {start} – {end}')
    print(f'  block_size    : {block_size}')
    print(f'  patch_overlap : {patch_overlap}')
    print(f'  batch_size    : {args.batch_size}')
    print(f'  denormalize   : {denorm_range is not None}  (norm_range={denorm_range})')
    print(f'  output        : {output_name}')

    fill_dataset(
        predict_fn=predict_fn,
        path=data_dir,
        var=var,
        aux_vars=aux_vars,
        block_size=block_size,
        patch_overlap=patch_overlap,
        norm_range=denorm_range,
        batch_size=args.batch_size,
        output_name=output_name,
        time_slice=time_slice,
        preprocessed_name=preprocessed_name,
    )

    print(f'Done. Gap-free cube written to {data_dir}{output_name}')


if __name__ == '__main__':
    main()
