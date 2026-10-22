"""
Create an ESDC pre-training or gap-filling dataset using patch-fill.

All dataset parameters are exposed as CLI arguments with defaults matching
the standard LST configuration. After dataset creation a meta.json file is
written to the output directory so downstream scripts (e.g. fill_datacube.py)
can read the parameters without re-specifying them.

Run from the project root:

    # Pre-training (no synthetic gaps):
    PYTHONPATH=. python esdc/src/download_data/create_esdc_dataset.py \
        --output_path esdc/data/lst_no_syn_gaps/ \
        --time_start 2003-01-05 \
        --time_end   2010-12-31 \
        --computed_ranges_lookup esdc/data/computed_ranges.json

    # Gap-filling evaluation (synthetic gaps, no train/test split):
    PYTHONPATH=. python esdc/src/download_data/create_esdc_dataset.py \
        --output_path esdc/data/lst_gap_filling/ \
        --time_start 2011-01-01 \
        --time_end   2011-12-23 \
        --syn_gaps \
        --no_split \
        --computed_ranges_lookup esdc/data/computed_ranges.json
"""
import argparse
import json
import os

import h5py
import xarray as xr

from patchfill.preprocessing import preprocess
from patchfill.dataset_building import create_dataset
from patchfill.computed_ranges import ensure_computed_ranges


def main():
    parser = argparse.ArgumentParser(
        description='Create an ESDC pre-training or gap-filling dataset using patch-fill.'
    )

    # --- required: dataset location and time range ---
    parser.add_argument(
        '--output_path', required=True,
        help='Directory where zarr intermediates and dataset.h5 are written.'
    )
    parser.add_argument(
        '--time_start', required=True,
        help='Start of the time range (ISO date, e.g. 2001-01-01).'
    )
    parser.add_argument(
        '--time_end', required=True,
        help='End of the time range (ISO date, e.g. 2019-12-31).'
    )

    # --- dataset-mode flags ---
    parser.add_argument(
        '--syn_gaps', action='store_true', default=False,
        help=(
            'Inject synthetic gaps into the middle input frame '
            '(gap-filling dataset). Default: no gaps (pre-training dataset).'
        )
    )
    parser.add_argument(
        '--max_gap_portion', type=float, default=0.2,
        help='Discard patches whose target has more than this fraction of NaN. Default: 0.2.'
    )
    parser.add_argument(
        '--gap_donor_min_ratio', type=float, default=0.05,
        help='Minimum combined gap fraction of the donor time series (inclusive). Default: 0.05.'
    )
    parser.add_argument(
        '--gap_donor_max_ratio', type=float, default=0.85,
        help='Maximum combined gap fraction of the donor time series (inclusive). Default: 0.85.'
    )
    parser.add_argument(
        '--no_split', action='store_true', default=False,
        help=(
            'Skip train/test splitting and write all samples to the train split. '
            'Use for evaluation-only datasets (e.g. gap-filling eval) where a '
            'held-out test set is not needed.'
        )
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help=(
            'Seed for the train/test block-split hash. Default: None (legacy '
            'unseeded behavior, byte-identical to pre-existing datasets). Pass the '
            'same value to two dataset-creation runs over the same time range/'
            'block_size to guarantee an identical split (e.g. pre-training vs. '
            'gap-filling training).'
        )
    )

    # --- fixed dataset parameters (defaults match the standard LST configuration) ---
    parser.add_argument(
        '--var', default='land_surface_temperature',
        help='Primary ESDC variable name. Default: land_surface_temperature.'
    )
    parser.add_argument(
        '--valid_range', nargs=2, type=float, default=None,
        metavar=('MIN', 'MAX'),
        help=(
            'Optional validity limits for the primary variable. '
            'Values outside [MIN MAX] are set to NaN before normalization. '
            'When provided, also used as the normalization range instead of the computed range.'
        )
    )
    parser.add_argument(
        '--aux_vars',
        default='[{"var": "air_temperature_2m"}]',
        help=(
            'Auxiliary variables as a JSON array of {"var": ...} objects. '
            'An optional "valid_range": [min, max] field applies cleaning and '
            'overrides the computed normalization range for that variable. '
            'Default: air_temperature_2m.'
        )
    )
    parser.add_argument(
        '--computed_ranges_lookup', required=True,
        help='Path to computed_ranges.json lookup file.'
    )
    parser.add_argument(
        '--block_size', nargs=3, type=int, default=[5, 64, 64],
        metavar=('T', 'H', 'W'),
        help='Spatiotemporal block size (time steps, lat pixels, lon pixels). Default: 5 64 64.'
    )
    parser.add_argument(
        '--strides', nargs=3, type=int, default=[3, 16, 16],
        metavar=('T', 'H', 'W'),
        help='Sampling strides (time, lat, lon). Default: 3 16 16.'
    )
    parser.add_argument(
        '--patch_overlap', type=int, default=0,
        help='Spatial patch overlap in pixels. Default: 0.'
    )
    args = parser.parse_args()

    output_path = args.output_path
    if not output_path.endswith('/'):
        output_path += '/'
    os.makedirs(output_path, exist_ok=True)

    aux_vars = json.loads(args.aux_vars)

    # Compute ranges for all variables (primary + aux).
    all_vars = [args.var] + [av['var'] for av in aux_vars]
    computed = ensure_computed_ranges(
        all_vars, args.time_start, args.time_end, args.computed_ranges_lookup
    )

    primary_computed = computed[args.var]
    primary_norm_range = args.valid_range if args.valid_range is not None else primary_computed

    for av in aux_vars:
        av['computed_range'] = computed[av['var']]

    mode = 'gap-filling' if args.syn_gaps else 'pre-training'
    print(f'Creating ESDC {mode} dataset')
    print(f'  output_path    : {output_path}')
    print(f'  time_span      : {args.time_start} → {args.time_end}')
    print(f'  var            : {args.var}')
    print(f'  computed_range : {primary_computed}')
    print(f'  valid_range    : {args.valid_range}')
    print(f'  norm_range     : {primary_norm_range}')
    print(f'  block_size     : {args.block_size}')
    print(f'  strides        : {args.strides}')
    print(f'  patch_overlap  : {args.patch_overlap}')
    print(f'  syn_gaps            : {args.syn_gaps}')
    print(f'  max_gap_portion     : {args.max_gap_portion}')
    print(f'  gap_donor_min_ratio : {args.gap_donor_min_ratio}')
    print(f'  gap_donor_max_ratio : {args.gap_donor_max_ratio}')
    print(f'  no_split            : {args.no_split}')
    print(f'  seed                : {args.seed}')

    time_span = slice(args.time_start, args.time_end)

    preprocessed_name = f'preprocessed_{args.time_start}_{args.time_end}.zarr'
    preprocess(
        args.var, output_path, primary_norm_range, args.block_size, args.patch_overlap,
        aux_vars, time_span,
        valid_range=args.valid_range,
        preprocessed_name=preprocessed_name,
    )

    create_dataset(
        args.var, output_path, args.block_size, args.strides, args.patch_overlap,
        aux_vars,
        max_gap_portion=args.max_gap_portion,
        syn_gaps=args.syn_gaps,
        no_split=args.no_split,
        preprocessed_name=preprocessed_name,
        gap_donor_min_ratio=args.gap_donor_min_ratio,
        gap_donor_max_ratio=args.gap_donor_max_ratio,
        seed=args.seed,
    )

    with h5py.File(output_path + 'dataset.h5', 'r') as _h5:
        _num_samples_train = int(_h5['X_train'].shape[0])
        _num_samples_test  = int(_h5['X_test'].shape[0])

    _ds = xr.open_zarr(output_path + preprocessed_name)
    _lat_res = float(abs(float(_ds.lat[1]) - float(_ds.lat[0])))
    _lon_res = float(abs(float(_ds.lon[1]) - float(_ds.lon[0])))
    _ds.close()
    meta = {
        'var': args.var,
        'lat_res': _lat_res,
        'lon_res': _lon_res,
        'computed_range': primary_computed,
        'valid_range': args.valid_range,
        'aux_vars': aux_vars,
        'computed_ranges_lookup': args.computed_ranges_lookup,
        'block_size': args.block_size,
        'strides': args.strides,
        'patch_overlap': args.patch_overlap,
        'time_start': args.time_start,
        'time_end': args.time_end,
        'syn_gaps': args.syn_gaps,
        'max_gap_portion': args.max_gap_portion,
        'gap_donor_min_ratio': args.gap_donor_min_ratio,
        'gap_donor_max_ratio': args.gap_donor_max_ratio,
        'no_split': args.no_split,
        'seed': args.seed,
        'num_samples_train': _num_samples_train,
        'num_samples_test': _num_samples_test,
    }
    meta_path = output_path + 'meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'Metadata written to {meta_path}')

    print(f'Done. Dataset written to {output_path}dataset.h5')


if __name__ == '__main__':
    main()
