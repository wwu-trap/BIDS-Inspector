#!/usr/bin/env python3
"""BIDS-Inspector — validate and summarise a BIDS dataset.

Scans a BIDS dataset and reports which acquisitions are present or missing
per subject/session. For each subject(/session), reports:
  - 1/0 whether both NIfTI and JSON sidecar are present
  - dim4 (4th NIfTI dimension) for functional bold and DWI scans
  - dim4_ok flag (0 when dim4 deviates from the dataset mode)
  - events.tsv presence and row count for functional bold scans
  - bval/bvec presence and direction count for DWI scans
  - dwi_ok flag (0 when dim4 / bval_n / bvec_n are inconsistent)

JSON sidecars are resolved using BIDS inheritance (file-level -> subject-level -> root-level).
Automatically merges participants.tsv when available.

Requires: nibabel, pandas
"""

import argparse
import io
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mode as stat_mode

import nibabel as nib
import pandas as pd


# ---------------------------------------------------------------------------
# Logging — writes to stdout and optionally to a log file
# ---------------------------------------------------------------------------

_log_file: io.TextIOBase | None = None


def log(msg: str = '') -> None:
    """Print to stdout and append to log file (if open)."""
    print(msg)
    if _log_file is not None:
        _log_file.write(msg + '\n')

# Modality directories that contain 4D NIfTI files where dim4 is meaningful
_DIM4_SUFFIXES = {'bold', 'dwi', 'epi'}

# Known BIDS modality directories
_MODALITY_DIRS = {'anat', 'func', 'dwi', 'fmap', 'perf', 'meg', 'eeg', 'ieeg', 'pet'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nifti_suffixes(path: Path) -> str:
    """Return the compound suffix for a NIfTI file ('.nii.gz' or '.nii')."""
    if path.name.endswith('.nii.gz'):
        return '.nii.gz'
    if path.name.endswith('.nii'):
        return '.nii'
    return ''


def _strip_nifti_ext(filename: str) -> str:
    """Remove .nii.gz or .nii extension from a filename."""
    if filename.endswith('.nii.gz'):
        return filename[:-7]
    if filename.endswith('.nii'):
        return filename[:-4]
    return filename


def _acquisition_key(modality_dir: str, nifti_filename: str) -> str:
    """Build a unique acquisition key from modality folder and BIDS filename.

    Strips subject/session entities so the key is comparable across subjects.
    Examples:
        anat, sub-01_T1w.nii.gz                                    -> anat_T1w
        func, sub-01_task-rest_run-01_bold.nii.gz                   -> func_task-rest_run-01_bold
        dwi,  sub-01_ses-01_acq-multiband_dwi.nii.gz               -> dwi_acq-multiband_dwi
    """
    stem = _strip_nifti_ext(nifti_filename)
    parts = stem.split('_')
    filtered = [p for p in parts if not p.startswith('sub-') and not p.startswith('ses-')]
    return modality_dir + '_' + '_'.join(filtered)


def _bids_suffix(acq_key: str) -> str:
    """Extract the BIDS suffix (last entity) from an acquisition key.

    E.g. 'func_task-rest_run-01_bold' -> 'bold'
    """
    return acq_key.rsplit('_', maxsplit=1)[-1]


def _wants_dim4(acq_key: str) -> bool:
    """Whether this acquisition type should have a dim4 column."""
    return _bids_suffix(acq_key) in _DIM4_SUFFIXES


def _get_dim4(nifti_path: Path) -> int | None:
    """Read the 4th dimension from a NIfTI file. Returns None on failure."""
    try:
        img = nib.load(str(nifti_path))
        shape = img.shape
        return shape[3] if len(shape) > 3 else None
    except Exception:
        return None


def _is_func_bold(acq_key: str) -> bool:
    """Whether this acquisition is a functional bold scan (expects events.tsv)."""
    return acq_key.startswith('func_') and acq_key.endswith('_bold')


def _find_events_tsv(nifti_path: Path) -> tuple[bool, int | str]:
    """Look for the BIDS events.tsv matching this NIfTI.

    Returns (exists, n_rows) where n_rows is the number of data rows
    (excluding the header), or '' if the file does not exist.
    """
    stem = _strip_nifti_ext(nifti_path.name)
    events_name = stem.rsplit('_bold', maxsplit=1)[0] + '_events.tsv'
    events_path = nifti_path.parent / events_name
    if not events_path.exists():
        return False, ''
    try:
        n_lines = sum(1 for _ in open(events_path))
        return True, max(n_lines - 1, 0)
    except OSError:
        return True, ''


def _is_dwi(acq_key: str) -> bool:
    """Whether this acquisition is a DWI scan (expects bval/bvec)."""
    return acq_key.startswith('dwi_') and acq_key.endswith('_dwi')


def _find_bval_bvec(nifti_path: Path) -> tuple[bool, int | str, bool, int | str]:
    """Look for BIDS .bval and .bvec files matching this DWI NIfTI.

    Returns (bval_exists, bval_n, bvec_exists, bvec_n) where:
      bval_n = number of b-values (space-separated entries in the file)
      bvec_n = number of directions (columns; each of 3 rows has this many entries)
    Values are '' when the file is absent or unreadable.
    """
    stem = _strip_nifti_ext(nifti_path.name)
    bval_path = nifti_path.parent / (stem + '.bval')
    bvec_path = nifti_path.parent / (stem + '.bvec')

    bval_exists = bval_path.exists()
    bval_n: int | str = ''
    if bval_exists:
        try:
            bval_n = len(bval_path.read_text().split())
        except OSError:
            pass

    bvec_exists = bvec_path.exists()
    bvec_n: int | str = ''
    if bvec_exists:
        try:
            lines = bvec_path.read_text().strip().splitlines()
            if lines:
                bvec_n = len(lines[0].split())
        except OSError:
            pass

    return bval_exists, bval_n, bvec_exists, bvec_n


def _find_json_sidecar(nifti_path: Path, bids_root: Path) -> bool:
    """Check whether a JSON sidecar exists for this NIfTI, using BIDS inheritance.

    Search order (most specific -> least specific):
        1. Same directory, exact filename match  (sub-01_task-X_run-01_bold.json)
        2. Same directory, without run entity     (sub-01_task-X_bold.json)
        3. Subject (or session) root directory
        4. BIDS root directory
    At levels 3-4, the sub-/ses- entities are stripped from the JSON filename.
    """
    stem = _strip_nifti_ext(nifti_path.name)

    # 1) Exact match in same directory
    if (nifti_path.parent / (stem + '.json')).exists():
        return True

    # Build entity-stripped versions for inheritance lookup
    parts = stem.split('_')
    without_sub_ses = [p for p in parts if not p.startswith('sub-') and not p.startswith('ses-')]
    without_run = [p for p in without_sub_ses if not p.startswith('run-')]

    # 2) Same directory without run
    if without_run != without_sub_ses:
        candidate = nifti_path.parent / ('_'.join(without_run) + '.json')
        if candidate.exists():
            return True

    # 3) Walk up to subject or session root
    search_dir = nifti_path.parent.parent  # one level above modality dir
    while search_dir >= bids_root:
        for variant in [without_sub_ses, without_run]:
            candidate = search_dir / ('_'.join(variant) + '.json')
            if candidate.exists():
                return True
        if search_dir == bids_root:
            break
        search_dir = search_dir.parent

    return False


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _iter_subject_dirs(bids_root: Path) -> list[Path]:
    """Return sorted list of sub-* directories."""
    return sorted(d for d in bids_root.iterdir()
                  if d.is_dir() and d.name.startswith('sub-'))


def _iter_sessions(subj_dir: Path) -> list[Path | None]:
    """Return sorted session directories, or [None] if no sessions."""
    ses_dirs = sorted(d for d in subj_dir.iterdir()
                      if d.is_dir() and d.name.startswith('ses-'))
    return ses_dirs if ses_dirs else [None]


def _data_root(subj_dir: Path, ses_dir: Path | None) -> Path:
    """Return the directory that contains modality folders (anat/, func/, ...)."""
    return ses_dir if ses_dir is not None else subj_dir


def discover_acquisitions(bids_root: Path) -> tuple[list[str], bool]:
    """First pass: discover all unique acquisition keys across the dataset.

    Returns (sorted_keys, has_sessions).
    """
    keys: set[str] = set()
    has_sessions = False

    for subj_dir in _iter_subject_dirs(bids_root):
        sessions = _iter_sessions(subj_dir)
        if sessions != [None]:
            has_sessions = True

        for ses_dir in sessions:
            data = _data_root(subj_dir, ses_dir)
            for mod_dir in sorted(data.iterdir()):
                if not mod_dir.is_dir() or mod_dir.name not in _MODALITY_DIRS:
                    continue
                for f in mod_dir.iterdir():
                    if f.is_file() and _nifti_suffixes(f):
                        keys.add(_acquisition_key(mod_dir.name, f.name))

    # Sort: group by modality, then alphabetically
    return sorted(keys), has_sessions


# ---------------------------------------------------------------------------
# Per-subject checking
# ---------------------------------------------------------------------------

def check_subject_session(bids_root: Path, subj_dir: Path, ses_dir: Path | None,
                          acquisitions: list[str], read_dim4: bool = True) -> dict:
    """Check one subject(/session) against the full set of acquisitions."""
    data = _data_root(subj_dir, ses_dir)

    # Build lookup: acquisition_key -> nifti_path
    found_niis: dict[str, Path] = {}
    for mod_dir in sorted(data.iterdir()):
        if not mod_dir.is_dir() or mod_dir.name not in _MODALITY_DIRS:
            continue
        for f in mod_dir.iterdir():
            if f.is_file() and _nifti_suffixes(f):
                key = _acquisition_key(mod_dir.name, f.name)
                found_niis[key] = f

    row: dict[str, object] = {
        'subject': subj_dir.name,
    }
    if ses_dir is not None:
        row['session'] = ses_dir.name

    for acq in acquisitions:
        nii_path = found_niis.get(acq)
        has_nii = nii_path is not None
        has_json = _find_json_sidecar(nii_path, bids_root) if has_nii else False

        row[acq] = 1 if (has_nii and has_json) else 0
        row[acq + '_nii'] = 1 if has_nii else 0

        if read_dim4 and _wants_dim4(acq):
            row[acq + '_dim4'] = _get_dim4(nii_path) if has_nii else ''

        if _is_func_bold(acq):
            if has_nii:
                ev_exists, ev_rows = _find_events_tsv(nii_path)
                row[acq + '_events'] = 1 if ev_exists else 0
                row[acq + '_events_nrows'] = ev_rows
            else:
                row[acq + '_events'] = 0
                row[acq + '_events_nrows'] = ''

        if _is_dwi(acq):
            if has_nii:
                bval_ok, bval_n, bvec_ok, bvec_n = _find_bval_bvec(nii_path)
                row[acq + '_bval'] = 1 if bval_ok else 0
                row[acq + '_bval_n'] = bval_n
                row[acq + '_bvec'] = 1 if bvec_ok else 0
                row[acq + '_bvec_n'] = bvec_n
            else:
                row[acq + '_bval'] = 0
                row[acq + '_bval_n'] = ''
                row[acq + '_bvec'] = 0
                row[acq + '_bvec_n'] = ''

    return row


def _check_worker(args: tuple) -> dict:
    """Wrapper for ThreadPoolExecutor — unpacks arguments for check_subject_session."""
    bids_root, subj_dir, ses_dir, acquisitions, read_dim4 = args
    return check_subject_session(bids_root, subj_dir, ses_dir, acquisitions, read_dim4)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _add_dim4_ok(df: pd.DataFrame, acquisitions: list[str]) -> None:
    """Add _dim4_ok columns: 0 when dim4 deviates from the dataset mode."""
    for acq in acquisitions:
        dim4_col = acq + '_dim4'
        if dim4_col not in df.columns:
            continue

        # Collect non-empty numeric values to compute mode
        values = pd.to_numeric(df[dim4_col], errors='coerce')
        valid = values.dropna()
        if valid.empty:
            continue

        expected = int(stat_mode(valid))
        # ok = 1 when value matches mode OR when nii is absent (empty)
        ok_col = acq + '_dim4_ok'
        df[ok_col] = [
            '' if v == '' else (1 if v == expected else 0)
            for v in df[dim4_col]
        ]

        n_bad = (df[ok_col] == 0).sum()
        if n_bad:
            log(f'  WARNING: {acq} dim4 mode={expected}, {n_bad} subject(s) deviate')


def _add_dwi_ok(df: pd.DataFrame, acquisitions: list[str]) -> None:
    """Add _dwi_ok column: 0 when dim4 / bval_n / bvec_n are inconsistent."""
    for acq in acquisitions:
        if not _is_dwi(acq):
            continue
        dim4_col = acq + '_dim4'
        bval_col = acq + '_bval_n'
        bvec_col = acq + '_bvec_n'
        # All three columns must exist
        if not all(c in df.columns for c in [dim4_col, bval_col, bvec_col]):
            continue

        ok_values = []
        for _, row in df.iterrows():
            d4 = row[dim4_col]
            bn = row[bval_col]
            vn = row[bvec_col]
            # If nii is absent, nothing to compare
            if d4 == '' and bn == '' and vn == '':
                ok_values.append('')
                continue
            nums = set()
            for v in [d4, bn, vn]:
                try:
                    nums.add(int(v))
                except (ValueError, TypeError):
                    pass
            # ok if all present numeric values agree
            ok_values.append(1 if len(nums) <= 1 and nums else 0)

        ok_col = acq + '_dwi_ok'
        df[ok_col] = ok_values
        n_bad = sum(1 for v in ok_values if v == 0)
        if n_bad:
            log(f'  WARNING: {acq} dim4/bval/bvec mismatch in {n_bad} subject(s)')


def _drop_redundant_nii_cols(df: pd.DataFrame, acquisitions: list[str]) -> None:
    """Drop _nii columns when they are identical to the main column for all rows."""
    to_drop = []
    for acq in acquisitions:
        nii_col = acq + '_nii'
        if nii_col in df.columns and (df[acq] == df[nii_col]).all():
            to_drop.append(nii_col)
    if to_drop:
        df.drop(columns=to_drop, inplace=True)


def _drop_never_present(df: pd.DataFrame, acquisitions: list[str]) -> None:
    """Drop events / bval / bvec columns when never present across all subjects."""
    for acq in acquisitions:
        if _is_func_bold(acq):
            ev_col = acq + '_events'
            if ev_col in df.columns and df[ev_col].sum() == 0:
                df.drop(columns=[ev_col, acq + '_events_nrows'], inplace=True)

        if _is_dwi(acq):
            for tag in ['bval', 'bvec']:
                col = acq + '_' + tag
                if col in df.columns and df[col].sum() == 0:
                    df.drop(columns=[col, col + '_n'], inplace=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _merge_participants(df: pd.DataFrame, bids_root: Path) -> pd.DataFrame:
    """Merge with participants.tsv if it exists. Returns the (possibly merged) DataFrame."""
    tsv_path = bids_root / 'participants.tsv'
    if not tsv_path.is_file():
        log('No participants.tsv found — skipping merge.')
        return df

    df_part = pd.read_csv(tsv_path, sep='\t')
    if 'participant_id' not in df_part.columns:
        log('participants.tsv has no participant_id column — skipping merge.')
        return df

    df_part = df_part.rename(columns={'participant_id': 'subject'})
    participant_cols = [c for c in df_part.columns if c != 'subject']

    df = df.merge(df_part, on='subject', how='left')

    # Reorder: subject, [session], participant columns, then everything else
    id_cols = ['subject']
    if 'session' in df.columns:
        id_cols.append('session')
    other_cols = [c for c in df.columns if c not in id_cols and c not in participant_cols]
    df = df[id_cols + participant_cols + other_cols]

    n_matched = df[participant_cols[0]].notna().sum() if participant_cols else len(df)
    log(f'Merged {n_matched}/{len(df)} subjects with participants.tsv ({", ".join(participant_cols)})')
    return df


def main():
    parser = argparse.ArgumentParser(
        prog='bids-inspector',
        description='BIDS-Inspector: validate and summarise a BIDS dataset.')
    parser.add_argument('bidsdir', nargs='?', default='.',
                        help='Path to BIDS root directory (default: current directory)')
    parser.add_argument('-o', '--output', default='bids_inspector.csv',
                        help='Output CSV file (default: bids_inspector.csv)')
    parser.add_argument('--no-dim4', action='store_true',
                        help='Skip reading NIfTI headers (faster, no dim4 columns)')
    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='Number of parallel threads (default: 1)')
    parser.add_argument('--log', dest='logfile', default=None, metavar='FILE',
                        help='Write log output to FILE (in addition to stdout)')
    args = parser.parse_args()

    # --- Open log file ---
    global _log_file
    if args.logfile:
        _log_file = open(args.logfile, 'w')
    else:
        # Default: same name as output CSV but with .log extension
        log_path = Path(args.output).with_suffix('.log')
        _log_file = open(log_path, 'w')

    log(f'BIDS-Inspector — {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    log(f'Command: {" ".join(sys.argv)}')

    bids_root = Path(args.bidsdir).resolve()
    if not bids_root.is_dir():
        log(f'Error: {bids_root} is not a directory')
        sys.exit(1)

    # --- Discovery pass ---
    log(f'\nDiscovering acquisitions in {bids_root} ...')
    acquisitions, has_sessions = discover_acquisitions(bids_root)

    if not acquisitions:
        log('No NIfTI files found in any sub-*/[ses-*/]{anat,func,dwi,fmap,...}/ directory.')
        sys.exit(1)

    log(f'Found {len(acquisitions)} unique acquisition types:')
    for a in acquisitions:
        dim4_tag = '  [+dim4]' if _wants_dim4(a) else ''
        log(f'  {a}{dim4_tag}')

    read_dim4 = not args.no_dim4

    # --- Check pass ---
    subj_dirs = _iter_subject_dirs(bids_root)
    work_items: list[tuple] = []
    for subj_dir in subj_dirs:
        for ses_dir in _iter_sessions(subj_dir):
            work_items.append((bids_root, subj_dir, ses_dir, acquisitions, read_dim4))

    total = len(work_items)
    log(f'\nChecking {total} subject/session combinations (jobs={args.jobs}) ...')

    if args.jobs > 1:
        rows: list[dict] = [{}] * total  # pre-allocate to preserve order
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_idx = {
                executor.submit(_check_worker, item): idx
                for idx, item in enumerate(work_items)
            }
            done = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                rows[idx] = future.result()
                done += 1
                item = work_items[idx]
                label = item[1].name  # subj_dir.name
                if item[2] is not None:
                    label += '/' + item[2].name
                log(f'  [{done}/{total}] {label}')
    else:
        rows = []
        for i, item in enumerate(work_items, 1):
            label = item[1].name
            if item[2] is not None:
                label += '/' + item[2].name
            log(f'  [{i}/{total}] {label}')
            rows.append(_check_worker(item))

    # --- Build output columns ---
    columns = ['subject']
    if has_sessions:
        columns.append('session')
    for acq in acquisitions:
        columns.append(acq)
        columns.append(acq + '_nii')
        if read_dim4 and _wants_dim4(acq):
            columns.append(acq + '_dim4')
        if _is_func_bold(acq):
            columns.append(acq + '_events')
            columns.append(acq + '_events_nrows')
        if _is_dwi(acq):
            columns.append(acq + '_bval')
            columns.append(acq + '_bval_n')
            columns.append(acq + '_bvec')
            columns.append(acq + '_bvec_n')

    df = pd.DataFrame(rows, columns=columns)

    # --- Post-processing ---
    log()
    if read_dim4:
        _add_dim4_ok(df, acquisitions)
        _add_dwi_ok(df, acquisitions)
    _drop_never_present(df, acquisitions)
    _drop_redundant_nii_cols(df, acquisitions)

    # Reorder: insert _dim4_ok right after _dim4, _dwi_ok after bvec_n
    final_cols = []
    for c in df.columns:
        final_cols.append(c)
        if c.endswith('_dim4') and c + '_ok' in df.columns:
            final_cols.append(c + '_ok')
        if c.endswith('_bvec_n') and c.replace('_bvec_n', '_dwi_ok') in df.columns:
            final_cols.append(c.replace('_bvec_n', '_dwi_ok'))
    # Deduplicate while preserving order
    seen = set()
    ordered = []
    for c in final_cols:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    df = df[ordered]

    # --- Merge participants.tsv if available ---
    df = _merge_participants(df, bids_root)

    df.to_csv(args.output, index=False)
    log(f'\nResults written to {args.output}')

    # --- Summary ---
    n_subj = len(df)
    for acq in acquisitions:
        nii_col = acq + '_nii'
        has_nii_col = nii_col in df.columns
        present = int(df[acq].sum())
        nii_only = (int(df[nii_col].sum()) - present) if has_nii_col else 0
        missing = n_subj - present - nii_only
        parts = [f'{present}/{n_subj} complete']
        if nii_only:
            parts.append(f'{nii_only} nii-only (json missing)')
        if missing:
            parts.append(f'{missing} missing')
        log(f'  {acq}: {", ".join(parts)}')

    if _log_file is not None:
        log_path = _log_file.name
        _log_file.close()
        print(f'Log written to {log_path}')


if __name__ == '__main__':
    main()
