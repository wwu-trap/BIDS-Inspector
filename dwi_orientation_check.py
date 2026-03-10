#!/usr/bin/env python3
"""dwi_orientation_check.py — Check DWI and fmap EPI acquisition orientation.

Detects problematic (tilted/oblique) MRI sequence planning by analysing:
  - ImageOrientationPatientDICOM from JSON sidecars (primary source)
  - NIfTI affine matrix (fallback when JSON field is absent)
  - ImageOrientationText for human-readable annotation (Siemens)

Checks performed
----------------
1. Tilt outliers         Slice normal deviates from dataset-median by > threshold.
2. PE direction pairs    Each fmap EPI should have exactly one partner with the
                         opposite phase-encoding direction (same axis, flipped sign).
                         Handles both 'j-' (BIDS spec) and '-j' (common variant).
3. DWI ↔ fmap-EPI match  Orientation of a fmap EPI must agree with its IntendedFor
                         DWI scan — flags cases where they were planned differently.
4. Session consistency   Same acquisition type should have a consistent slice
                         orientation across all sessions of the same subject.

Output
------
  <prefix>.tsv  — per-file report with all check columns (tab-separated)
  <prefix>.png  — scatter plot (requires matplotlib)

Requires: nibabel, pandas
Optional: matplotlib (for plot)
"""

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import nibabel as nib
import pandas as pd


# ---------------------------------------------------------------------------
# Logging helper — mirrors all print() output to a log file
# ---------------------------------------------------------------------------

class _Tee:
    """Write to both sys.stdout and a log file simultaneously."""

    def __init__(self, log_path: str):
        self._file = open(log_path, 'w', encoding='utf-8')  # noqa: WPS515
        self._stdout = sys.stdout

    def write(self, msg: str) -> None:
        self._stdout.write(msg)
        self._file.write(msg)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


@contextlib.contextmanager
def _tee_stdout(log_path: str | None):
    """Context manager: redirect sys.stdout through _Tee when log_path is given."""
    if not log_path:
        yield
        return
    tee = _Tee(log_path)
    old_stdout = sys.stdout
    sys.stdout = tee
    try:
        yield
    finally:
        sys.stdout = old_stdout
        tee.close()


# ---------------------------------------------------------------------------
# Vector helpers (pure Python — no numpy required)
# ---------------------------------------------------------------------------

def _cross3(a: list, b: list) -> list:
    return [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0],
    ]


def _dot3(a: list, b: list) -> float:
    return sum(ai * bi for ai, bi in zip(a, b))


def _norm3(v: list) -> list:
    mag = math.sqrt(sum(x*x for x in v))
    return [x / mag for x in v] if mag > 1e-10 else list(v)


def _angle_deg(a: list, b: list) -> float:
    """Angle in degrees between two vectors (anti-parallel treated as parallel)."""
    dot = max(-1.0, min(1.0, _dot3(_norm3(a), _norm3(b))))
    return math.degrees(math.acos(abs(dot)))


def _median3(vecs: list) -> list:
    """Component-wise median of a list of [x, y, z] vectors."""
    if not vecs:
        return [0.0, 0.0, 1.0]
    return [sorted(v[i] for v in vecs)[len(vecs) // 2] for i in range(3)]


def _fmt3(v: list) -> str:
    return '[{:.4f}, {:.4f}, {:.4f}]'.format(*v)


# Cardinal axes for orientation labelling
_CARDINALS = {
    'X (L-R)': ([1.0, 0.0, 0.0], 'sagittal'),
    'Y (A-P)': ([0.0, 1.0, 0.0], 'coronal'),
    'Z (S-I)': ([0.0, 0.0, 1.0], 'axial'),
}


def _nearest_cardinal(v: list) -> tuple:
    """Return (cardinal_key, orientation_label, tilt_deg) for the nearest cardinal axis."""
    best_key, best_label, best_angle = 'Z (S-I)', 'axial', 90.0
    for key, (cardinal, label) in _CARDINALS.items():
        angle = _angle_deg(v, cardinal)
        if angle < best_angle:
            best_angle, best_key, best_label = angle, key, label
    return best_key, best_label, best_angle


# ---------------------------------------------------------------------------
# Phase-encoding direction helpers
# ---------------------------------------------------------------------------

def _normalize_ped(ped: str) -> tuple[str, str]:
    """Normalise PhaseEncodingDirection to (axis, polarity).

    Handles both BIDS-spec 'j-' and the common variant '-j', as well as
    plain 'j' (positive) and the unusual 'j+' form.

    Returns
    -------
    axis     : 'i', 'j', 'k', or '' (if unknown/absent)
    polarity : '+' or '-', or '' (if unknown/absent)
    """
    ped = ped.strip()
    if not ped:
        return '', ''
    # Leading minus: '-j', '-i', '-k'
    if ped.startswith('-'):
        return ped[1:].lower(), '-'
    # Trailing minus: 'j-', 'i-', 'k-'
    if ped.endswith('-'):
        return ped[:-1].lower(), '-'
    # Trailing plus: 'j+', 'i+' (unusual but valid)
    if ped.endswith('+'):
        return ped[:-1].lower(), '+'
    return ped.lower(), '+'


def _ped_canonical(axis: str, polarity: str) -> str:
    """Return the canonical BIDS form, e.g. 'j' or 'j-'."""
    if not axis:
        return ''
    return axis if polarity == '+' else axis + '-'


# ---------------------------------------------------------------------------
# BIDS path helpers
# ---------------------------------------------------------------------------

def _nifti_suffix(path: Path) -> str:
    if path.name.endswith('.nii.gz'):
        return '.nii.gz'
    if path.name.endswith('.nii'):
        return '.nii'
    return ''


def _strip_ext(filename: str) -> str:
    if filename.endswith('.nii.gz'):
        return filename[:-7]
    if filename.endswith('.nii'):
        return filename[:-4]
    return filename


def _bids_suffix(filename: str) -> str:
    return _strip_ext(filename).rsplit('_', maxsplit=1)[-1]


def _acq_key_no_sub_ses(filename: str) -> str:
    """Acquisition key without sub-/ses- entities (for cross-session comparison)."""
    stem = _strip_ext(filename)
    parts = stem.split('_')
    filtered = [p for p in parts
                if not p.startswith('sub-') and not p.startswith('ses-')]
    return '_'.join(filtered)


def _fmap_pair_key(filename: str) -> str:
    """Group key for fmap EPI reverse-PE pairs: strips sub-, ses-, dir- entities.

    Files that form a valid AP/PA pair will share the same pair key.
    """
    stem = _strip_ext(filename)
    parts = stem.split('_')
    filtered = [p for p in parts
                if not p.startswith('sub-')
                and not p.startswith('ses-')
                and not p.startswith('dir-')]
    return '_'.join(filtered)


def _iter_subject_dirs(bids_root: Path) -> list:
    return sorted(d for d in bids_root.iterdir()
                  if d.is_dir() and d.name.startswith('sub-'))


def _iter_sessions(subj_dir: Path) -> list:
    ses_dirs = sorted(d for d in subj_dir.iterdir()
                      if d.is_dir() and d.name.startswith('ses-'))
    return ses_dirs if ses_dirs else [None]


# ---------------------------------------------------------------------------
# JSON sidecar reading — BIDS inheritance
# ---------------------------------------------------------------------------

def _read_json_sidecar(nifti_path: Path, bids_root: Path) -> dict:
    """Read the JSON sidecar for a NIfTI using BIDS inheritance. Returns {} on failure."""
    stem = _strip_ext(nifti_path.name)
    parts = stem.split('_')
    without_sub_ses = [p for p in parts
                       if not p.startswith('sub-') and not p.startswith('ses-')]
    without_run = [p for p in without_sub_ses if not p.startswith('run-')]

    candidates = [nifti_path.parent / (stem + '.json')]
    if without_run != without_sub_ses:
        candidates.append(nifti_path.parent / ('_'.join(without_run) + '.json'))

    search_dir = nifti_path.parent.parent
    while search_dir >= bids_root:
        for variant in [without_sub_ses, without_run]:
            candidates.append(search_dir / ('_'.join(variant) + '.json'))
        if search_dir == bids_root:
            break
        search_dir = search_dir.parent

    for cand in candidates:
        if cand.exists():
            try:
                return json.loads(cand.read_text())
            except Exception:
                return {}
    return {}


def _intended_for_basenames(sidecar: dict) -> list[str]:
    """Extract target file basenames from IntendedFor (normalised)."""
    raw = sidecar.get('IntendedFor', [])
    if isinstance(raw, str):
        raw = [raw]
    result = []
    for entry in raw:
        if not isinstance(entry, str):
            continue
        # Strip 'bids::' prefix used in some converters
        if entry.startswith('bids::'):
            entry = entry[6:]
        result.append(Path(entry).name)
    return result


# ---------------------------------------------------------------------------
# Orientation extraction
# ---------------------------------------------------------------------------

def _orientation_from_iop(iop: list) -> tuple | None:
    """Parse 6-element ImageOrientationPatientDICOM into (row, col, normal) or None."""
    try:
        if len(iop) != 6:
            return None
        row = _norm3([float(iop[0]), float(iop[1]), float(iop[2])])
        col = _norm3([float(iop[3]), float(iop[4]), float(iop[5])])
        normal = _norm3(_cross3(row, col))
        return row, col, normal
    except Exception:
        return None


def _orientation_from_nifti(nifti_path: Path) -> tuple | None:
    """Derive slice orientation from the NIfTI affine (RAS convention)."""
    try:
        img = nib.load(str(nifti_path))
        aff = img.affine
        row    = _norm3([float(aff[0, 0]), float(aff[1, 0]), float(aff[2, 0])])
        col    = _norm3([float(aff[0, 1]), float(aff[1, 1]), float(aff[2, 1])])
        normal = _norm3(_cross3(row, col))
        return row, col, normal
    except Exception:
        return None


def _get_orientation(nifti_path: Path, sidecar: dict) -> tuple:
    """Extract (row, col, normal, source).

    Priority: ImageOrientationPatientDICOM → NIfTI affine.
    source = 'json_iop' | 'nifti_affine' | 'unavailable'
    """
    iop = sidecar.get('ImageOrientationPatientDICOM')
    if iop is not None:
        parsed = _orientation_from_iop(iop)
        if parsed:
            return parsed[0], parsed[1], parsed[2], 'json_iop'

    parsed = _orientation_from_nifti(nifti_path)
    if parsed:
        return parsed[0], parsed[1], parsed[2], 'nifti_affine'

    return None, None, None, 'unavailable'


# ---------------------------------------------------------------------------
# Dataset scanning
# ---------------------------------------------------------------------------

_TARGET_SUFFIXES = {'dwi', 'epi'}


def collect_orientation_records(bids_root: Path) -> list:
    """Scan bids_root and return orientation records for all DWI + fmap EPI files."""
    records = []

    for subj_dir in _iter_subject_dirs(bids_root):
        for ses_dir in _iter_sessions(subj_dir):
            data_root = ses_dir if ses_dir is not None else subj_dir

            for mod_name in ('dwi', 'fmap'):
                mod_dir = data_root / mod_name
                if not mod_dir.is_dir():
                    continue

                for nii_path in sorted(mod_dir.glob('*.nii*')):
                    if not _nifti_suffix(nii_path):
                        continue
                    suffix = _bids_suffix(nii_path.name)
                    if suffix not in _TARGET_SUFFIXES:
                        continue

                    sidecar = _read_json_sidecar(nii_path, bids_root)
                    row_vec, col_vec, normal, src = _get_orientation(nii_path, sidecar)

                    ped_raw = sidecar.get('PhaseEncodingDirection',
                                         sidecar.get('PhaseEncodingAxis', ''))
                    pe_axis, pe_polarity = _normalize_ped(ped_raw)

                    intended = _intended_for_basenames(sidecar)

                    rec: dict = {
                        'subject':                subj_dir.name,
                        'session':                ses_dir.name if ses_dir else '',
                        'modality':               mod_name,
                        'suffix':                 suffix,
                        'file':                   nii_path.name,
                        'ImageOrientationText':   sidecar.get('ImageOrientationText', ''),
                        'PhaseEncodingDirection': ped_raw,
                        'pe_axis':                pe_axis,
                        'pe_polarity':            pe_polarity,
                        'pe_canonical':           _ped_canonical(pe_axis, pe_polarity),
                        'orientation_source':     src,
                        'intended_for':           '; '.join(intended),
                        # Internal — dropped before CSV output
                        '_nx':               normal[0] if normal is not None else float('nan'),
                        '_ny':               normal[1] if normal is not None else float('nan'),
                        '_nz':               normal[2] if normal is not None else float('nan'),
                        '_intended_for_list': intended,
                        '_pair_key':          _fmap_pair_key(nii_path.name),
                        '_acq_key_no_ses':    _acq_key_no_sub_ses(nii_path.name),
                    }

                    if normal is not None:
                        cardinal_ax, orient_label, cardinal_tilt = _nearest_cardinal(normal)
                        rec.update({
                            'row_vec':            _fmt3(row_vec),
                            'col_vec':            _fmt3(col_vec),
                            'slice_normal':       _fmt3(normal),
                            'nearest_cardinal':   cardinal_ax,
                            'orientation_label':  orient_label,
                            'tilt_from_cardinal': round(cardinal_tilt, 3),
                        })
                    else:
                        rec['tilt_from_cardinal'] = float('nan')

                    records.append(rec)

    return records


# ---------------------------------------------------------------------------
# Check 1 — Tilt outliers (global dataset-median reference)
# ---------------------------------------------------------------------------

def _check_tilt_outliers(df: pd.DataFrame, threshold_deg: float) -> None:
    """Add tilt_from_median_deg and tilt_outlier columns in-place."""
    has_normal = df[['_nx', '_ny', '_nz']].notna().all(axis=1)
    if not has_normal.any():
        return

    normals = df.loc[has_normal, ['_nx', '_ny', '_nz']].values.tolist()
    med_normal = _median3(normals)

    deviations = []
    for _, row in df.iterrows():
        try:
            if math.isnan(row['_nx']):
                deviations.append(float('nan'))
            else:
                dev = _angle_deg([row['_nx'], row['_ny'], row['_nz']], med_normal)
                deviations.append(round(dev, 3))
        except Exception:
            deviations.append(float('nan'))

    df['tilt_from_median_deg'] = deviations
    df['tilt_outlier'] = [
        (not math.isnan(d)) and (d > threshold_deg)
        for d in deviations
    ]


# ---------------------------------------------------------------------------
# Check 2 — Phase-encoding direction pairs
# ---------------------------------------------------------------------------

def _check_pe_pairs(df: pd.DataFrame) -> None:
    """Add pe_pair_status column to all records.

    Statuses
    --------
    ok               valid AP/PA (or RL/LR) pair found within fmap/
    ok_multiple      > 2 files share this pair key, but at least one +/- pair exists
    ok_with_dwi      single fmap EPI whose IntendedFor DWI provides the reverse-PE
                     partner (the most common DWI acquisition pattern)
    no_partner       only one file for this pair key and no reverse-PE DWI partner found
    same_polarity    both/all files have the same PE direction — no true reversal
    mixed_axis       files in the group use different PE axes (likely a labelling error)
    axis_unknown     PE axis not determinable from the sidecar
    n/a              not a fmap EPI (DWI main scan or other)
    """
    df['pe_pair_status'] = 'n/a'

    fmap_mask = (df['modality'] == 'fmap') & (df['suffix'] == 'epi')
    if not fmap_mask.any():
        return

    # Group fmap EPI files by (subject, session, pair_key)
    fmap_idx = df.index[fmap_mask]
    fmap_df  = df.loc[fmap_idx]

    groups = fmap_df.groupby(['subject', 'session', '_pair_key'])

    for (_, _, _), grp in groups:
        idxs = grp.index.tolist()

        axes      = grp['pe_axis'].tolist()
        polarities = grp['pe_polarity'].tolist()

        # All axes unknown?
        if all(a == '' for a in axes):
            df.loc[idxs, 'pe_pair_status'] = 'axis_unknown'
            continue

        # Mixed axes?
        known_axes = [a for a in axes if a]
        if len(set(known_axes)) > 1:
            df.loc[idxs, 'pe_pair_status'] = 'mixed_axis'
            continue

        if len(grp) == 1:
            df.loc[idxs, 'pe_pair_status'] = 'no_partner'
        elif len(grp) == 2:
            pol_set = set(polarities)
            if pol_set == {'+', '-'}:
                df.loc[idxs, 'pe_pair_status'] = 'ok'
            else:
                df.loc[idxs, 'pe_pair_status'] = 'same_polarity'
        else:
            # > 2 files with same pair key
            if '+' in polarities and '-' in polarities:
                df.loc[idxs, 'pe_pair_status'] = 'ok_multiple'
            else:
                df.loc[idxs, 'pe_pair_status'] = 'same_polarity'

    # ── Second pass: upgrade no_partner → ok_with_dwi ────────────────────────
    # The most common DWI fieldmap setup is: DWI main scan (e.g. AP) + single
    # fmap EPI b0 (e.g. PA), linked via IntendedFor.  In that case no second
    # fmap EPI exists, but the pair is perfectly valid.
    no_partner_mask = df['pe_pair_status'] == 'no_partner'
    if not no_partner_mask.any():
        return

    # Build a lookup: (subject, session, dwi_basename) → pe_axis, pe_polarity
    dwi_pe: dict[tuple, tuple] = {}
    for _, row in df[df['suffix'] == 'dwi'].iterrows():
        key = (row['subject'], row['session'], row['file'])
        dwi_pe[key] = (row['pe_axis'], row['pe_polarity'])

    for idx, row in df[no_partner_mask].iterrows():
        fmap_axis     = row['pe_axis']
        fmap_polarity = row['pe_polarity']
        if not fmap_axis or not fmap_polarity:
            continue

        for dwi_name in row.get('_intended_for_list', []):
            if _bids_suffix(dwi_name) != 'dwi':
                continue
            key = (row['subject'], row['session'], dwi_name)
            dwi_axis, dwi_polarity = dwi_pe.get(key, ('', ''))
            if (dwi_axis == fmap_axis
                    and dwi_polarity
                    and dwi_polarity != fmap_polarity):
                df.at[idx, 'pe_pair_status'] = 'ok_with_dwi'
                break


# ---------------------------------------------------------------------------
# Check 3 — DWI ↔ fmap-EPI orientation consistency
# ---------------------------------------------------------------------------

def _check_dwi_fmap_consistency(df: pd.DataFrame, threshold_deg: float) -> None:
    """Add dwi_fmap_angle_deg and dwi_fmap_consistent columns.

    For each fmap EPI whose IntendedFor list contains DWI filenames, the
    angle between the fmap EPI slice normal and the DWI slice normal is
    computed. A mismatch (> threshold_deg) means the two scans were likely
    planned with different orientations — fieldmap correction will be wrong.

    Only filled for fmap EPI files that can be linked to a DWI in the dataset.
    """
    df['dwi_fmap_angle_deg'] = float('nan')
    df['dwi_fmap_consistent'] = None

    # Build filename → normal lookup for all DWI main scans
    dwi_mask = df['suffix'] == 'dwi'
    dwi_normals: dict[str, list] = {}
    for _, row in df[dwi_mask].iterrows():
        nx = row.get('_nx', float('nan'))
        if not math.isnan(nx):
            dwi_normals[row['file']] = [row['_nx'], row['_ny'], row['_nz']]

    # For each fmap EPI check against its intended DWI target(s)
    fmap_mask = (df['modality'] == 'fmap') & (df['suffix'] == 'epi')
    for idx, row in df[fmap_mask].iterrows():
        nx = row.get('_nx', float('nan'))
        if math.isnan(nx):
            continue

        dwi_targets = [f for f in row.get('_intended_for_list', [])
                       if _bids_suffix(f) == 'dwi']
        if not dwi_targets:
            continue

        fmap_normal = [row['_nx'], row['_ny'], row['_nz']]
        matched_angles = []
        for target in dwi_targets:
            if target in dwi_normals:
                matched_angles.append(_angle_deg(fmap_normal, dwi_normals[target]))

        if matched_angles:
            worst = round(max(matched_angles), 3)
            df.at[idx, 'dwi_fmap_angle_deg']  = worst
            df.at[idx, 'dwi_fmap_consistent'] = worst <= threshold_deg


# ---------------------------------------------------------------------------
# Check 4 — Session-to-session orientation consistency
# ---------------------------------------------------------------------------

def _check_session_consistency(df: pd.DataFrame, threshold_deg: float) -> None:
    """Add session_max_deviation_deg and session_consistent columns.

    For each subject, files sharing the same acquisition key (i.e. same
    filename modulo sub-/ses- entities) are compared across sessions.
    The maximum pairwise angular deviation between slice normals is reported.
    Only subjects with ≥ 2 sessions that have the same acquisition key are
    evaluated; everything else gets NaN / empty.
    """
    df['session_max_deviation_deg'] = float('nan')
    df['session_consistent'] = None

    for (subject, acq_key), grp in df.groupby(['subject', '_acq_key_no_ses']):
        sessions_with_normal: dict[str, list] = {}
        for ses, ses_grp in grp.groupby('session'):
            for _, row in ses_grp.iterrows():
                nx = row.get('_nx', float('nan'))
                if not math.isnan(nx):
                    sessions_with_normal[ses] = [row['_nx'], row['_ny'], row['_nz']]
                    break  # one representative per session is enough

        if len(sessions_with_normal) < 2:
            continue

        ses_list = list(sessions_with_normal.keys())
        max_dev = 0.0
        for i in range(len(ses_list)):
            for j in range(i + 1, len(ses_list)):
                angle = _angle_deg(sessions_with_normal[ses_list[i]],
                                   sessions_with_normal[ses_list[j]])
                max_dev = max(max_dev, angle)

        max_dev = round(max_dev, 3)
        for idx in grp.index:
            df.at[idx, 'session_max_deviation_deg'] = max_dev
            df.at[idx, 'session_consistent']        = max_dev <= threshold_deg


# ---------------------------------------------------------------------------
# Console reporting helpers
# ---------------------------------------------------------------------------

def _report_tilt(df: pd.DataFrame, threshold_deg: float) -> int:
    if 'tilt_outlier' not in df.columns:
        return 0
    bad = df[df['tilt_outlier'] == True]
    n = len(bad)
    if n:
        print(f'\n  [Check 1] *** {n} file(s) with tilt > {threshold_deg}° from dataset median ***')
        for _, row in bad.iterrows():
            ses_str = f"/{row['session']}" if row.get('session') else ''
            print(f'    {row["subject"]}{ses_str}  [{row["modality"]}/{row["file"]}]')
            print(f'      nearest cardinal   : {row.get("nearest_cardinal","?")} '
                  f'({row.get("orientation_label","?")})')
            print(f'      tilt_from_cardinal : {row.get("tilt_from_cardinal", "?"):.2f}°')
            print(f'      tilt_from_median   : {row.get("tilt_from_median_deg", "?"):.2f}°')
            if row.get('ImageOrientationText'):
                print(f'      ImageOrientationText: {row["ImageOrientationText"]}')
    else:
        print(f'\n  [Check 1] OK — all files within {threshold_deg}° tilt threshold.')
    return n


def _report_pe_pairs(df: pd.DataFrame) -> int:
    if 'pe_pair_status' not in df.columns:
        return 0
    problem_statuses = {'no_partner', 'same_polarity', 'mixed_axis', 'axis_unknown'}
    bad = df[df['pe_pair_status'].isin(problem_statuses)]
    n = len(bad)
    if n:
        print(f'\n  [Check 2] *** {n} fmap EPI file(s) with PE-pair issues ***')
        for _, row in bad.iterrows():
            ses_str = f"/{row['session']}" if row.get('session') else ''
            print(f'    {row["subject"]}{ses_str}  [{row["modality"]}/{row["file"]}]')
            print(f'      pe_pair_status      : {row["pe_pair_status"]}')
            print(f'      PhaseEncodingDirection (raw): {row.get("PhaseEncodingDirection","?")}')
            print(f'      pe_canonical        : {row.get("pe_canonical","?")}')
    else:
        n_ok_dwi = (df['pe_pair_status'] == 'ok_with_dwi').sum()
        n_ok     = (df['pe_pair_status'].isin({'ok', 'ok_multiple'})).sum()
        extra = []
        if n_ok:
            extra.append(f'{n_ok} paired within fmap/')
        if n_ok_dwi:
            extra.append(f'{n_ok_dwi} paired with IntendedFor DWI')
        extra_str = f'  ({", ".join(extra)})' if extra else ''
        print(f'\n  [Check 2] OK — all fmap EPI files have valid reverse-PE partners.{extra_str}')
    return n


def _report_dwi_fmap(df: pd.DataFrame, threshold_deg: float) -> int:
    if 'dwi_fmap_consistent' not in df.columns:
        return 0
    # Only rows where the check was actually performed (not empty string)
    checked = df[df['dwi_fmap_consistent'].notna()]
    if checked.empty:
        print(f'\n  [Check 3] No fmap EPI → DWI IntendedFor links found; check skipped.')
        return 0
    bad = checked[checked['dwi_fmap_consistent'] == False]
    n = len(bad)
    if n:
        print(f'\n  [Check 3] *** {n} fmap EPI file(s) with orientation mismatch vs. intended DWI ***')
        for _, row in bad.iterrows():
            ses_str = f"/{row['session']}" if row.get('session') else ''
            print(f'    {row["subject"]}{ses_str}  [{row["modality"]}/{row["file"]}]')
            print(f'      dwi_fmap_angle_deg  : {row.get("dwi_fmap_angle_deg","?"):.2f}°  '
                  f'(threshold: {threshold_deg}°)')
            print(f'      intended_for        : {row.get("intended_for","")}')
    else:
        print(f'\n  [Check 3] OK — all fmap EPI orientations match their IntendedFor DWI '
              f'({len(checked)} pairs checked).')
    return n


def _report_session_consistency(df: pd.DataFrame, threshold_deg: float) -> int:
    if 'session_consistent' not in df.columns:
        return 0
    checked = df[df['session_consistent'].notna()]
    if checked.empty:
        print(f'\n  [Check 4] No multi-session subjects found; check skipped.')
        return 0
    bad_acq = checked[checked['session_consistent'] == False][
        ['subject', '_acq_key_no_ses', 'session_max_deviation_deg']
    ].drop_duplicates(subset=['subject', '_acq_key_no_ses'])
    n = len(bad_acq)
    if n:
        print(f'\n  [Check 4] *** {n} subject/acquisition pair(s) with inconsistent '
              f'session-to-session orientation ***')
        for _, row in bad_acq.iterrows():
            print(f'    {row["subject"]}  acq: {row["_acq_key_no_ses"]}')
            print(f'      session_max_deviation_deg: {row["session_max_deviation_deg"]:.2f}°  '
                  f'(threshold: {threshold_deg}°)')
    else:
        n_pairs = len(checked[['subject', '_acq_key_no_ses']].drop_duplicates())
        print(f'\n  [Check 4] OK — session-to-session orientation consistent '
              f'({n_pairs} subject/acq pair(s) checked).')
    return n


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

def _make_plot(df: pd.DataFrame, out_path: str, threshold_deg: float,
               warn_counts: dict,
               max_scatter_labels: int = 10,
               max_bar_labels: int = 20) -> None:
    """Two-panel plot: slice-normal scatter (left) + tilt bar chart (right).

    Readability strategy for large datasets
    ----------------------------------------
    Scatter:
      Labels are capped at `max_scatter_labels` total.  Priority order:
        1. DWI-fmap mismatch  (always critical, usually rare)
        2. Session inconsistency  (usually rare)
        3. Worst tilt outliers by tilt_from_median_deg
        4. Worst PE-pair issues by tilt_from_median_deg
      The legend is placed *below* both axes (figure legend) so it never
      overlaps the data.

    Bar chart:
      When total flagged bars > `max_bar_labels`: ALL individual bar labels
      are suppressed — a compact issue-count summary box appears in the
      top-right corner instead ("see TSV for subject details").
      When ≤ `max_bar_labels`: the worst flagged bars get a short label.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D
    except ImportError:
        print('  matplotlib not available — skipping plot')
        return

    has_normal = df[['_nx', '_ny', '_nz']].notna().all(axis=1)
    has_normal = has_normal & df['_nx'].apply(lambda x: not math.isnan(x))
    pdf = df[has_normal].copy().reset_index(drop=True)
    if pdf.empty:
        print('  No orientation data available for plotting.')
        return

    subjects = pdf['subject'].unique()
    n_subj   = max(len(subjects), 1)
    cmap     = plt.colormaps.get_cmap('tab20')
    subj_color = {s: cmap(i / n_subj) for i, s in enumerate(subjects)}

    suffix_markers = {'dwi': 'o', 'epi': 's'}
    default_marker  = 'D'

    tilt_col = ('tilt_from_median_deg' if 'tilt_from_median_deg' in pdf.columns
                else 'tilt_from_cardinal')

    _PE_BAD  = {'no_partner', 'same_polarity', 'mixed_axis', 'axis_unknown'}

    def _issue_type(row):
        if row.get('dwi_fmap_consistent') == False:   return 'fmap'
        if row.get('session_consistent')  == False:   return 'session'
        if bool(row.get('tilt_outlier', False)):       return 'tilt'
        if row.get('pe_pair_status', 'n/a') in _PE_BAD: return 'pe'
        return None

    def _edge(issue):
        return {'fmap': '#cc0000', 'tilt': '#cc0000',
                'session': '#e07b00', 'pe': '#9933cc'}.get(issue, 'white')

    # ── Build scatter label set (capped at max_scatter_labels, priority order) ─
    # buckets: fmap/session always go first, then tilt, then pe
    tilt_val = pdf[tilt_col] if tilt_col in pdf.columns else pd.Series(0.0, index=pdf.index)

    label_idx: set = set()
    for bucket in ('fmap', 'session', 'tilt', 'pe'):
        remaining = max_scatter_labels - len(label_idx)
        if remaining <= 0:
            break
        mask = pdf.index.map(lambda i: _issue_type(pdf.loc[i]) == bucket)
        candidates = pdf.index[mask]
        # Sort by tilt magnitude descending (worst first)
        ordered = sorted(candidates,
                         key=lambda i: tilt_val.get(i, 0.0) if not math.isnan(tilt_val.get(i, 0.0)) else 0.0,
                         reverse=True)
        label_idx.update(ordered[:remaining])

    # ── Figure: leave room at bottom for the figure-level legend ─────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 7.5))

    # ── Panel 1: 2D slice-normal scatter ─────────────────────────────────────
    for i, row in pdf.iterrows():
        issue  = _issue_type(row)
        ec     = _edge(issue) if issue else 'white'
        lw     = 2.0 if issue in ('fmap', 'tilt', 'session') else (1.8 if issue == 'pe' else 0.4)
        color  = subj_color[row['subject']]
        marker = suffix_markers.get(row.get('suffix', ''), default_marker)

        ax1.scatter(
            row['_nx'], row['_ny'],
            c=[color], marker=marker,
            s=110 if issue else 50,
            edgecolors=ec, linewidths=lw,
            zorder=4 if issue else 2, alpha=0.9,
        )

        if i in label_idx and issue:
            lbl = row['subject'].replace('sub-', '')
            if row.get('session'):
                lbl += '/' + row['session'].replace('ses-', '')
            ax1.annotate(
                lbl, (row['_nx'], row['_ny']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=7, color=ec, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15', fc='white',
                          alpha=0.75, ec=ec, lw=0.7),
            )

    # Threshold circle
    r_thresh = math.sin(math.radians(threshold_deg))
    ax1.add_patch(plt.Circle((0, 0), r_thresh, fill=False, color='darkorange',
                              linestyle='--', linewidth=1.8, zorder=5))
    ax1.axhline(0, color='#cccccc', lw=0.7, zorder=1)
    ax1.axvline(0, color='#cccccc', lw=0.7, zorder=1)
    ax1.plot(0, 0, 'k+', ms=11, zorder=6, markeredgewidth=1.5)

    for txt, xy, ha, va in [
        ('ideal axial', (0.02, 0.98), 'left',   'top'),
        ('← L-tilt',   (0.02, 0.50), 'left',   'center'),
        ('R-tilt →',   (0.98, 0.50), 'right',  'center'),
        ('↓ post.',    (0.50, 0.02), 'center', 'bottom'),
        ('↑ ant.',     (0.50, 0.98), 'center', 'top'),
    ]:
        ax1.text(*xy, txt, transform=ax1.transAxes,
                 fontsize=7, color='#aaaaaa', ha=ha, va=va)

    n_labelled   = len(label_idx)
    n_issues_tot = sum(1 for i in pdf.index if _issue_type(pdf.loc[i]))
    if n_issues_tot > n_labelled:
        ax1.text(0.01, 0.01,
                 f'{n_labelled}/{n_issues_tot} issues labelled  (worst shown)',
                 transform=ax1.transAxes, fontsize=6.5,
                 color='#999999', va='bottom')

    ax1.set_xlabel('Slice Normal — X  (L↔R)', fontsize=10)
    ax1.set_ylabel('Slice Normal — Y  (A↔P)', fontsize=10)
    ax1.set_title(
        'Slice-Normal 2D Projection\n'
        r'$(n_x,\,n_y)\approx(0,0)$ = ideal axial',
        fontsize=11, pad=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.18)

    # ── Panel 2: Tilt bar chart ───────────────────────────────────────────────
    sorted_df = pdf.sort_values(tilt_col).reset_index(drop=True)

    def _bar_color(row):
        issue = _issue_type(row)
        return {'fmap': '#cc0000', 'tilt': '#cc0000',
                'session': '#e07b00', 'pe': '#9933cc'}.get(issue, '#3a86c8')

    bar_colors = [_bar_color(r) for _, r in sorted_df.iterrows()]
    ax2.bar(range(len(sorted_df)), sorted_df[tilt_col],
            color=bar_colors, edgecolor='none', alpha=0.88, width=0.9)
    ax2.axhline(threshold_deg, color='darkorange', linestyle='--',
                linewidth=1.8, zorder=5)
    ax2.set_xticks([])

    # Count flagged bars per issue type
    n_flagged_total = sum(1 for _, r in sorted_df.iterrows() if _issue_type(r))

    if n_flagged_total <= max_bar_labels and n_flagged_total > 0:
        # ── Few enough flagged bars: label each one individually ──────────────
        # Sort flagged by tilt descending so we draw the worst first (back-to-front)
        flagged_rows = [(i, row) for i, (_, row) in enumerate(sorted_df.iterrows())
                        if _issue_type(row)]
        for bar_i, row in flagged_rows:
            val   = row[tilt_col]
            lbl   = row['subject'].replace('sub-', '')
            if row.get('session'):
                lbl += '/' + row['session'].replace('ses-', '')
            lbl  += f" [{row.get('suffix','?')}]"
            color = _bar_color(row)
            ymax  = ax2.get_ylim()[1] if ax2.get_ylim()[1] > 0 else 1.0
            label_y = max(val * 0.45, ymax * 0.03)
            ax2.text(bar_i, val + ymax * 0.012, f'{val:.1f}°',
                     ha='center', va='bottom', fontsize=7, color=color,
                     fontweight='bold')
            ax2.text(bar_i, label_y, lbl,
                     ha='center', va='center', fontsize=6.5, color='white',
                     fontweight='bold', rotation=90,
                     clip_on=True)
    else:
        # ── Too many flagged bars: show a summary box, no individual labels ───
        from collections import Counter
        type_counts = Counter(
            _issue_type(r) for _, r in sorted_df.iterrows() if _issue_type(r)
        )
        type_labels = {'tilt': ('Tilt', '#cc0000'),
                       'fmap': ('DWI-fmap', '#cc0000'),
                       'session': ('Session', '#e07b00'),
                       'pe': ('PE-pair', '#9933cc')}
        lines = [f'Flagged files: {n_flagged_total} / {len(sorted_df)}', '']
        for key, (name, _) in type_labels.items():
            if type_counts.get(key, 0):
                lines.append(f'  {name}: {type_counts[key]}')
        lines += ['', '→ see TSV for subject details']
        summary = '\n'.join(lines)
        ax2.text(0.98, 0.97, summary,
                 transform=ax2.transAxes,
                 fontsize=8.5, va='top', ha='right',
                 family='monospace',
                 bbox=dict(boxstyle='round,pad=0.5', fc='white',
                           ec='#cccccc', lw=1.0, alpha=0.92))

    ax2.text(0.01, 0.99, f'N = {len(sorted_df)} files  (sorted by tilt)',
             transform=ax2.transAxes, fontsize=7, color='#999999',
             va='top', ha='left')

    ax2.set_ylabel(f'Tilt from dataset-median normal (°)', fontsize=10)
    ax2.set_title('Per-File Slice Tilt  (sorted ascending)', fontsize=11, pad=8)
    ax2.grid(True, alpha=0.18, axis='y')
    ax2.set_xlim(-0.7, len(sorted_df) - 0.3)
    ax2.set_ylim(bottom=0)

    # ── Figure-level legend — placed below both panels ────────────────────────
    legend_elems = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555',
               markersize=9, label='DWI main (dwi/)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#555',
               markersize=9, label='Rev-PE b0 (fmap/ epi)'),
        mpatches.Patch(fill=False, edgecolor='darkorange', linestyle='--',
                       label=f'{threshold_deg}° tilt threshold'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w',
               markeredgecolor='#cc0000', markeredgewidth=2, markersize=10,
               label='Tilt outlier / DWI-fmap mismatch'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w',
               markeredgecolor='#e07b00', markeredgewidth=2, markersize=10,
               label='Session inconsistency'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='w',
               markeredgecolor='#9933cc', markeredgewidth=2, markersize=10,
               label='PE pair issue'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=3,
               fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.0))

    # ── Suptitle ─────────────────────────────────────────────────────────────
    src_counts = pdf['orientation_source'].value_counts().to_dict()
    src_str    = ', '.join(f'{cnt}× {src}' for src, cnt in src_counts.items())

    warn_parts = []
    for lbl, key in [('tilt', 'tilt'), ('PE-pair', 'pe_pair'),
                     ('DWI-fmap', 'dwi_fmap'), ('session', 'session')]:
        cnt = warn_counts.get(key, 0)
        if cnt:
            warn_parts.append(f'{cnt} {lbl}')
    warn_str = ('  \u26a0  ' + ', '.join(warn_parts)) if warn_parts else '  \u2713 all checks passed'

    fig.suptitle(
        f'DWI / fmap-EPI Acquisition Orientation Check\n'
        f'{len(pdf)} files  ({src_str}){warn_str}',
        fontsize=13, fontweight='bold', y=0.99,
    )
    # rect: [left, bottom, right, top] — bottom margin for the figure legend
    plt.tight_layout(rect=[0, 0.09, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Scatter plot saved → {out_path}')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def check_dwi_orientation(
    bids_root,
    output_prefix: str = 'dwi_orientation_check',
    threshold_deg: float = 5.0,
    plot: bool = True,
    log_file: str | None = None,
) -> pd.DataFrame:
    """Check DWI and fmap EPI acquisition orientation across a BIDS dataset.

    Four checks are performed (all use the same threshold_deg):
      1. Tilt outliers           — slice normal vs. dataset-median
      2. PE direction pairs      — valid AP/PA (or RL/LR) pairs in fmap/
      3. DWI ↔ fmap-EPI match   — orientation consistency via IntendedFor
      4. Session consistency     — same acquisition across all sessions

    Parameters
    ----------
    bids_root      : path-like — BIDS root directory
    output_prefix  : str — prefix for <prefix>.tsv and <prefix>.png
    threshold_deg  : float — angular deviation threshold for all checks (degrees)
    plot           : bool — generate PNG scatter plot (requires matplotlib)
    log_file       : str | None — path for log file; None = no log file.
                     When not specified via this argument, the CLI uses
                     <output_prefix>.log automatically.

    Returns
    -------
    pd.DataFrame  — one row per DWI / fmap-EPI file found in the dataset
    """
    bids_root = Path(bids_root).resolve()
    with _tee_stdout(log_file):
        return _run_checks(bids_root, output_prefix, threshold_deg, plot)


def _run_checks(bids_root: Path, output_prefix: str,
                threshold_deg: float, plot: bool) -> pd.DataFrame:
    """Internal: all checks + output, assumes stdout already redirected."""
    print(f'\nDWI / fmap-EPI orientation check')
    print(f'  BIDS root : {bids_root}')
    print(f'  Threshold : {threshold_deg}°  (all checks)')

    records = collect_orientation_records(bids_root)
    if not records:
        print('  No DWI or fmap EPI files found.')
        return pd.DataFrame()

    df = pd.DataFrame(records)
    internal_cols = [c for c in df.columns if c.startswith('_')]

    n_files = len(df)
    has_normal = df['_nx'].apply(lambda x: not math.isnan(x))
    print(f'  Files found : {n_files}  ({has_normal.sum()} with usable orientation data)')
    for src, cnt in df['orientation_source'].value_counts().items():
        print(f'    {cnt}× via {src}')

    # ── Run all checks ────────────────────────────────────────────────────────
    if has_normal.any():
        _check_tilt_outliers(df, threshold_deg)
    _check_pe_pairs(df)
    _check_dwi_fmap_consistency(df, threshold_deg)
    _check_session_consistency(df, threshold_deg)

    # ── Console report ────────────────────────────────────────────────────────
    warn_counts = {
        'tilt':    _report_tilt(df, threshold_deg),
        'pe_pair': _report_pe_pairs(df),
        'dwi_fmap':_report_dwi_fmap(df, threshold_deg),
        'session': _report_session_consistency(df, threshold_deg),
    }

    total_warnings = sum(warn_counts.values())
    print(f'\n  Summary: {total_warnings} total issue(s) across all checks.')

    # ── Save TSV (drop internal _ columns) ───────────────────────────────────
    df_save = df.drop(columns=internal_cols, errors='ignore')
    out_tsv = output_prefix + '.tsv'
    df_save.to_csv(out_tsv, sep='\t', index=False)
    print(f'  Report → {out_tsv}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    if plot and has_normal.any():
        _make_plot(df, output_prefix + '.png', threshold_deg, warn_counts)

    return df_save


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog='dwi-orientation-check',
        description=(
            'Check DWI and fmap EPI acquisition orientation across a BIDS dataset.\n\n'
            'Checks:\n'
            '  1. Tilt outliers        (slice normal vs. dataset median)\n'
            '  2. PE direction pairs   (valid AP/PA reversal in fmap/)\n'
            '  3. DWI ↔ fmap-EPI      (consistent orientation via IntendedFor)\n'
            '  4. Session consistency  (same acquisition across sessions)\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'bidsdir', nargs='?', default='.',
        help='Path to BIDS root directory (default: current directory)',
    )
    parser.add_argument(
        '-o', '--output', default='dwi_orientation_check',
        help='Output file prefix for .tsv and .png (default: dwi_orientation_check)',
    )
    parser.add_argument(
        '-t', '--threshold', type=float, default=5.0,
        help='Angular deviation threshold in degrees for all checks (default: 5.0)',
    )
    parser.add_argument(
        '--no-plot', action='store_true',
        help='Skip the PNG scatter plot',
    )
    parser.add_argument(
        '--log', default=None, metavar='FILE',
        help='Log file path (default: <output>.log)',
    )
    parser.add_argument(
        '--no-log', action='store_true',
        help='Disable log file output entirely',
    )
    args = parser.parse_args()

    bids_root = Path(args.bidsdir).resolve()
    if not bids_root.is_dir():
        print(f'Error: {bids_root} is not a directory', file=sys.stderr)
        sys.exit(1)

    if args.no_log:
        log_path = None
    elif args.log:
        log_path = args.log
    else:
        log_path = args.output + '.log'

    check_dwi_orientation(
        bids_root=bids_root,
        output_prefix=args.output,
        threshold_deg=args.threshold,
        plot=not args.no_plot,
        log_file=log_path,
    )
    if log_path:
        print(f'  Log       → {log_path}')


if __name__ == '__main__':
    main()
