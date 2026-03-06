# BIDS-Inspector

Validate and summarise a [BIDS](https://bids-specification.readthedocs.io/) dataset. Scans all subjects and sessions, auto-discovers every acquisition type, and produces a single CSV showing what is present, what is missing, and where data looks off.

## What it checks

| Check | Columns produced |
|---|---|
| NIfTI + JSON sidecar present | `<acq>` (1/0), `<acq>_nii` (kept only when JSON status differs) |
| 4th dimension (func bold, DWI, fmap EPI) | `<acq>_dim4`, `<acq>_dim4_ok` (0 = deviates from dataset mode) |
| events.tsv for functional tasks | `<acq>_events` (1/0), `<acq>_events_nrows` |
| bval/bvec for DWI | `<acq>_bval`, `<acq>_bval_n`, `<acq>_bvec`, `<acq>_bvec_n`, `<acq>_dwi_ok` |
| participants.tsv | Merged automatically when present (sex, age, etc.) |

Columns that are never relevant (e.g. events.tsv when no task has one, or `_nii` when JSON is always present) are dropped automatically to keep the output clean.

## Requirements

- Python 3.10+
- nibabel
- pandas

## Usage

```bash
# Basic usage
python3 bids_inspector.py /path/to/bids

# Custom output file
python3 bids_inspector.py /path/to/bids -o results.csv

# Skip NIfTI header reading (faster, no dim4 columns)
python3 bids_inspector.py /path/to/bids --no-dim4

# Parallel processing (4 threads)
python3 bids_inspector.py /path/to/bids -j 4

# Custom log file location
python3 bids_inspector.py /path/to/bids --log run.log
```

## Output

- **CSV** (default: `bids_inspector.csv`) with one row per subject/session
- **Log** (default: `bids_inspector.log`) mirrors all console output, including warnings about dim4 outliers or DWI mismatches

## Example output (truncated)

```
subject  sex  age  func_task-rest_run-01_bold  func_task-rest_run-01_bold_dim4  func_task-rest_run-01_bold_dim4_ok  anat_T1w  dwi_dwi  ...
sub-01   F    26   1                           300                              1                                   1         1        ...
sub-02   M    24   1                           300                              1                                   1         1        ...
sub-03   F    27   1                           298                              0                                   1         0        ...
```

## JSON sidecar resolution

Follows BIDS inheritance: checks for a matching `.json` in the same directory, then walks up through session, subject, and dataset root levels.

## Tested with

- [ds000001](https://openneuro.org/datasets/ds000001) — single-session, functional task (Balloon Analog Risk Task), 16 subjects
- [ds000221](https://openneuro.org/datasets/ds000221) — multi-session, DWI + resting-state fMRI + fieldmaps, 24 subjects

---

Built with [Claude Code](https://claude.ai/claude-code).
