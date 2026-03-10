# BIDS-Inspector — TODO

## bids_inspector.py

### #4 — JSON field consistency checker
Scan key JSON fields across subjects and flag deviations from the dataset mode.
Fields of interest: `RepetitionTime`, `EchoTime`, `TotalReadoutTime`,
`MultibandAccelerationFactor`, `EffectiveEchoSpacing`, `FlipAngle`.
Same outlier-detection logic as the existing `_dim4_ok` column:
compute mode across subjects, flag anyone who deviates.
Useful for catching scanner protocol drift between sessions or subjects.

### #5 — SliceTiming length vs. NIfTI slice count
`SliceTiming` in the JSON sidecar must have exactly as many entries as slices
(NIfTI dim3). Mismatches silently break slice-timing correction in fMRIPrep,
FSL FEAT, etc.
Implementation: read `len(SliceTiming)` from JSON, compare to `img.shape[2]`.

### #6 — fmap `IntendedFor` validation
Check that every path listed in `IntendedFor` actually exists in the dataset.
Missing targets cause cryptic failures in fMRIPrep / QSIPrep.
Report: list of fmap files whose IntendedFor targets are absent.

## New script: bids_summary_plot.py

### #7 — Per-subject summary heatmap
A single figure (subjects × checks) where each cell is coloured green/yellow/red.
Combines results from bids_inspector.py and dwi_orientation_check.py into one
at-a-glance QC overview.
Rows = subjects (or subject/session combinations).
Columns = checks (completeness, dim4_ok, dwi_ok, tilt_outlier, pe_pair_status,
dwi_fmap_consistent, session_consistent, …).
Requires: matplotlib, pandas.
