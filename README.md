# Naming the birds BirdNET doesn't know

Code and data to reproduce the main results of the paper "Naming the birds BirdNET doesn't know: a few-shot
classifier and confidence-based routing for tropical species" by Molion Surya Pradana and Chuan-Kai Yang.

BirdNET has a fixed label set, so many birds in species-rich tropical regions are outside it, and BirdNET
still names a known species for every recording. We train a second classifier on a few local recordings
per species, using only BirdNET's frozen embeddings and scores, and route between BirdNET and this
classifier at a fixed wrong-answer rate. The benchmarks are built from public Xeno-canto recordings from
Indonesia, the Philippines, Malaysia and Taiwan.

## Contents

| Path | Contents |
|---|---|
| `scripts/` | Scripts for the results listed below |
| `data/bench_files_*.csv` | Benchmark recordings per region: species, Xeno-canto catalogue number, media link, licence of each recording, anonymised recordist ID, rounded coordinates |
| `data/model_outputs/` | BirdNET and Perch 2.0 outputs for every benchmark recording (scores, embeddings), BirdNET range lists, and the cross-fitted outputs of the extension classifier |
| `data/name_reconciliation/` | For each region, every benchmark species, whether it is in BirdNET's label set, and any synonym link found in the IOC cross-reference |
| `external/` | IOC World Bird List v15.2 cross-reference workbook (worldbirdnames.org, CC BY 3.0) |
| `CONSERVATION_STATUS.md` | IUCN Red List check and the species whose coordinates are withheld |

All results below are computed from `data/model_outputs/`, so no audio and no GPU are needed.

## Setup

Python 3.11 with the package versions in `requirements.txt`. Other versions of numpy, scipy or
scikit-learn can change some values in the third decimal.

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the results

Scripts locate the data relative to their own folder, so they can be run from any directory.

| Result in the paper | Command |
|---|---|
| Table 1, Indonesia | `REGION=ID python scripts/table1_birdnet_ID_PH.py` |
| Table 1, Philippines | `python scripts/table1_birdnet_ID_PH.py` |
| Table 1, Malaysia | `python scripts/table1_birdnet_MY.py` |
| Table 1, Perch 2.0 columns | `REGION=ID python scripts/table1_perch2.py` (also `REGION=PH`, `REGION=MY`) |
| Table 2 | `python scripts/table2_range_filter.py` |
| Figure 3 | `python scripts/figure3_data.py && python scripts/figure3_plot.py` (writes `fig_hybrid_curves.pdf`) |
| Section 3.1, crosswalk counts | `python scripts/taxonomy_crosswalk.py` |
| Section 3.1, sensitivity to name reconciliation | `python scripts/check_name_reconciliation.py` |
| Section 4.3, logistic regression instead of the SVM | `REGION=ID CLF=lr python scripts/check_logistic_regression.py` (also `PH`, `MY`) |
| Section 4.3, method of Jana et al. (2026) | `python scripts/compare_jana_method.py` |
| Name-reconciliation tables in `data/name_reconciliation/` | `python scripts/build_name_reconciliation_tables.py` |

<!-- TODO before publishing: update this table to the final version of the paper. -->

## Getting the audio

The recordings are not redistributed, because each one carries its own Creative Commons licence and some
licences do not allow derivatives. Every recording can be downloaded from Xeno-canto with its
`catalogNumber` (for example `XC205219` is https://xeno-canto.org/205219) or from the `media_url` column.
Audio is only needed to recompute the model outputs.

## Data notes

- Recordist names are replaced by anonymous IDs (`rec_000001`, ...). The IDs follow the alphabetical
  order of the real names, so the recordist-disjoint splits are the same as in the paper.
- Coordinates are rounded to 0.1 degree. For the 136 species assessed Near Threatened or worse on the
  IUCN Red List, coordinates are removed (see `CONSERVATION_STATUS.md`). Table 2 is computed from the range
  lists in `data/model_outputs/`, which were produced from the exact coordinates.
- Recordings come from the Xeno-canto occurrence dataset on GBIF (https://doi.org/10.15468/qv0ksn).

## Licence

Code in `scripts/` is released under the MIT License (`LICENSE`). The tables in `data/` are derived from
the Xeno-canto dataset on GBIF (CC BY-NC 4.0) and are released under the same terms. Individual
recordings keep their own licences (column `media_license`). The IOC workbook in `external/` is CC BY 3.0.

## Citation

Please cite the paper (reference to be added on publication).
