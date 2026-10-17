# PDGuard AI

Partial discharge risk monitor for GIS, power transformers, shunt reactors and cables.
Part of the PowerGuard AI platform. Confidential and proprietary.

## What it does
- Reads the CSV export of the online PD monitoring system (UHF / HFCT): peak level, pulse rate and the PRPD phase profile.
- Classifies the defect type from the PRPD pattern (corona, floating electrode, void, surface discharge, free particle, external noise).
- Scores the risk from the level above the sensor background, the trend, the pulse rate and the defect severity at that equipment.
- Confirms with other methods: acoustic tests, SF6 decomposition tests and the online DGA monitor of transformers and reactors.
- Suppresses interference that the noise reference antenna also sees.
- FMEA (RPN = S × O × D), asset ranking, diagnostic report and a historical study that measures how early a failure would have been flagged.

## Files
| File | Purpose |
|---|---|
| `PDGuard_AI.app.py` | Streamlit app (main file) |
| `pdguard_core.py` | Risk engine, demo substation and historical study |
| `train_classifier.py` | Trains `pd_classifier.pkl` |
| `pd_classifier.pkl` | PRPD defect classifier (Random Forest) |

## Run
```
pip install -r requirements.txt
streamlit run PDGuard_AI.app.py
```
Login credentials are read from Streamlit secrets: `APP_USERNAME` and `APP_PASSWORD` (never in the code).

## Note on the classifier
The classifier is trained on synthetic PRPD patterns built from the published signature of each defect type.
It must be re-trained on labelled field data before any accuracy figure is quoted.
