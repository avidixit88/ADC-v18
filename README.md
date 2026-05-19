# Invenra ADC Capital Map — Layer 1.9 Registry v2

Premium Streamlit application for mapping clinical-trial activity across ADC-relevant targets.

## Layer 1.9 changes

- Expanded ADC target registry to 87 relevant targets.
- Added registry metadata fields:
  - target category
  - indication focus
  - registry confidence
  - asset expansion status
  - source basis
- Hardened aliases with gene symbols, common name variants, and known ADC asset names where available.
- Added audit-ready registry table in the app.
- Preserved Layer 1.8 premium header, controls, future-start calendar, basket momentum, sponsor-by-target, and trial evidence views.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data layer

The target registry is currently stored in `data/adc_targets.csv` for portability. The app structure is intentionally modular so this registry can later be moved into Postgres, Supabase, or a scheduled backend service without rewriting the UI.

## Layer 1.5 additions

This build adds inferred clinical intelligence derived from ClinicalTrials.gov protocol fields and text fields:

- Modality classification (ADC, bispecific/biparatopic ADC, radioconjugate, CAR-T/cellular therapy, bispecific/TCE, antibody/mAb, small molecule/other)
- Payload-family hints when trial text exposes recognizable terms (DXd/Topo-I, SN-38, MMAE, MMAF, maytansinoid, PBD, duocarmycin, calicheamicin)
- Combination strategy detection (checkpoint, chemo, targeted, radiation, multi-agent)
- Line-of-therapy inference (frontline, 2L, 3L+, relapsed/refractory, adjuvant/maintenance)
- Sponsor conviction scoring by target and sponsor
- Asset-level intervention tracking
- Termination/withdrawal/suspension risk signals

These are directional intelligence fields. They should be treated as protocol-text-derived signals and later enriched with PubMed, company pipelines, press releases, conference data, and manual scientific review.

## Layer 1.11 Clinical Intelligence Upgrade

This build deepens the Clinical Intelligence page while still staying inside the ClinicalTrials.gov record. Added:

- Confirmed ADC vs likely ADC vs multispecific/biparatopic ADC modality decomposition
- Target exploitation profile with ADC share, combo share, dominant modality, modality diversity, and profile readout
- Crowding and whitespace pressure score based on active trials, sponsor density, Phase 2/3 density, enrollment, and modality diversity
- Target evolution timeline by year, including ADC-inferred studies, combo trials, sponsors, phase maturity, stopped studies, and enrollment
- Named combination-agent extraction for checkpoint, chemo, PARP, VEGF, HER2-directed, and radiation patterns
- Preserved sponsor conviction, line-of-therapy, stopped-study, and asset-level tracking views


## Layer 1.12 patch notes

- Replaced deprecated Streamlit `use_container_width` calls with `width="stretch"`.
- Added an active/upcoming scope toggle to the Clinical Intelligence page so the page can focus on current capital deployment while preserving stopped-study intelligence.
- Added modality confidence and modality evidence fields to trial-level outputs.
- Renamed the summary card from `ADC-inferred` to `ADC-classified` to better communicate that the metric includes confirmed and likely ADC classifications.
- Kept stopped/terminated trial analytics based on the full retrieved dataset even when active scope is enabled.

## Layer 1.13 Clinical Intelligence Refinement

- Split broad radioconjugate inference into three separate buckets:
  - Therapeutic radioconjugate
  - Radiolabeled imaging / diagnostic
  - Possible radioconjugate
- Tightened radiopharmaceutical logic so PET/SPECT/tracer/biodistribution studies do not inflate therapeutic radioconjugate activity.
- Added dedicated Clinical Intelligence summary cards for therapeutic radio signal vs imaging/diagnostic signal.
- Tightened combination detection so single-agent ADC names like trastuzumab deruxtecan are less likely to be mislabeled as targeted combinations solely because they contain a targeted antibody name.
- Updated UI explanatory text to make clear that modality labels are ClinicalTrials.gov text-derived inference signals.

## Layer 1.14 Clinical Intelligence UI/performance patch
- Added explicit, high-contrast modality color mapping so categories remain distinguishable in large basket scans.
- Separated chart color handling for modality-colored charts versus target-colored charts.
- Added solid dark Plotly backgrounds and high-resolution PNG export options so expanded/full-screen charts are cleaner for screenshots and PowerPoint use.
- Cached the Clinical Intelligence computation bundle to reduce re-computation lag when switching tabs or toggling chart scope.
- Capped the heaviest basket charts to top targets for readability and browser performance while preserving full detail in the tables.
