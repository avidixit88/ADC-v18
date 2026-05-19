from __future__ import annotations

from datetime import datetime
import re
import pandas as pd

ACTIVE_STATUSES = {"RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION", "NOT_YET_RECRUITING"}
LATE_PHASES = {"PHASE2", "PHASE3", "PHASE2_PHASE3"}
PHASE_ORDER = ["EARLY_PHASE1", "PHASE1", "PHASE1_PHASE2", "PHASE2", "PHASE2_PHASE3", "PHASE3", "PHASE4", "NOT_APPLICABLE", "UNKNOWN"]
PHASE_LABELS = {
    "EARLY_PHASE1": "Early Phase 1",
    "PHASE1": "Phase 1",
    "PHASE1_PHASE2": "Phase 1/2",
    "PHASE2": "Phase 2",
    "PHASE2_PHASE3": "Phase 2/3",
    "PHASE3": "Phase 3",
    "PHASE4": "Phase 4",
    "NOT_APPLICABLE": "Not applicable",
    "UNKNOWN": "Unspecified phase",
    "": "Unspecified phase",
}


def normalize_phase(raw) -> str:
    """Turn ClinicalTrials.gov phase strings/lists into one canonical bucket."""
    if raw is None or pd.isna(raw):
        return "UNKNOWN"
    s = str(raw).strip().upper()
    if not s or s in {"N/A", "NA", "NONE", "NULL", "UNKNOWN"}:
        return "UNKNOWN"
    tokens = [t.strip() for t in re.split(r"[;,|/]+", s) if t.strip()]
    joined = " ".join(tokens)
    has_early = "EARLY_PHASE1" in joined or "EARLY PHASE 1" in joined
    has_p1 = has_early or "PHASE1" in joined or "PHASE 1" in joined
    has_p2 = "PHASE2" in joined or "PHASE 2" in joined
    has_p3 = "PHASE3" in joined or "PHASE 3" in joined
    has_p4 = "PHASE4" in joined or "PHASE 4" in joined
    if "NOT_APPLICABLE" in joined or "NOT APPLICABLE" in joined:
        return "NOT_APPLICABLE"
    if has_p2 and has_p3:
        return "PHASE2_PHASE3"
    if has_p1 and has_p2:
        return "PHASE1_PHASE2"
    if has_early:
        return "EARLY_PHASE1"
    if has_p4:
        return "PHASE4"
    if has_p3:
        return "PHASE3"
    if has_p2:
        return "PHASE2"
    if has_p1:
        return "PHASE1"
    cleaned = s.replace("; ", "_").replace(" ", "_").replace("-", "_")
    return cleaned if cleaned in PHASE_LABELS else "UNKNOWN"


def add_activity_flags(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if d.empty:
        return d
    d["overall_status"] = d.get("overall_status", pd.Series(dtype=str)).fillna("").astype(str)
    d["phase_bucket"] = d.get("phases", pd.Series(dtype=str)).apply(normalize_phase)
    d["phase_label"] = d["phase_bucket"].map(PHASE_LABELS).fillna(d["phase_bucket"].str.replace("_", " ").str.title())
    d["is_active"] = d["overall_status"].isin(ACTIVE_STATUSES)
    d["is_recruiting"] = d["overall_status"].eq("RECRUITING")
    d["has_late_phase"] = d["phase_bucket"].apply(lambda x: any(p == str(x) or p in str(x) for p in LATE_PHASES))
    d["enrollment_count"] = pd.to_numeric(d.get("enrollment_count", 0), errors="coerce").fillna(0).astype(int)
    if "start_year" not in d.columns:
        d["start_year"] = pd.to_datetime(d.get("start_date", pd.Series(dtype=str)), errors="coerce").dt.year
    return d


def target_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    d = add_activity_flags(df)
    grouped = d.groupby("target", dropna=False).agg(
        total_trials=("nct_id", "nunique"),
        active_trials=("is_active", "sum"),
        recruiting_trials=("is_recruiting", "sum"),
        cumulative_enrollment=("enrollment_count", "sum"),
        active_enrollment=("enrollment_count", lambda s: int(s[d.loc[s.index, "is_active"]].sum())),
        avg_enrollment=("enrollment_count", "mean"),
        sponsor_count=("lead_sponsor", "nunique"),
        country_count=("countries", lambda s: len(set("; ".join(s.dropna()).split("; "))) if len(s.dropna()) else 0),
        late_phase_trials=("has_late_phase", "sum"),
        latest_update=("last_update_posted", "max"),
        earliest_start=("start_date", "min"),
    ).reset_index()
    grouped["heat_score"] = (
        grouped["active_trials"] * 3
        + grouped["recruiting_trials"] * 4
        + grouped["late_phase_trials"] * 5
        + grouped["sponsor_count"] * 1.5
        + grouped["country_count"] * 0.5
        + (grouped["active_enrollment"] / 100).clip(upper=30)
    ).round(1)
    return grouped.sort_values("heat_score", ascending=False)


def phase_distribution(df: pd.DataFrame, active_only: bool = False, include_unspecified: bool = False) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["phase_bucket", "phase_label", "count", "enrollment"])
    d = add_activity_flags(df)
    if active_only:
        d = d[d["is_active"]]
    if not include_unspecified:
        d = d[~d["phase_bucket"].isin(["UNKNOWN", "NOT_APPLICABLE"])]
    grouped = d.groupby(["phase_bucket", "phase_label"], dropna=False).agg(
        count=("nct_id", "nunique"),
        enrollment=("enrollment_count", "sum"),
    ).reset_index()
    grouped["phase_sort"] = grouped["phase_bucket"].apply(lambda x: PHASE_ORDER.index(x) if x in PHASE_ORDER else 99)
    grouped = grouped.sort_values("phase_sort")
    return grouped[["phase_bucket", "phase_label", "count", "enrollment"]]


def status_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["overall_status", "count"])
    d = add_activity_flags(df)
    return d.groupby("overall_status").size().reset_index(name="count").sort_values("count", ascending=False)


def yearly_trial_momentum(df: pd.DataFrame, current_year: int | None = None, include_future: bool = False) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["start_year", "new_trials", "active_trials", "enrollment"])
    current_year = current_year or datetime.now().year
    d = add_activity_flags(df)
    d = d.dropna(subset=["start_year"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["start_year", "new_trials", "active_trials", "enrollment"])
    d["start_year"] = d["start_year"].astype(int)
    if not include_future:
        d = d[d["start_year"] <= current_year]
    grouped = d.groupby("start_year").agg(
        new_trials=("nct_id", "nunique"),
        active_trials=("is_active", "sum"),
        enrollment=("enrollment_count", "sum"),
    ).reset_index().sort_values("start_year")
    grouped["yoy_delta"] = grouped["new_trials"].diff().fillna(0).astype(int)
    grouped["yoy_pct"] = grouped["new_trials"].pct_change().replace([float("inf"), -float("inf")], pd.NA)
    return grouped


def yoy_summary(df: pd.DataFrame, current_year: int | None = None) -> dict[str, int | float | str | None]:
    current_year = current_year or datetime.now().year
    momentum = yearly_trial_momentum(df, current_year=current_year, include_future=False)
    if momentum.empty:
        return {"this_year": 0, "last_year": 0, "delta": 0, "pct": None, "label": "No dated starts", "current_year": current_year, "last_year_label": current_year - 1}
    this_year = int(momentum.loc[momentum["start_year"].eq(current_year), "new_trials"].sum())
    last_year = int(momentum.loc[momentum["start_year"].eq(current_year - 1), "new_trials"].sum())
    delta = this_year - last_year
    pct = None if last_year == 0 else delta / last_year
    return {"this_year": this_year, "last_year": last_year, "delta": delta, "pct": pct, "label": f"{current_year} YTD vs {current_year - 1}", "current_year": current_year, "last_year_label": current_year - 1}


def target_year_heatmap(df: pd.DataFrame, current_year: int | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    current_year = current_year or datetime.now().year
    d = add_activity_flags(df).dropna(subset=["start_year"]).copy()
    if d.empty:
        return pd.DataFrame()
    d["start_year"] = d["start_year"].astype(int)
    d = d[d["start_year"] <= current_year]
    return d.pivot_table(index="target", columns="start_year", values="nct_id", aggfunc="nunique", fill_value=0)


def target_phase_heatmap(df: pd.DataFrame, active_only: bool = True) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    d = add_activity_flags(df)
    if active_only:
        d = d[d["is_active"]]
    d = d[~d["phase_bucket"].isin(["UNKNOWN", "NOT_APPLICABLE"])]
    if d.empty:
        return pd.DataFrame()
    table = d.pivot_table(index="target", columns="phase_label", values="nct_id", aggfunc="nunique", fill_value=0)
    phase_cols = [PHASE_LABELS[p] for p in PHASE_ORDER if PHASE_LABELS[p] in table.columns]
    remaining = [c for c in table.columns if c not in phase_cols]
    return table[phase_cols + remaining]


def sponsor_activity(df: pd.DataFrame, active_only: bool = True, limit: int = 12, by_target: bool = False) -> pd.DataFrame:
    if df.empty:
        cols = ["target", "lead_sponsor", "trials", "active_trials", "enrollment"] if by_target else ["lead_sponsor", "trials", "active_trials", "enrollment", "targets"]
        return pd.DataFrame(columns=cols)
    d = add_activity_flags(df)
    if active_only:
        d = d[d["is_active"]]
    if d.empty:
        cols = ["target", "lead_sponsor", "trials", "active_trials", "enrollment"] if by_target else ["lead_sponsor", "trials", "active_trials", "enrollment", "targets"]
        return pd.DataFrame(columns=cols)
    if by_target:
        grouped = d.groupby(["target", "lead_sponsor"], dropna=False).agg(
            trials=("nct_id", "nunique"),
            active_trials=("is_active", "sum"),
            enrollment=("enrollment_count", "sum"),
        ).reset_index()
        return grouped.sort_values(["target", "active_trials", "enrollment"], ascending=[True, False, False]).groupby("target").head(limit).reset_index(drop=True)
    grouped = d.groupby("lead_sponsor", dropna=False).agg(
        trials=("nct_id", "nunique"),
        active_trials=("is_active", "sum"),
        enrollment=("enrollment_count", "sum"),
        targets=("target", lambda s: "; ".join(sorted(set(s.dropna())))[:140]),
    ).reset_index()
    return grouped.sort_values(["active_trials", "enrollment", "trials"], ascending=False).head(limit)


def target_momentum_table(df: pd.DataFrame, current_year: int | None = None) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["target", "new_trials_current_ytd", "new_trials_prior_year", "yoy_delta", "active_trials", "active_enrollment", "phase2plus_active"])
    current_year = current_year or datetime.now().year
    d = add_activity_flags(df)
    d = d.dropna(subset=["start_year"]).copy()
    d["start_year"] = d["start_year"].astype("Int64")
    d = d[d["start_year"] <= current_year]
    rows = []
    for target, tdf in d.groupby("target"):
        cur = int(tdf.loc[tdf["start_year"].eq(current_year), "nct_id"].nunique())
        prev = int(tdf.loc[tdf["start_year"].eq(current_year - 1), "nct_id"].nunique())
        active = tdf[tdf["is_active"]]
        phase2plus = int(active[active["phase_bucket"].isin(["PHASE2", "PHASE2_PHASE3", "PHASE3"])] ["nct_id"].nunique())
        rows.append({
            "target": target,
            "new_trials_current_ytd": cur,
            "new_trials_prior_year": prev,
            "yoy_delta": cur - prev,
            "active_trials": int(active["nct_id"].nunique()),
            "active_enrollment": int(active["enrollment_count"].sum()),
            "phase2plus_active": phase2plus,
        })
    return pd.DataFrame(rows).sort_values(["yoy_delta", "active_enrollment"], ascending=False)


def future_start_table(df: pd.DataFrame, current_date: datetime | None = None) -> pd.DataFrame:
    """Future-dated estimated starts, grouped for the forward-looking trial calendar.

    ClinicalTrials.gov exposes startDateStruct.date and startDateStruct.type.
    We treat a record as a forward estimated start when the date is after today
    and the date type is ESTIMATED. Current-year future starts are labeled
    "Later this year" to avoid making the UI look like a full-year historical
    comparison.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "target", "start_year", "start_period", "estimated_future_starts",
            "earliest_estimated_start", "statuses", "top_sponsors", "total_planned_enrollment"
        ])
    current_date = current_date or datetime.now()
    today = pd.Timestamp(current_date.date())
    current_year = current_date.year
    d = add_activity_flags(df).copy()
    d["start_dt"] = pd.to_datetime(d.get("start_date", pd.Series(dtype=str)), errors="coerce")
    d["start_date_type"] = d.get("start_date_type", pd.Series(dtype=str)).fillna("").astype(str).str.upper()
    d = d.dropna(subset=["start_dt"])
    d = d[(d["start_dt"] > today) & (d["start_date_type"].eq("ESTIMATED"))]
    if d.empty:
        return pd.DataFrame(columns=[
            "target", "start_year", "start_period", "estimated_future_starts",
            "earliest_estimated_start", "statuses", "top_sponsors", "total_planned_enrollment"
        ])
    d["start_year"] = d["start_dt"].dt.year.astype(int)
    rows = []
    for (target, year), g in d.groupby(["target", "start_year"], dropna=False):
        period = f"Later this year ({current_year})" if int(year) == current_year else str(int(year))
        sponsors = [x for x in g["lead_sponsor"].dropna().astype(str).unique().tolist() if x]
        statuses = [x for x in g["overall_status"].dropna().astype(str).unique().tolist() if x]
        rows.append({
            "target": target,
            "start_year": int(year),
            "start_period": period,
            "estimated_future_starts": int(g["nct_id"].nunique()),
            "earliest_estimated_start": g["start_dt"].min().strftime("%Y-%m-%d"),
            "statuses": "; ".join(sorted(statuses))[:180],
            "top_sponsors": "; ".join(sponsors[:4])[:220],
            "total_planned_enrollment": int(g["enrollment_count"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["start_year", "estimated_future_starts", "total_planned_enrollment"], ascending=[True, False, False]).reset_index(drop=True)


def future_start_summary(df: pd.DataFrame, current_date: datetime | None = None) -> dict[str, int]:
    current_date = current_date or datetime.now()
    table = future_start_table(df, current_date=current_date)
    if table.empty:
        return {"future_starts": 0, "later_this_year": 0, "future_years": 0}
    current_year = current_date.year
    return {
        "future_starts": int(table["estimated_future_starts"].sum()),
        "later_this_year": int(table.loc[table["start_year"].eq(current_year), "estimated_future_starts"].sum()),
        "future_years": int(table.loc[table["start_year"].gt(current_year), "start_year"].nunique()),
    }

TOP_PHARMA_TERMS = [
    "AstraZeneca", "Daiichi", "Merck", "Roche", "Genentech", "Pfizer", "Bristol", "BMS",
    "Johnson", "Janssen", "AbbVie", "Gilead", "Novartis", "Eli Lilly", "Sanofi", "Amgen",
    "Boehringer", "Takeda", "Astellas", "Bayer", "GSK", "Glaxo", "Regeneron", "BeiGene",
    "BioNTech", "Seagen", "Genmab", "Ipsen", "Bicycle", "Mersana", "Sutro", "ImmunoGen"
]

CHECKPOINT_AGENTS = ["pembrolizumab", "keytruda", "nivolumab", "opdivo", "atezolizumab", "durvalumab", "avelumab", "cemiplimab", "ipilimumab"]
CHEMO_TERMS = ["carboplatin", "cisplatin", "paclitaxel", "docetaxel", "gemcitabine", "irinotecan", "etoposide", "pemetrexed", "capecitabine", "folfox", "folfiri"]
TARGETED_COMBO_TERMS = ["olaparib", "bevacizumab", "lenvatinib", "osimertinib", "tucatinib", "pertuzumab", "trastuzumab", "rituximab", "cetuximab"]


def _text_blob(row) -> str:
    fields = [
        "brief_title", "official_title", "brief_summary", "detailed_description", "interventions",
        "intervention_types", "arm_groups", "eligibility_criteria", "primary_outcomes", "secondary_outcomes", "conditions"
    ]
    return " | ".join(str(row.get(f, "")) for f in fields if f in row and pd.notna(row.get(f, ""))).lower()


CONFIRMED_ADC_TERMS = [
    "antibody-drug conjugate", "antibody drug conjugate", "antibody-drug-conjugate",
    " adc", "adc ", "adcs", "immunoconjugate"
]
LIKELY_ADC_TERMS = [
    "-dxd", " dxd", "deruxtecan", "vedotin", "emtansine", "duocarmazine",
    "ozogamicin", "mafodotin", "tesirine", "govitecan", "maytansinoid",
    "pyrrolobenzodiazepine", "calicheamicin", "auristatin"
]
MULTISPECIFIC_ADC_TERMS = [
    "bispecific adc", "bispecific antibody-drug conjugate", "biparatopic",
    "dual epitope", "dual-epitope", "multispecific adc", "trispecific"
]

# Radiolabeled studies are common on ClinicalTrials.gov. Many are diagnostic
# PET/SPECT/biodistribution studies rather than therapeutic radioconjugate
# programs, so we intentionally separate them to avoid overstating
# radiopharmaceutical competition.
THERAPEUTIC_RADIO_TERMS = [
    "radioligand therapy", "radiopharmaceutical therapy", "radioimmunotherapy",
    "targeted radionuclide therapy", "therapeutic radiopharmaceutical",
    "therapeutic radioligand", "radiotherapeutic", "therapeutic radionuclide",
    "alpha therapy", "targeted alpha", "alpha-emitting", "beta-emitting",
    "actinium-225", "ac-225", "225ac", "lutetium-177", "lu-177", "177lu",
    "yttrium-90", "y-90", "90y", "iodine-131", "i-131", "131i"
]
DIAGNOSTIC_RADIO_TERMS = [
    "pet", "spect", "imaging", "diagnostic", "tracer", "biodistribution",
    "radiolabeled", "radio-labeled", "zirconium-89", "zr-89", "89zr",
    "copper-64", "cu-64", "64cu", "gallium-68", "ga-68", "68ga",
    "fluorine-18", "f-18", "18f", "indium-111", "in-111", "111in"
]
AMBIGUOUS_RADIO_TERMS = [
    "radioconjugate", "radioisotope", "radiopharmaceutical", "radioligand", "radioimmun"
]


def _matched_terms(txt: str, terms: list[str]) -> list[str]:
    return [term.strip() for term in terms if term in txt]


def classify_modality(row) -> str:
    """Classify therapeutic modality from ClinicalTrials.gov protocol text.

    This is intentionally conservative. We separate Confirmed ADC from Likely ADC
    and split therapeutic radioconjugates from radiolabeled imaging/diagnostic
    studies so modality counts do not overstate radiopharmaceutical activity.
    """
    txt = _text_blob(row)
    ints = str(row.get("interventions", "")).lower()

    if _matched_terms(txt, MULTISPECIFIC_ADC_TERMS) and (_matched_terms(txt, CONFIRMED_ADC_TERMS) or _matched_terms(txt, LIKELY_ADC_TERMS)):
        return "Multispecific / biparatopic ADC"
    if _matched_terms(txt, CONFIRMED_ADC_TERMS):
        return "Confirmed ADC"
    if _matched_terms(txt, LIKELY_ADC_TERMS):
        return "Likely ADC"

    therapeutic_radio_hits = _matched_terms(txt, THERAPEUTIC_RADIO_TERMS)
    diagnostic_radio_hits = _matched_terms(txt, DIAGNOSTIC_RADIO_TERMS)
    ambiguous_radio_hits = _matched_terms(txt, AMBIGUOUS_RADIO_TERMS)
    if therapeutic_radio_hits:
        return "Therapeutic radioconjugate"
    if diagnostic_radio_hits:
        return "Radiolabeled imaging / diagnostic"
    if ambiguous_radio_hits:
        return "Possible radioconjugate"

    if any(x in txt for x in ["car-t", "cart", "chimeric antigen receptor", "cell therapy", "cellular therapy"]):
        return "CAR-T / cellular therapy"
    if any(x in txt for x in ["bispecific", "t-cell engager", "t cell engager", "tce", "bite", "teclistamab", "glofitamab"]):
        return "Bispecific / TCE"
    if "vaccine" in txt:
        return "Vaccine"
    if any(x in txt for x in ["monoclonal antibody", "mab", "antibody"]):
        return "Antibody / mAb"
    if "drug" in str(row.get("intervention_types", "")).lower() or ints:
        return "Drug / small molecule"
    return "Unclassified"

def modality_evidence(row) -> str:
    txt = _text_blob(row)
    modality = classify_modality(row)
    if modality == "Multispecific / biparatopic ADC":
        hits = _matched_terms(txt, MULTISPECIFIC_ADC_TERMS) + _matched_terms(txt, CONFIRMED_ADC_TERMS + LIKELY_ADC_TERMS)
    elif modality == "Confirmed ADC":
        hits = _matched_terms(txt, CONFIRMED_ADC_TERMS)
    elif modality == "Likely ADC":
        hits = _matched_terms(txt, LIKELY_ADC_TERMS)
    elif modality == "Therapeutic radioconjugate":
        hits = _matched_terms(txt, THERAPEUTIC_RADIO_TERMS)
    elif modality == "Radiolabeled imaging / diagnostic":
        hits = _matched_terms(txt, DIAGNOSTIC_RADIO_TERMS)
    elif modality == "Possible radioconjugate":
        hits = _matched_terms(txt, AMBIGUOUS_RADIO_TERMS)
    elif modality == "CAR-T / cellular therapy":
        hits = _matched_terms(txt, ["car-t", "cart", "chimeric antigen receptor", "cell therapy", "cellular therapy"])
    elif modality == "Bispecific / TCE":
        hits = _matched_terms(txt, ["bispecific", "t-cell engager", "t cell engager", "tce", "bite", "teclistamab", "glofitamab"])
    else:
        hits = []
    return "; ".join(dict.fromkeys(hits)) if hits else "Protocol text heuristic"


def modality_confidence(row) -> str:
    modality = classify_modality(row)
    if modality in {"Confirmed ADC", "Multispecific / biparatopic ADC", "Therapeutic radioconjugate", "CAR-T / cellular therapy", "Bispecific / TCE"}:
        return "High"
    if modality in {"Likely ADC", "Antibody / mAb", "Radiolabeled imaging / diagnostic"}:
        return "Medium"
    if modality in {"Possible radioconjugate", "Unclassified"}:
        return "Low"
    return "Medium"


def is_adc_modality(modality: str) -> bool:
    return str(modality) in {"Confirmed ADC", "Likely ADC", "Multispecific / biparatopic ADC"}

def infer_payload_family(row) -> str:
    txt = _text_blob(row)
    hits = []
    if any(x in txt for x in ["deruxtecan", "dxd", "datopotamab deruxtecan", "trastuzumab deruxtecan"]): hits.append("Topo-I / DXd")
    if any(x in txt for x in ["sacituzumab", "govitecan", "sn-38"]): hits.append("Topo-I / SN-38")
    if any(x in txt for x in ["vedotin", "mmae", "auristatin"]): hits.append("Auristatin / MMAE")
    if any(x in txt for x in ["mafodotin", "mmaf"]): hits.append("Auristatin / MMAF")
    if any(x in txt for x in ["emtansine", "dm1", "dm4", "maytansinoid"]): hits.append("Maytansinoid")
    if any(x in txt for x in ["tesirine", "pbd", "pyrrolobenzodiazepine"]): hits.append("PBD")
    if "duocarmazine" in txt: hits.append("Duocarmycin")
    if "ozogamicin" in txt or "calicheamicin" in txt: hits.append("Calicheamicin")
    return "; ".join(dict.fromkeys(hits)) if hits else "Not specified"


def extract_combo_class(row) -> str:
    txt = _text_blob(row)
    interventions = [x.strip() for x in str(row.get("interventions", "")).split(";") if x.strip()]
    combo_language = any(x in txt for x in [
        " in combination with ", " combined with ", " plus ", " add-on", " addon",
        " backbone", " with pembrolizumab", " with nivolumab", " with chemotherapy",
        " and pembrolizumab", " and nivolumab", " and carboplatin", " and paclitaxel"
    ])
    combo_context = len(interventions) > 1 or combo_language
    if not combo_context:
        return "Monotherapy / unclear"

    classes = []
    if any(x in txt for x in CHECKPOINT_AGENTS) or "checkpoint" in txt or "pd-1" in txt or "pd-l1" in txt:
        classes.append("Checkpoint combo")
    if any(x in txt for x in CHEMO_TERMS) or "chemotherapy" in txt:
        classes.append("Chemo combo")
    if any(x in txt for x in TARGETED_COMBO_TERMS) or "parp" in txt or "vegf" in txt or "tyrosine kinase" in txt:
        classes.append("Targeted combo")
    if "radiation" in txt or "radiotherapy" in txt:
        classes.append("Radiation combo")
    if not classes and len(interventions) > 1:
        classes.append("Multi-agent protocol")
    return "; ".join(classes) if classes else "Monotherapy / unclear"

def infer_line_of_therapy(row) -> str:
    txt = _text_blob(row)
    if any(x in txt for x in ["first-line", "first line", "1l", "frontline", "previously untreated", "treatment-naive", "treatment naive"]):
        return "1L / frontline"
    if any(x in txt for x in ["second-line", "second line", "2l"]):
        return "2L"
    if any(x in txt for x in ["third-line", "third line", "3l", "fourth-line", "fourth line", "4l", "later-line", "later line"]):
        return "3L+ / later-line"
    if any(x in txt for x in ["refractory", "relapsed", "progressed", "previously treated", "after prior", "resistant"]):
        return "Relapsed / refractory"
    if any(x in txt for x in ["adjuvant", "neoadjuvant", "maintenance"]):
        return "Adjuvant / maintenance"
    return "Not specified"


def add_clinical_intelligence_flags(df: pd.DataFrame) -> pd.DataFrame:
    d = add_activity_flags(df)
    if d.empty:
        return d
    d["modality_class"] = d.apply(classify_modality, axis=1)
    d["modality_confidence"] = d.apply(modality_confidence, axis=1)
    d["modality_evidence"] = d.apply(modality_evidence, axis=1)
    d["payload_family"] = d.apply(infer_payload_family, axis=1)
    d["combo_class"] = d.apply(extract_combo_class, axis=1)
    d["line_of_therapy"] = d.apply(infer_line_of_therapy, axis=1)
    d["is_combo"] = ~d["combo_class"].eq("Monotherapy / unclear")
    d["is_terminated_or_withdrawn"] = d["overall_status"].isin(["TERMINATED", "WITHDRAWN", "SUSPENDED"])
    d["sponsor_is_large_pharma"] = d["lead_sponsor"].fillna("").apply(lambda x: any(term.lower() in str(x).lower() for term in TOP_PHARMA_TERMS))
    return d


def modality_mix(df: pd.DataFrame, active_only: bool = False) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["modality_class", "trials", "active_trials", "enrollment"])
    d = add_clinical_intelligence_flags(df)
    if active_only:
        d = d[d["is_active"]]
    return d.groupby("modality_class", dropna=False).agg(
        trials=("nct_id", "nunique"),
        active_trials=("is_active", "sum"),
        enrollment=("enrollment_count", "sum"),
    ).reset_index().sort_values(["active_trials", "trials"], ascending=False)


def combo_intelligence(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["target", "combo_class", "trials", "active_trials", "enrollment", "top_sponsors"])
    d = add_clinical_intelligence_flags(df)
    d = d[d["is_combo"]]
    if d.empty:
        return pd.DataFrame(columns=["target", "combo_class", "trials", "active_trials", "enrollment", "top_sponsors"])
    out = d.groupby(["target", "combo_class"], dropna=False).agg(
        trials=("nct_id", "nunique"),
        active_trials=("is_active", "sum"),
        enrollment=("enrollment_count", "sum"),
        top_sponsors=("lead_sponsor", lambda s: "; ".join(list(dict.fromkeys([x for x in s.dropna().astype(str) if x]))[:4])),
    ).reset_index()
    return out.sort_values(["target", "active_trials", "trials", "enrollment"], ascending=[True, False, False, False])


def sponsor_conviction(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["target", "lead_sponsor", "conviction_score", "active_trials", "phase2plus_active", "enrollment", "countries", "large_pharma_flag"])
    d = add_clinical_intelligence_flags(df)
    rows = []
    for (target, sponsor), g in d.groupby(["target", "lead_sponsor"], dropna=False):
        active = g[g["is_active"]]
        countries = set()
        for item in g["countries"].dropna():
            for c in str(item).split("; "):
                if c.strip(): countries.add(c.strip())
        phase2plus = int(active[active["phase_bucket"].isin(["PHASE2", "PHASE2_PHASE3", "PHASE3"])] ["nct_id"].nunique())
        large = bool(g["sponsor_is_large_pharma"].any())
        score = int(active["nct_id"].nunique()) * 4 + phase2plus * 5 + min(int(active["enrollment_count"].sum()) / 100, 25) + len(countries) * 0.75 + (6 if large else 0)
        rows.append({
            "target": target,
            "lead_sponsor": sponsor,
            "conviction_score": round(score, 1),
            "active_trials": int(active["nct_id"].nunique()),
            "phase2plus_active": phase2plus,
            "enrollment": int(active["enrollment_count"].sum()),
            "countries": len(countries),
            "large_pharma_flag": "Yes" if large else "No",
        })
    return pd.DataFrame(rows).sort_values(["conviction_score", "active_trials", "enrollment"], ascending=False)


def line_of_therapy_mix(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["target", "line_of_therapy", "trials", "active_trials", "enrollment"])
    d = add_clinical_intelligence_flags(df)
    out = d.groupby(["target", "line_of_therapy"], dropna=False).agg(
        trials=("nct_id", "nunique"),
        active_trials=("is_active", "sum"),
        enrollment=("enrollment_count", "sum"),
    ).reset_index()
    return out.sort_values(["target", "active_trials", "trials"], ascending=[True, False, False])


def risk_signal_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["target", "terminated_withdrawn", "terminated_enrollment", "why_stopped_examples"])
    d = add_clinical_intelligence_flags(df)
    r = d[d["is_terminated_or_withdrawn"]]
    if r.empty:
        return pd.DataFrame(columns=["target", "terminated_withdrawn", "terminated_enrollment", "why_stopped_examples"])
    return r.groupby("target", dropna=False).agg(
        terminated_withdrawn=("nct_id", "nunique"),
        terminated_enrollment=("enrollment_count", "sum"),
        why_stopped_examples=("why_stopped", lambda s: "; ".join(list(dict.fromkeys([x for x in s.dropna().astype(str) if x]))[:3])[:280]),
    ).reset_index().sort_values(["terminated_withdrawn", "terminated_enrollment"], ascending=False)


def asset_level_tracking(df: pd.DataFrame, limit: int = 80) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["target", "asset_or_intervention", "modality", "payload_hint", "trials", "active_trials", "sponsors", "enrollment"])
    d = add_clinical_intelligence_flags(df).copy()
    rows = []
    for _, row in d.iterrows():
        assets = [x.strip() for x in str(row.get("interventions", "")).split(";") if x.strip()]
        if not assets:
            assets = ["Unspecified intervention"]
        for asset in assets[:8]:
            rows.append({
                "target": row.get("target", ""),
                "asset_or_intervention": asset,
                "modality": row.get("modality_class", ""),
                "payload_hint": row.get("payload_family", ""),
                "nct_id": row.get("nct_id", ""),
                "is_active": row.get("is_active", False),
                "lead_sponsor": row.get("lead_sponsor", ""),
                "enrollment_count": row.get("enrollment_count", 0),
            })
    a = pd.DataFrame(rows)
    if a.empty:
        return pd.DataFrame(columns=["target", "asset_or_intervention", "modality", "payload_hint", "trials", "active_trials", "sponsors", "enrollment"])
    out = a.groupby(["target", "asset_or_intervention", "modality", "payload_hint"], dropna=False).agg(
        trials=("nct_id", "nunique"),
        active_trials=("is_active", "sum"),
        sponsors=("lead_sponsor", lambda s: "; ".join(list(dict.fromkeys([x for x in s.dropna().astype(str) if x]))[:4])),
        enrollment=("enrollment_count", "sum"),
    ).reset_index()
    return out.sort_values(["active_trials", "trials", "enrollment"], ascending=False).head(limit)



def modality_by_target(df: pd.DataFrame, active_only: bool = False) -> pd.DataFrame:
    """Target-level modality decomposition with enrollment and active study counts."""
    if df.empty:
        return pd.DataFrame(columns=["target", "modality_class", "trials", "active_trials", "enrollment"])
    d = add_clinical_intelligence_flags(df)
    if active_only:
        d = d[d["is_active"]]
    if d.empty:
        return pd.DataFrame(columns=["target", "modality_class", "trials", "active_trials", "enrollment"])
    out = d.groupby(["target", "modality_class"], dropna=False).agg(
        trials=("nct_id", "nunique"),
        active_trials=("is_active", "sum"),
        enrollment=("enrollment_count", "sum"),
    ).reset_index()
    return out.sort_values(["target", "active_trials", "trials", "enrollment"], ascending=[True, False, False, False])


def target_exploitation_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize how each target is being exploited clinically.

    This stays strictly within ClinicalTrials.gov data: modality, combo use,
    sponsor counts, enrollment, phase maturity, and stopped-study rates.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "target", "trials", "active_trials", "adc_trials", "adc_share", "combo_share",
            "modality_diversity", "phase2plus_active", "sponsor_count", "active_enrollment",
            "stopped_trials", "dominant_modality", "profile_readout"
        ])
    d = add_clinical_intelligence_flags(df)
    rows = []
    for target, g in d.groupby("target", dropna=False):
        trials = int(g["nct_id"].nunique())
        active = g[g["is_active"]]
        active_trials = int(active["nct_id"].nunique())
        adc_trials = int(g[g["modality_class"].apply(is_adc_modality)]["nct_id"].nunique())
        combo_trials = int(g[g["is_combo"]]["nct_id"].nunique())
        modality_diversity = int(g["modality_class"].nunique())
        phase2plus_active = int(active[active["phase_bucket"].isin(["PHASE2", "PHASE2_PHASE3", "PHASE3"])] ["nct_id"].nunique())
        sponsor_count = int(g["lead_sponsor"].nunique())
        stopped = int(g[g["is_terminated_or_withdrawn"]]["nct_id"].nunique())
        dom = g.groupby("modality_class")["nct_id"].nunique().sort_values(ascending=False)
        dominant = dom.index[0] if not dom.empty else "Unclassified"
        adc_share = adc_trials / trials if trials else 0
        combo_share = combo_trials / trials if trials else 0
        if adc_share >= .45:
            readout = "ADC-led target activity"
        elif modality_diversity >= 4:
            readout = "Modality-diversified target"
        elif phase2plus_active >= 2:
            readout = "Clinically maturing target"
        elif stopped >= max(2, trials * .25):
            readout = "High stopped-study signal"
        elif active_trials == 0:
            readout = "Dormant / historical activity"
        else:
            readout = "Early or mixed clinical activity"
        rows.append({
            "target": target,
            "trials": trials,
            "active_trials": active_trials,
            "adc_trials": adc_trials,
            "adc_share": round(adc_share, 3),
            "combo_share": round(combo_share, 3),
            "modality_diversity": modality_diversity,
            "phase2plus_active": phase2plus_active,
            "sponsor_count": sponsor_count,
            "active_enrollment": int(active["enrollment_count"].sum()),
            "stopped_trials": stopped,
            "dominant_modality": dominant,
            "profile_readout": readout,
        })
    return pd.DataFrame(rows).sort_values(["active_trials", "adc_trials", "active_enrollment"], ascending=False)


def crowding_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Clinical crowding score based only on trial footprint and redundancy signals."""
    if df.empty:
        return pd.DataFrame(columns=["target", "crowding_score", "crowding_band", "active_trials", "sponsor_count", "phase2plus_active", "modality_diversity", "active_enrollment"])
    prof = target_exploitation_profile(df)
    if prof.empty:
        return prof
    out = prof.copy()
    out["crowding_score"] = (
        out["active_trials"] * 3.0
        + out["sponsor_count"] * 2.0
        + out["phase2plus_active"] * 4.0
        + out["modality_diversity"] * 1.5
        + (out["active_enrollment"] / 150).clip(upper=30)
    ).round(1)
    def band(x):
        if x >= 55: return "Overcrowded / consensus-heavy"
        if x >= 32: return "Competitive / active"
        if x >= 16: return "Emerging attention"
        return "Undercapitalized / sparse"
    out["crowding_band"] = out["crowding_score"].apply(band)
    return out[["target", "crowding_score", "crowding_band", "active_trials", "sponsor_count", "phase2plus_active", "modality_diversity", "active_enrollment", "dominant_modality", "profile_readout"]].sort_values("crowding_score", ascending=False)


def target_evolution_timeline(df: pd.DataFrame, current_year: int | None = None) -> pd.DataFrame:
    """Year-by-year target evolution: new studies, ADC studies, sponsors, combos, stopped studies."""
    if df.empty:
        return pd.DataFrame(columns=["target", "start_year", "new_trials", "adc_trials", "combo_trials", "new_sponsors", "phase2plus_trials", "stopped_trials", "enrollment"])
    current_year = current_year or datetime.now().year
    d = add_clinical_intelligence_flags(df).dropna(subset=["start_year"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["target", "start_year", "new_trials", "adc_trials", "combo_trials", "new_sponsors", "phase2plus_trials", "stopped_trials", "enrollment"])
    d["start_year"] = d["start_year"].astype(int)
    d = d[d["start_year"] <= current_year]
    out = d.groupby(["target", "start_year"], dropna=False).agg(
        new_trials=("nct_id", "nunique"),
        adc_trials=("modality_class", lambda s: int(sum(is_adc_modality(x) for x in s))),
        combo_trials=("is_combo", "sum"),
        new_sponsors=("lead_sponsor", "nunique"),
        phase2plus_trials=("phase_bucket", lambda s: int(sum(x in ["PHASE2", "PHASE2_PHASE3", "PHASE3"] for x in s))),
        stopped_trials=("is_terminated_or_withdrawn", "sum"),
        enrollment=("enrollment_count", "sum"),
    ).reset_index().sort_values(["target", "start_year"])
    return out


def combo_agent_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Extract common combo agents/classes from trial text for a clean ranked table."""
    if df.empty:
        return pd.DataFrame(columns=["target", "agent_or_class", "trials", "active_trials", "enrollment"])
    agent_terms = {
        "Pembrolizumab / Keytruda": ["pembrolizumab", "keytruda"],
        "Nivolumab / Opdivo": ["nivolumab", "opdivo"],
        "Atezolizumab": ["atezolizumab"],
        "Durvalumab": ["durvalumab"],
        "Ipilimumab": ["ipilimumab"],
        "Platinum chemotherapy": ["carboplatin", "cisplatin", "platinum"],
        "Taxane chemotherapy": ["paclitaxel", "docetaxel", "taxane"],
        "PARP inhibitor": ["olaparib", "niraparib", "rucaparib", "parp"],
        "VEGF / anti-angiogenic": ["bevacizumab", "vegf", "lenvatinib"],
        "HER2-directed combo": ["trastuzumab", "pertuzumab", "tucatinib"],
        "Radiation": ["radiation", "radiotherapy"],
    }
    d = add_clinical_intelligence_flags(df).copy()
    rows = []
    for _, row in d.iterrows():
        txt = _text_blob(row)
        for label, terms in agent_terms.items():
            if any(t in txt for t in terms):
                rows.append({
                    "target": row.get("target", ""),
                    "agent_or_class": label,
                    "nct_id": row.get("nct_id", ""),
                    "is_active": row.get("is_active", False),
                    "enrollment_count": row.get("enrollment_count", 0),
                })
    if not rows:
        return pd.DataFrame(columns=["target", "agent_or_class", "trials", "active_trials", "enrollment"])
    a = pd.DataFrame(rows)
    out = a.groupby(["target", "agent_or_class"], dropna=False).agg(
        trials=("nct_id", "nunique"),
        active_trials=("is_active", "sum"),
        enrollment=("enrollment_count", "sum"),
    ).reset_index()
    return out.sort_values(["target", "active_trials", "trials", "enrollment"], ascending=[True, False, False, False])
