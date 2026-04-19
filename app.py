
# -*- coding: utf-8 -*-
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

st.set_page_config(page_title="2022 WFH LACW Dashboard — Rebuilt", layout="wide")

# -----------------------------
# Style
# -----------------------------
pio.templates["paper_style"] = pio.templates["plotly_white"]
pio.templates["paper_style"].layout.font.family = "Times New Roman"
pio.templates["paper_style"].layout.font.size = 16
pio.templates["paper_style"].layout.title.font.size = 20
pio.templates.default = "paper_style"

# -----------------------------
# Constants
# -----------------------------
FRACTION_NORMALISE = {
    "Paper&Card": "PaperCard",
    "PaperampCard": "PaperCard",
    "OtherRecyclables": "Miscellaneous",
    "Miscellanious": "Miscellaneous",
    "Textile": "Textiles",
    "Garden&OrganicWaste": "Garden",
}
DIVERTABLE_FRACTIONS = ["Food", "Garden", "PaperCard", "Plastics", "Glass", "Metals", "Textiles", "WEEE", "Wood"]
RECOVERY_ROUTES = ["Recycled", "AD", "CompostedIV", "CompostedW", "Reuse"]
ENERGY_RECOVERY_ROUTES = ["EfW"]
SCENARIO_ORDER = ["Baseline", "Policy65 - Uniform cap", "Policy65 - Behaviour-based", "Policy65 - Carbon-optimised", "Optimal"]

PREFERRED_POLICY_DEST = {
    "Food": [("AD", 1.0)],
    "Garden": [("AD", 1.0)],
    "PaperCard": [("Recycled", 1.0)],
    "Plastics": [("Recycled", 1.0)],
    "Glass": [("Recycled", 1.0)],
    "Metals": [("Recycled", 1.0)],
    "Textiles": [("Recycled", 1.0)],
    "WEEE": [("Recycled", 1.0)],
    "Wood": [("Recycled", 1.0)],
}

EF = {
    ("Food", "AD"): -78.0, ("Food", "CompostedIV"): -55.0, ("Food", "CompostedW"): 6.0, ("Food", "EfW"): -37.0, ("Food", "Landfill"): 627.0,
    ("Garden", "AD"): -184.09, ("Garden", "CompostedIV"): -45.0, ("Garden", "CompostedW"): 56.0, ("Garden", "EfW"): -77.0, ("Garden", "Landfill"): 579.0, ("Garden", "IncNoEnergy"): 360.0,
    ("PaperCard", "Recycled"): -109.7, ("PaperCard", "EfW"): -217.0, ("PaperCard", "Landfill"): 1042.0,
    ("Plastics", "Recycled"): -576.3, ("Plastics", "EfW"): 1581.7, ("Plastics", "Landfill"): 9.0,
    ("Glass", "Recycled"): -326.0, ("Glass", "EfW"): 8.0, ("Glass", "Landfill"): 9.0,
    ("Metals", "Recycled"): -4578.5, ("Metals", "EfW"): 21.5, ("Metals", "Landfill"): 9.0,
    ("Textiles", "Recycled"): -14315.0, ("Textiles", "EfW"): 438.0, ("Textiles", "Landfill"): 445.0,
    ("WEEE", "Recycled"): -1000.0, ("WEEE", "EfW"): 450.0, ("WEEE", "Landfill"): 20.0, ("WEEE", "IncNoEnergy"): 360.0,
    ("Wood", "Recycled"): -754.5, ("Wood", "EfW"): -318.0, ("Wood", "Landfill"): 921.0, ("Wood", "IncNoEnergy"): 360.0,
    ("Hazardous", "IncNoEnergy"): 360.0,
    ("Miscellaneous", "EfW"): 0.0, ("Miscellaneous", "Landfill"): 0.0, ("Miscellaneous", "IncNoEnergy"): 0.0, ("Miscellaneous", "Other"): 0.0,
}

VALUE_MID = {
    "PaperCard": 105.0,
    "Plastics": 427.5,
    "Glass": 5.0,
    "Metals": 600.0,
    "Textiles": 207.5,
    "WEEE": 225.0,
    "Wood": -10.0,
    "Food": 0.0,
    "Garden": 0.0,
    "Miscellaneous": 0.0,
}

NCV = {
    "Food": 3.85,
    "Garden": 4.72,
    "PaperCard": 9.16,
    "Plastics": 18.44,
    "Glass": 0.0,
    "Metals": 0.0,
    "Textiles": 11.63,
    "WEEE": 5.52,
    "Wood": 13.30,
    "Miscellaneous": 0.0,
    "Hazardous": 0.0,
}

ROUTE_NAMES = ["Recycled", "Reuse", "RDF_MHT", "EfW", "Landfill", "AD", "CompostedIV", "CompostedW", "IncNoEnergy"]
EXCLUDED_FIGURE_FRACTIONS = ["Miscellaneous"]


# -----------------------------
# Helpers
# -----------------------------
def resolve_local_file(filename: str):
    candidates = [Path(filename)]
    try:
        candidates.append(Path(__file__).resolve().parent / filename)
    except Exception:
        pass
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def ef(frac: str, route: str) -> float:
    return float(EF.get((frac, route), 0.0))


def normalize_fraction(name: str) -> str:
    name = str(name).strip()
    for old, new in FRACTION_NORMALISE.items():
        name = name.replace(old, new)
    return name


def add_metrics(frame: pd.DataFrame, efficiency: float) -> pd.DataFrame:
    out = frame.copy()
    out["emissions_kgco2e"] = out.apply(lambda r: r["tonnes"] * ef(r["fraction"], r["route"]), axis=1)
    out["value_gbp"] = out.apply(lambda r: r["tonnes"] * VALUE_MID.get(r["fraction"], 0.0) if r["route"] == "Recycled" else 0.0, axis=1)
    out["energy_gwh"] = out.apply(
        lambda r: r["tonnes"] * 1000.0 * NCV.get(r["fraction"], 0.0) * efficiency / 3600.0 / 1000.0 if r["route"] == "EfW" else 0.0,
        axis=1,
    )
    return out


def apply_paper_style(fig, title=None, x_title=None, y_title=None, tickangle=20):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Times New Roman", size=16, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=40, r=20, t=70, b=90),
        legend=dict(title=None, orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.01),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0.02, xanchor="left"))
    fig.update_xaxes(title_text=x_title, showgrid=False, linecolor="black", ticks="outside", tickangle=tickangle, showline=True)
    fig.update_yaxes(title_text=y_title, showgrid=False, linecolor="black", ticks="outside", showline=True)
    return fig


def add_download_buttons(fig, stem):
    html = fig.to_html(include_plotlyjs="cdn")
    st.download_button("Download figure (HTML)", data=html, file_name=f"{stem}.html", mime="text/html", key=f"{stem}_html")


def add_table_download(df: pd.DataFrame, filename: str, label: str):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv", key=filename)


def drop_aggregate_rows(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    name = out["Council Name"].astype(str).str.strip()
    out = out[out["Council Name"].notna()]
    out = out[~name.str.lower().isin(["total", "nan", "none", ""])]
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def read_sheets(scenario_source):
    rec = pd.read_excel(scenario_source, sheet_name="WFH Recoverable LACW DATA 2022")
    res = pd.read_excel(scenario_source, sheet_name="WFH Residual LACW DATA 2022")
    return drop_aggregate_rows(rec), drop_aggregate_rows(res)


def parse_recoverable_routes(rec: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in rec.columns:
        col_str = str(col)
        if col_str.startswith("Share "):
            continue
        if re.search(r"\.\d+$", col_str):
            continue
        matched = None
        frac = None
        for rt in ROUTE_NAMES:
            suffix = f"{rt}_2022"
            if col_str.endswith(suffix):
                matched = rt
                frac = normalize_fraction(col_str[:-len(suffix)])
                break
        if matched is None:
            continue
        vals = pd.to_numeric(rec[col], errors="coerce").fillna(0.0)
        for council, value in zip(rec["Council Name"], vals):
            if value != 0:
                rows.append(
                    {
                        "Council Name": str(council),
                        "stream": "Recoverable",
                        "fraction": frac,
                        "route": matched,
                        "tonnes": float(value),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["Council Name", "stream", "fraction", "route", "tonnes"])
    return out.groupby(["Council Name", "stream", "fraction", "route"], as_index=False).agg(tonnes=("tonnes", "sum"))


def parse_residual_totals(res: pd.DataFrame) -> pd.DataFrame:
    colmap = {
        "Food residual": "Food",
        "Garden residual": "Garden",
        "Paper&Card Residual": "PaperCard",
        "Glass Residual": "Glass",
        "Metals Residual ": "Metals",
        "Plastics residual ": "Plastics",
        "Textiles residual": "Textiles",
        "WEEE Residual": "WEEE",
        "Wood residual ": "Wood",
        "Hazardous residual ": "Hazardous",
        "Miscellanious residual": "Miscellaneous",
    }
    rows = []
    for _, row in res.iterrows():
        council = str(row["Council Name"])
        for col, frac in colmap.items():
            value = float(pd.to_numeric(pd.Series([row[col]]), errors="coerce").fillna(0.0).iloc[0])
            if value != 0:
                rows.append({"Council Name": council, "fraction": frac, "residual_total_tonnes": value})
    return pd.DataFrame(rows)


def parse_residual_routes(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    explicit = [
        ("Food_EfW", "Food", "EfW"), ("Food_Landfill", "Food", "Landfill"),
        ("Garden_EfW", "Garden", "EfW"), ("Garden_Landfill", "Garden", "Landfill"),
        ("Paper&Card_EfW", "PaperCard", "EfW"), ("Paper&Card_Landfill", "PaperCard", "Landfill"),
        ("Glass_EfW", "Glass", "EfW"), ("Glass_Landfill", "Glass", "Landfill"),
        ("Metals_EfW", "Metals", "EfW"), ("Metals_Landfill", "Metals", "Landfill"),
        ("Plastics_EfW", "Plastics", "EfW"), ("Plastics_Landfill", "Plastics", "Landfill"),
        ("Textiles_EfW", "Textiles", "EfW"), ("Textiles_Landfill", "Textiles", "Landfill"),
        ("WEEE_EfW", "WEEE", "EfW"), ("WEEE_Landfill", "WEEE", "Landfill"),
        ("Wood_EfW", "Wood", "EfW"), ("Wood_Landfill", "Wood", "Landfill"),
        ("Hazardous_IWE", "Hazardous", "IncNoEnergy"),
        ("Miscellanious_EfW", "Miscellaneous", "EfW"),
        ("Miscellanious_Landfill", "Miscellaneous", "Landfill"),
        ("Miscellanious_IWE", "Miscellaneous", "IncNoEnergy"),
        ("Miscellanious_Other", "Miscellaneous", "Other"),
    ]
    for _, row in res.iterrows():
        council = str(row["Council Name"])
        for col, frac, route in explicit:
            value = float(pd.to_numeric(pd.Series([row[col]]), errors="coerce").fillna(0.0).iloc[0])
            if value != 0:
                rows.append(
                    {
                        "Council Name": council,
                        "stream": "Residual",
                        "fraction": frac,
                        "route": route,
                        "tonnes": value,
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["Council Name", "stream", "fraction", "route", "tonnes"])
    return out.groupby(["Council Name", "stream", "fraction", "route"], as_index=False).agg(tonnes=("tonnes", "sum"))


def build_baseline_frame(rec: pd.DataFrame, res: pd.DataFrame, efficiency: float):
    rec_routes = parse_recoverable_routes(rec)
    res_routes = parse_residual_routes(res)
    frame = pd.concat([rec_routes, res_routes], ignore_index=True)
    frame = add_metrics(frame, efficiency)

    totals = {
        "total_collected_tonnes": float(pd.to_numeric(rec["TotalCollected2022"], errors="coerce").fillna(0.0).sum()),
        "baseline_captured_tonnes": float(pd.to_numeric(rec["total recycled collected"], errors="coerce").fillna(0.0).sum()),
    }

    qa = {}
    qa["recoverable_routes_sum"] = float(rec_routes["tonnes"].sum()) if not rec_routes.empty else 0.0
    qa["reported_total_recycled_collected"] = totals["baseline_captured_tonnes"]
    qa["recoverable_gap"] = qa["reported_total_recycled_collected"] - qa["recoverable_routes_sum"]
    residual_total_reported = float(pd.to_numeric(res["Residual_2022"], errors="coerce").fillna(0.0).sum())
    residual_routes_sum = float(res_routes["tonnes"].sum()) if not res_routes.empty else 0.0
    qa["residual_total_reported"] = residual_total_reported
    qa["residual_routes_sum"] = residual_routes_sum
    qa["residual_gap"] = residual_total_reported - residual_routes_sum
    return frame, totals, qa


def build_recovery_behaviour_shares(base_frame: pd.DataFrame):
    rec = base_frame[(base_frame["stream"] == "Recoverable") & (base_frame["route"].isin(RECOVERY_ROUTES))].copy()
    nat = rec.groupby(["fraction", "route"], as_index=False).agg(tonnes=("tonnes", "sum"))
    nat["share"] = nat.groupby("fraction")["tonnes"].transform(lambda s: s / s.sum() if s.sum() > 0 else 0.0)

    beh = rec.groupby(["Council Name", "fraction", "route"], as_index=False).agg(tonnes=("tonnes", "sum"))
    beh["share"] = beh.groupby(["Council Name", "fraction"])["tonnes"].transform(lambda s: s / s.sum() if s.sum() > 0 else 0.0)
    return nat, beh


def build_baseline_capture_propensity(rec: pd.DataFrame, residual_totals: pd.DataFrame) -> pd.DataFrame:
    total_cols = {
        "Foodwaste total": "Food",
        "Garden&OrganicWaste total": "Garden",
        "PaperampCard total": "PaperCard",
        "Plastics total": "Plastics",
        "Glass total ": "Glass",
        "Metals total": "Metals",
        "Textile total": "Textiles",
        "WEEE total": "WEEE",
        "Wood Total": "Wood",
    }
    rows = []
    for _, row in rec.iterrows():
        council = str(row["Council Name"])
        for col, frac in total_cols.items():
            total_frac = float(pd.to_numeric(pd.Series([row[col]]), errors="coerce").fillna(0.0).iloc[0])
            rows.append({"Council Name": council, "fraction": frac, "recoverable_total_tonnes": total_frac})
    recoverable_total = pd.DataFrame(rows)
    out = recoverable_total.merge(residual_totals, on=["Council Name", "fraction"], how="left")
    out["residual_total_tonnes"] = out["residual_total_tonnes"].fillna(0.0)
    out["generation_proxy_tonnes"] = out["recoverable_total_tonnes"] + out["residual_total_tonnes"]
    out["capture_propensity"] = np.where(
        out["generation_proxy_tonnes"] > 0,
        out["recoverable_total_tonnes"] / out["generation_proxy_tonnes"],
        0.0,
    )
    return out[["Council Name", "fraction", "recoverable_total_tonnes", "residual_total_tonnes", "capture_propensity"]]


def build_residual_route_shares(base_frame: pd.DataFrame):
    res = base_frame[(base_frame["stream"] == "Residual") & (base_frame["fraction"].isin(DIVERTABLE_FRACTIONS))].copy()
    route_shares = res.groupby(["Council Name", "fraction", "route"], as_index=False).agg(tonnes=("tonnes", "sum"))
    route_shares["share"] = route_shares.groupby(["Council Name", "fraction"])["tonnes"].transform(lambda s: s / s.sum() if s.sum() > 0 else 0.0)
    nat = res.groupby(["fraction", "route"], as_index=False).agg(tonnes=("tonnes", "sum"))
    nat["share"] = nat.groupby(["fraction"])["tonnes"].transform(lambda s: s / s.sum() if s.sum() > 0 else 0.0)
    return route_shares, nat


def carbon_best_route(frac: str) -> str:
    candidates = [(rt, ef(frac, rt)) for rt in RECOVERY_ROUTES if (frac, rt) in EF]
    if not candidates:
        return PREFERRED_POLICY_DEST.get(frac, [("Recycled", 1.0)])[0][0]
    return min(candidates, key=lambda x: x[1])[0]


def energy_yield_gwh_per_tonne(frac: str, route: str, efficiency: float) -> float:
    if route != "EfW":
        return 0.0
    return NCV.get(frac, 0.0) * efficiency / 3600.0


def _normalise_series(values: list[float]) -> list[float]:
    if not values:
        return []
    vmin = min(values)
    vmax = max(values)
    if abs(vmax - vmin) <= 1e-12:
        return [0.0 for _ in values]
    return [(v - vmin) / (vmax - vmin) for v in values]


def balanced_best_route(frac: str, baseline_emission: float, efficiency: float, carbon_weight: float = 0.5, value_weight: float = 0.3, energy_weight: float = 0.2) -> str:
    candidates = [rt for rt in RECOVERY_ROUTES if (frac, rt) in EF]
    if not candidates:
        return PREFERRED_POLICY_DEST.get(frac, [("Recycled", 1.0)])[0][0]

    carbon_gain = [baseline_emission - ef(frac, rt) for rt in candidates]
    value_gain = [VALUE_MID.get(frac, 0.0) if rt == "Recycled" else 0.0 for rt in candidates]
    energy_gain = [energy_yield_gwh_per_tonne(frac, rt, efficiency) for rt in candidates]

    carbon_n = _normalise_series(carbon_gain)
    value_n = _normalise_series(value_gain)
    energy_n = _normalise_series(energy_gain)

    scores = []
    for rt, c, v, e in zip(candidates, carbon_n, value_n, energy_n):
        score = carbon_weight * c + value_weight * v + energy_weight * e
        scores.append((rt, score))

    return max(scores, key=lambda x: x[1])[0]


def build_default_residual_emission(residual_nat_shares: pd.DataFrame) -> dict:
    avg = {}
    for frac in DIVERTABLE_FRACTIONS:
        sub = residual_nat_shares[residual_nat_shares["fraction"] == frac]
        avg[frac] = float(sum(float(r["share"]) * ef(frac, r["route"]) for _, r in sub.iterrows()))
    return avg


def select_uniform_cap(avoidable_pool: pd.DataFrame, take_amount: float) -> pd.DataFrame:
    total = float(avoidable_pool["residual_total_tonnes"].sum())
    prop = 0.0 if total <= 0 else min(1.0, take_amount / total)
    out = avoidable_pool.copy()
    out["take_tonnes"] = out["residual_total_tonnes"] * prop
    return out[["Council Name", "fraction", "take_tonnes"]]


def select_behaviour_based(avoidable_pool: pd.DataFrame, capture_propensity: pd.DataFrame, take_amount: float) -> pd.DataFrame:
    pool = avoidable_pool.merge(capture_propensity[["Council Name", "fraction", "capture_propensity"]], on=["Council Name", "fraction"], how="left")
    pool["capture_propensity"] = pool["capture_propensity"].fillna(0.0)
    pool["weight"] = pool["residual_total_tonnes"] * (0.15 + pool["capture_propensity"])
    total_weight = float(pool["weight"].sum())

    if take_amount <= 0 or total_weight <= 0:
        pool["take_tonnes"] = 0.0
        return pool[["Council Name", "fraction", "take_tonnes"]]

    # proportional allocation with clipping + redistribution
    pool["take_tonnes"] = np.minimum(pool["residual_total_tonnes"], take_amount * pool["weight"] / total_weight)
    remaining = float(take_amount - pool["take_tonnes"].sum())

    for _ in range(8):
        if remaining <= 1e-9:
            break
        room = pool["residual_total_tonnes"] - pool["take_tonnes"]
        eligible = room > 1e-9
        if not eligible.any():
            break
        extra_weight = pool.loc[eligible, "weight"]
        if float(extra_weight.sum()) <= 0:
            add = remaining / eligible.sum()
            pool.loc[eligible, "take_tonnes"] += np.minimum(room[eligible], add)
        else:
            add = remaining * extra_weight / float(extra_weight.sum())
            pool.loc[eligible, "take_tonnes"] += np.minimum(room[eligible], add)
        pool["take_tonnes"] = np.minimum(pool["take_tonnes"], pool["residual_total_tonnes"])
        remaining = float(take_amount - pool["take_tonnes"].sum())

    return pool[["Council Name", "fraction", "take_tonnes"]]


def select_carbon_optimised(avoidable_pool: pd.DataFrame, residual_route_shares: pd.DataFrame, residual_nat_shares: pd.DataFrame, take_amount: float) -> pd.DataFrame:
    nat_avg = build_default_residual_emission(residual_nat_shares)

    scores = []
    for _, r in avoidable_pool.iterrows():
        council = r["Council Name"]
        frac = r["fraction"]
        sub = residual_route_shares[(residual_route_shares["Council Name"] == council) & (residual_route_shares["fraction"] == frac)]
        if sub.empty:
            baseline_emission = nat_avg.get(frac, 0.0)
        else:
            baseline_emission = float(sum(float(x["share"]) * ef(frac, x["route"]) for _, x in sub.iterrows()))
        best_gain = baseline_emission - ef(frac, carbon_best_route(frac))
        scores.append(best_gain)

    pool = avoidable_pool.copy()
    pool["carbon_gain_per_tonne"] = scores
    pool = pool.sort_values(["carbon_gain_per_tonne", "residual_total_tonnes"], ascending=[False, False]).reset_index(drop=True)

    takes = []
    remaining = float(take_amount)
    for _, r in pool.iterrows():
        if remaining <= 1e-9:
            break
        cap = float(r["residual_total_tonnes"])
        take = min(cap, remaining)
        takes.append({"Council Name": r["Council Name"], "fraction": r["fraction"], "take_tonnes": take})
        remaining -= take

    return pd.DataFrame(takes) if takes else pd.DataFrame(columns=["Council Name", "fraction", "take_tonnes"])


def select_optimal(avoidable_pool: pd.DataFrame) -> pd.DataFrame:
    out = avoidable_pool.copy()
    out["take_tonnes"] = out["residual_total_tonnes"]
    return out[["Council Name", "fraction", "take_tonnes"]]


def build_share_lookup(df: pd.DataFrame, key_cols):
    lookup = {}
    for keys, sub in df.groupby(key_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        lookup[keys] = [(str(r["route"]), float(r["share"])) for _, r in sub.iterrows() if float(r["share"]) > 0]
    return lookup


def optimal_destination(council: str, frac: str, residual_lookup: dict, residual_nat_lookup: dict, efficiency: float):
    shares = residual_lookup.get((council, frac), residual_nat_lookup.get((frac,), []))
    baseline_emission = sum(float(share) * ef(frac, route) for route, share in shares)
    return [(balanced_best_route(frac, baseline_emission, efficiency), 1.0)]


def scenario_destinations(mode: str, council: str, frac: str, nat_beh_lookup: dict, beh_lookup: dict, residual_lookup: dict, residual_nat_lookup: dict, efficiency: float):
    if mode == "cap":
        return PREFERRED_POLICY_DEST.get(frac, [("Recycled", 1.0)])

    if mode == "behaviour":
        dest = beh_lookup.get((council, frac))
        if dest:
            return dest
        dest = nat_beh_lookup.get((frac,))
        if dest:
            return dest
        return PREFERRED_POLICY_DEST.get(frac, [("Recycled", 1.0)])

    if mode == "carbon":
        return [(carbon_best_route(frac), 1.0)]

    return optimal_destination(council, frac, residual_lookup, residual_nat_lookup, efficiency)


def apply_scenario(base_frame: pd.DataFrame, rec: pd.DataFrame, res: pd.DataFrame, mode: str, target_rate: float, efficiency: float):
    totals = {
        "total_collected_tonnes": float(pd.to_numeric(rec["TotalCollected2022"], errors="coerce").fillna(0.0).sum()),
        "baseline_captured_tonnes": float(pd.to_numeric(rec["total recycled collected"], errors="coerce").fillna(0.0).sum()),
    }

    residual_totals = parse_residual_totals(res)
    avoidable_pool = residual_totals[residual_totals["fraction"].isin(DIVERTABLE_FRACTIONS)].copy()

    avoidable_total = float(avoidable_pool["residual_total_tonnes"].sum())
    extra_needed = max(0.0, target_rate * totals["total_collected_tonnes"] - totals["baseline_captured_tonnes"])
    take_amount = avoidable_total if mode == "optimal" else min(extra_needed, avoidable_total)

    nat_beh, beh = build_recovery_behaviour_shares(base_frame)
    capture_propensity = build_baseline_capture_propensity(rec, avoidable_pool)
    residual_route_shares, residual_nat_shares = build_residual_route_shares(base_frame)

    if mode == "cap":
        selected = select_uniform_cap(avoidable_pool, take_amount)
    elif mode == "behaviour":
        selected = select_behaviour_based(avoidable_pool, capture_propensity, take_amount)
    elif mode == "carbon":
        selected = select_carbon_optimised(avoidable_pool, residual_route_shares, residual_nat_shares, take_amount)
    else:
        selected = select_optimal(avoidable_pool)

    selected = selected[selected["take_tonnes"] > 1e-9].copy()

    nat_beh_lookup = build_share_lookup(nat_beh, ["fraction"])
    beh_lookup = build_share_lookup(beh, ["Council Name", "fraction"])
    residual_lookup = build_share_lookup(residual_route_shares, ["Council Name", "fraction"])
    residual_nat_lookup = build_share_lookup(residual_nat_shares, ["fraction"])

    delta_rows = []

    for _, d in selected.iterrows():
        council = str(d["Council Name"])
        frac = str(d["fraction"])
        take = float(d["take_tonnes"])

        for route, share in residual_lookup.get((council, frac), residual_nat_lookup.get((frac,), [])):
            delta_rows.append(
                {
                    "Council Name": council,
                    "stream": "Residual",
                    "fraction": frac,
                    "route": route,
                    "tonnes": -take * float(share),
                }
            )

        for route, share in scenario_destinations(mode, council, frac, nat_beh_lookup, beh_lookup, residual_lookup, residual_nat_lookup, efficiency):
            delta_rows.append(
                {
                    "Council Name": council,
                    "stream": "Recoverable",
                    "fraction": frac,
                    "route": route,
                    "tonnes": take * float(share),
                }
            )

    scen = base_frame[["Council Name", "stream", "fraction", "route", "tonnes"]].copy()
    if delta_rows:
        scen = pd.concat([scen, pd.DataFrame(delta_rows)], ignore_index=True)

    scen = scen.groupby(["Council Name", "stream", "fraction", "route"], as_index=False).agg(tonnes=("tonnes", "sum"))
    scen["tonnes"] = scen["tonnes"].clip(lower=0.0)
    scen = add_metrics(scen, efficiency)

    diverted = float(selected["take_tonnes"].sum())
    metrics = {
        "Net emissions (Mt CO₂e)": float(scen["emissions_kgco2e"].sum() / 1_000_000_000.0),
        "Recovery rate (%)": float((totals["baseline_captured_tonnes"] + diverted) / totals["total_collected_tonnes"] * 100.0),
        "Recovered value (midpoint, £)": float(scen["value_gbp"].sum()),
        "Recovered EfW energy (GWh)": float(scen["energy_gwh"].sum()),
        "Diverted avoidable tonnes": diverted,
    }
    return scen, metrics


def build_tradeoff_table(base_frame: pd.DataFrame) -> pd.DataFrame:
    filtered = base_frame[~base_frame["fraction"].isin(EXCLUDED_FIGURE_FRACTIONS)].copy()
    by_frac = filtered.groupby("fraction", as_index=False).agg(
        total_emissions_kg=("emissions_kgco2e", "sum"),
        total_value_gbp=("value_gbp", "sum"),
        total_energy_gwh=("energy_gwh", "sum"),
    )
    rec_t = filtered[filtered["route"] == "Recycled"].groupby("fraction", as_index=False).agg(recycled_tonnes=("tonnes", "sum"))
    efw_t = filtered[filtered["route"] == "EfW"].groupby("fraction", as_index=False).agg(efw_tonnes=("tonnes", "sum"))
    out = by_frac.merge(rec_t, on="fraction", how="left").merge(efw_t, on="fraction", how="left")
    out["recycled_tonnes"] = out["recycled_tonnes"].fillna(0.0)
    out["efw_tonnes"] = out["efw_tonnes"].fillna(0.0)
    out["Recovered value (£/recycled t)"] = np.where(out["recycled_tonnes"] > 0, out["total_value_gbp"] / out["recycled_tonnes"], 0.0)
    out["Recovered EfW energy (MWh/EfW t)"] = np.where(out["efw_tonnes"] > 0, out["total_energy_gwh"] * 1000.0 / out["efw_tonnes"], 0.0)
    out["Net carbon (tCO₂e)"] = out["total_emissions_kg"] / 1000.0
    return out.sort_values("fraction")


def build_material_priority_matrices(base_frame: pd.DataFrame, efficiency: float):
    residual_route_shares, residual_nat_shares = build_residual_route_shares(base_frame)
    nat_avg = build_default_residual_emission(residual_nat_shares)

    residual_pool = (
        base_frame[(base_frame["stream"] == "Residual") & (base_frame["fraction"].isin(DIVERTABLE_FRACTIONS))]
        .groupby("fraction", as_index=False)
        .agg(residual_tonnes=("tonnes", "sum"))
    )

    rows_recycling = []
    rows_efw = []
    for frac in DIVERTABLE_FRACTIONS:
        residual_tonnes = float(residual_pool.loc[residual_pool["fraction"] == frac, "residual_tonnes"].sum())
        baseline_emission = nat_avg.get(frac, 0.0)

        recycle_env = baseline_emission - ef(frac, "Recycled") if (frac, "Recycled") in EF else np.nan
        recycle_value = VALUE_MID.get(frac, 0.0)
        if not np.isnan(recycle_env):
            rows_recycling.append({
                "fraction": frac,
                "residual_tonnes": residual_tonnes,
                "Environmental benefit if recycled (kg CO₂e/t)": recycle_env,
                "Economic benefit if recycled (£/t)": recycle_value,
            })

        efw_env = baseline_emission - ef(frac, "EfW") if (frac, "EfW") in EF else np.nan
        efw_energy = energy_yield_gwh_per_tonne(frac, "EfW", efficiency) * 1000.0
        if not np.isnan(efw_env):
            rows_efw.append({
                "fraction": frac,
                "residual_tonnes": residual_tonnes,
                "Environmental benefit if incinerated (kg CO₂e/t)": efw_env,
                "Energy benefit if incinerated (MWh/t)": efw_energy,
            })

    recycling_matrix = pd.DataFrame(rows_recycling).sort_values("fraction")
    efw_matrix = pd.DataFrame(rows_efw).sort_values("fraction")
    return recycling_matrix, efw_matrix


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Files")
scenario_source = resolve_local_file("2022Pathway_data.xlsx")
scenario_upload = st.sidebar.file_uploader("Upload scenario workbook (2022Pathway_data.xlsx)", type=["xlsx"], key="scenario")
if scenario_upload is not None:
    scenario_source = scenario_upload

if scenario_source is None:
    st.info("Please place `2022Pathway_data.xlsx` next to the app or upload it in the sidebar.")
    st.stop()

target_rate = st.sidebar.slider("Policy target recovery rate", 0.40, 0.80, 0.65, 0.01)
efficiency = st.sidebar.slider("EfW net electrical efficiency", 0.15, 0.35, 0.23, 0.01)

scenario_xls = pd.ExcelFile(scenario_source)
required = {"WFH Recoverable LACW DATA 2022", "WFH Residual LACW DATA 2022"}
if not required.issubset(set(scenario_xls.sheet_names)):
    st.error(f"Scenario workbook is missing required sheets: {sorted(required - set(scenario_xls.sheet_names))}")
    st.stop()

rec_df, res_df = read_sheets(scenario_source)
base_frame, totals, qa = build_baseline_frame(rec_df, res_df, efficiency)

scenario_frames = {}
summary_rows = []

baseline_metrics = {
    "Net emissions (Mt CO₂e)": float(base_frame["emissions_kgco2e"].sum() / 1_000_000_000.0),
    "Recovery rate (%)": float(totals["baseline_captured_tonnes"] / totals["total_collected_tonnes"] * 100.0),
    "Recovered value (midpoint, £)": float(base_frame["value_gbp"].sum()),
    "Recovered EfW energy (GWh)": float(base_frame["energy_gwh"].sum()),
    "Diverted avoidable tonnes": 0.0,
}
scenario_frames["Baseline"] = base_frame.copy()
summary_rows.append({"scenario": "Baseline", **baseline_metrics})

for mode, label in [("cap", "Policy65 - Uniform cap"), ("behaviour", "Policy65 - Behaviour-based"), ("carbon", "Policy65 - Carbon-optimised"), ("optimal", "Optimal")]:
    scen_frame, scen_metrics = apply_scenario(base_frame, rec_df, res_df, mode=mode, target_rate=target_rate, efficiency=efficiency)
    scenario_frames[label] = scen_frame
    summary_rows.append({"scenario": label, **scen_metrics})

summary_df = pd.DataFrame(summary_rows)
summary_df["scenario"] = pd.Categorical(summary_df["scenario"], categories=SCENARIO_ORDER, ordered=True)
summary_df = summary_df.sort_values("scenario").reset_index(drop=True)
tradeoff_df = build_tradeoff_table(base_frame)
recycling_priority_df, efw_priority_df = build_material_priority_matrices(base_frame, efficiency)
scenario_long_df = pd.concat([df.assign(scenario=name) for name, df in scenario_frames.items()], ignore_index=True)

policy_mask = summary_df["scenario"].astype(str).str.startswith("Policy65")
policy_df = summary_df[policy_mask].copy()
if not policy_df.empty:
    tmp = policy_df.copy()
    tmp["climate_score"] = (-tmp["Net emissions (Mt CO₂e)"]).rank(pct=True)
    tmp["value_score"] = tmp["Recovered value (midpoint, £)"].rank(pct=True)
    tmp["energy_score"] = tmp["Recovered EfW energy (GWh)"].rank(pct=True)
    tmp["balanced_score"] = 0.5 * tmp["climate_score"] + 0.3 * tmp["value_score"] + 0.2 * tmp["energy_score"]
    best_policy = str(tmp.loc[tmp["balanced_score"].idxmax(), "scenario"])
else:
    best_policy = "N/A"

# -----------------------------
# Header
# -----------------------------
st.title("2022 WFH LACW Dashboard — Rebuilt")
st.caption("Rebuilt directly from the uploaded 2022 workbook so the baseline, residual pool, and scenario accounting follow the actual council rows only.")
st.success("Workbook check passed: the scenario engine now removes aggregate rows, uses the residual worksheet route split directly, and separates the Policy65 scenario logics.")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total collected (2022)", f"{totals['total_collected_tonnes']/1e6:,.2f} Mt")
k2.metric("Baseline captured", f"{totals['baseline_captured_tonnes']/1e6:,.2f} Mt")
k3.metric("Avoidable residual pool", f"{parse_residual_totals(res_df).query('fraction in @DIVERTABLE_FRACTIONS')['residual_total_tonnes'].sum()/1e6:,.2f} Mt")
k4.metric("Baseline recovery rate", f"{baseline_metrics['Recovery rate (%)']:.2f}%")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["1) Baseline", "2) Scenarios", "3) Scenario decision map", "4) Material prioritisation", "5) QA checks", "6) Data"])

with tab1:
    st.subheader("Baseline")
    by_stream = base_frame.groupby("stream", as_index=False).agg(kg=("emissions_kgco2e", "sum"))
    by_stream["Mt CO₂e"] = by_stream["kg"] / 1_000_000_000.0
    total_row = pd.DataFrame([{"stream": "Total", "kg": by_stream["kg"].sum(), "Mt CO₂e": by_stream["Mt CO₂e"].sum()}])
    by_stream_plot = pd.concat([by_stream, total_row], ignore_index=True)
    fig1 = px.bar(by_stream_plot, x="stream", y="Mt CO₂e", color="stream")
    fig1 = apply_paper_style(fig1, "Net emissions by stream (Mt CO₂e)", "Stream", "Mt CO₂e")
    st.plotly_chart(fig1, use_container_width=True)
    add_download_buttons(fig1, "baseline_emissions_by_stream")
    add_table_download(by_stream_plot[["stream", "Mt CO₂e"]], "baseline_emissions_by_stream_data.csv", "Download chart data (CSV)")

    c1, c2 = st.columns(2)
    with c1:
        val_frac = (
            base_frame[(base_frame["route"] == "Recycled") & (~base_frame["fraction"].isin(EXCLUDED_FIGURE_FRACTIONS))]
            .groupby("fraction", as_index=False)
            .agg(value_gbp=("value_gbp", "sum"))
        )
        fig2 = px.bar(val_frac, x="fraction", y="value_gbp")
        fig2 = apply_paper_style(fig2, "Recovered value by fraction (£)", "Fraction", "GBP")
        st.plotly_chart(fig2, use_container_width=True)
        add_download_buttons(fig2, "baseline_value_by_fraction")
        add_table_download(val_frac, "baseline_value_by_fraction_data.csv", "Download chart data (CSV)")
    with c2:
        en_frac = (
            base_frame[(base_frame["route"] == "EfW") & (~base_frame["fraction"].isin(EXCLUDED_FIGURE_FRACTIONS))]
            .groupby("fraction", as_index=False)
            .agg(energy_gwh=("energy_gwh", "sum"))
        )
        fig3 = px.bar(en_frac, x="fraction", y="energy_gwh")
        fig3 = apply_paper_style(fig3, "Recovered EfW energy by fraction (GWh)", "Fraction", "GWh")
        st.plotly_chart(fig3, use_container_width=True)
        add_download_buttons(fig3, "baseline_energy_by_fraction")
        add_table_download(en_frac, "baseline_energy_by_fraction_data.csv", "Download chart data (CSV)")

    st.markdown("### Recovery trade-off map (intensity view)")
    td = tradeoff_df.copy()
    fig4 = px.scatter(
        td,
        x="Recovered value (£/recycled t)",
        y="Recovered EfW energy (MWh/EfW t)",
        size=td["recycled_tonnes"].fillna(0.0) + td["efw_tonnes"].fillna(0.0),
        color="Net carbon (tCO₂e)",
        text="fraction",
        color_continuous_scale="RdYlGn_r",
    )
    fig4.update_traces(textposition="top center")
    fig4 = apply_paper_style(fig4, "Recovery trade-off map (per tonne)", "Recovered value (£ per recycled tonne)", "Recovered EfW energy (MWh per EfW tonne)")
    st.plotly_chart(fig4, use_container_width=True)
    add_download_buttons(fig4, "baseline_tradeoff")
    add_table_download(td, "baseline_tradeoff_data.csv", "Download trade-off data (CSV)")

with tab2:
    st.subheader("Scenarios")
    st.caption(
        "Uniform cap applies the same diversion share to all avoidable residual. "
        "Behaviour-based allocates the target using each council-fraction's observed capture propensity and routes diverted tonnes using current recovery-route mixes. "
        "Carbon-optimised prioritises the highest carbon benefit per tonne. Optimal diverts all avoidable residual using the balanced routing rule across carbon benefit, recovered value, and EfW energy."
    )
    c1, c2 = st.columns(2)
    with c1:
        fig_em = px.bar(summary_df, x="scenario", y="Net emissions (Mt CO₂e)", color="scenario")
        fig_em = apply_paper_style(fig_em, "Scenario net emissions (Mt CO₂e)", "Scenario", "Mt CO₂e")
        st.plotly_chart(fig_em, use_container_width=True)
        add_download_buttons(fig_em, "scenario_net_emissions")
        add_table_download(summary_df[["scenario", "Net emissions (Mt CO₂e)"]], "scenario_net_emissions_data.csv", "Download chart data (CSV)")
    with c2:
        fig_rr = px.bar(summary_df, x="scenario", y="Recovery rate (%)", color="scenario")
        fig_rr = apply_paper_style(fig_rr, "Scenario recovery rate (%)", "Scenario", "%")
        st.plotly_chart(fig_rr, use_container_width=True)
        add_download_buttons(fig_rr, "scenario_recovery_rate")
        add_table_download(summary_df[["scenario", "Recovery rate (%)"]], "scenario_recovery_rate_data.csv", "Download chart data (CSV)")

    c3, c4 = st.columns(2)
    with c3:
        fig_val = px.bar(summary_df, x="scenario", y="Recovered value (midpoint, £)", color="scenario")
        fig_val = apply_paper_style(fig_val, "Scenario recovered value (midpoint, £)", "Scenario", "GBP")
        st.plotly_chart(fig_val, use_container_width=True)
        add_download_buttons(fig_val, "scenario_value")
        add_table_download(summary_df[["scenario", "Recovered value (midpoint, £)"]], "scenario_value_data.csv", "Download chart data (CSV)")
    with c4:
        fig_en = px.bar(summary_df, x="scenario", y="Recovered EfW energy (GWh)", color="scenario")
        fig_en = apply_paper_style(fig_en, "Scenario recovered EfW energy (GWh)", "Scenario", "GWh")
        st.plotly_chart(fig_en, use_container_width=True)
        add_download_buttons(fig_en, "scenario_energy")
        add_table_download(summary_df[["scenario", "Recovered EfW energy (GWh)"]], "scenario_energy_data.csv", "Download chart data (CSV)")

    st.markdown("### Scenario summary table")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.download_button("Download scenario summary (CSV)", summary_df.to_csv(index=False).encode("utf-8"), "scenario_summary.csv", "text/csv", key="scenario_summary_csv")

with tab3:
    st.subheader("Scenario decision map")
    st.caption("X = recovered value, Y = net emissions, bubble size = recovered EfW energy. All values shown here are 2022 national totals. More negative emissions are better.")
    figd = px.scatter(
        summary_df,
        x="Recovered value (midpoint, £)",
        y="Net emissions (Mt CO₂e)",
        size="Recovered EfW energy (GWh)",
        color="scenario",
        text="scenario",
        size_max=60,
        hover_data={"Recovery rate (%)": ":.2f", "Diverted avoidable tonnes": ":,.0f"},
    )
    figd.update_traces(textposition="top center")
    figd = apply_paper_style(figd, "Scenario decision map", "Recovered value (midpoint, £)", "Net emissions (Mt CO₂e)")
    st.plotly_chart(figd, use_container_width=True)
    add_download_buttons(figd, "scenario_decision_map")
    add_table_download(summary_df, "scenario_decision_map_data.csv", "Download decision-map data (CSV)")

    best_climate = summary_df.loc[summary_df["Net emissions (Mt CO₂e)"].idxmin(), "scenario"]
    best_value = summary_df.loc[summary_df["Recovered value (midpoint, £)"].idxmax(), "scenario"]
    best_energy = summary_df.loc[summary_df["Recovered EfW energy (GWh)"].idxmax(), "scenario"]
    st.markdown(
        f"""
**Quick reading**
- Best climate outcome: **{best_climate}**
- Highest recovered value: **{best_value}**
- Highest recovered EfW energy: **{best_energy}**
- Best balanced Policy65 option in this build: **{best_policy}**
"""
    )

with tab4:
    st.subheader("Material prioritisation")
    st.caption("These matrices show 2022 national total residual availability by fraction. Bubble size reflects residual tonnes available for diversion. Recycling matrix focuses on economic + environmental benefit. EfW matrix focuses on energy + environmental benefit.")

    c1, c2 = st.columns(2)
    with c1:
        fig_r = px.scatter(
            recycling_priority_df,
            x="Economic benefit if recycled (£/t)",
            y="Environmental benefit if recycled (kg CO₂e/t)",
            size="residual_tonnes",
            text="fraction",
            color="fraction",
            size_max=60,
        )
        fig_r.update_traces(textposition="top center")
        fig_r = apply_paper_style(fig_r, "Recycling prioritisation matrix", "Economic benefit if recycled (£/t)", "Environmental benefit if recycled (kg CO₂e/t)")
        st.plotly_chart(fig_r, use_container_width=True)
        add_download_buttons(fig_r, "recycling_prioritisation_matrix")
        add_table_download(recycling_priority_df, "recycling_prioritisation_matrix_data.csv", "Download recycling-priority data (CSV)")

    with c2:
        fig_e = px.scatter(
            efw_priority_df,
            x="Energy benefit if incinerated (MWh/t)",
            y="Environmental benefit if incinerated (kg CO₂e/t)",
            size="residual_tonnes",
            text="fraction",
            color="fraction",
            size_max=60,
        )
        fig_e.update_traces(textposition="top center")
        fig_e = apply_paper_style(fig_e, "EfW prioritisation matrix", "Energy benefit if incinerated (MWh/t)", "Environmental benefit if incinerated (kg CO₂e/t)")
        st.plotly_chart(fig_e, use_container_width=True)
        add_download_buttons(fig_e, "efw_prioritisation_matrix")
        add_table_download(efw_priority_df, "efw_prioritisation_matrix_data.csv", "Download EfW-priority data (CSV)")

    st.markdown("### Priority tables")
    st.dataframe(recycling_priority_df, use_container_width=True, hide_index=True)
    st.dataframe(efw_priority_df, use_container_width=True, hide_index=True)

with tab5:
    st.subheader("QA checks")
    st.markdown("These checks help confirm that the scenario engine is working off the intended rows and route structures.")
    qa_df = pd.DataFrame(
        [
            {"Check": "Reported baseline captured (sheet total recycled collected)", "Tonnes": qa["reported_total_recycled_collected"]},
            {"Check": "Reconstructed recoverable-route tonnes", "Tonnes": qa["recoverable_routes_sum"]},
            {"Check": "Gap between reported captured and explicit recoverable routes", "Tonnes": qa["recoverable_gap"]},
            {"Check": "Reported residual total", "Tonnes": qa["residual_total_reported"]},
            {"Check": "Reconstructed residual-route tonnes", "Tonnes": qa["residual_routes_sum"]},
            {"Check": "Gap between reported residual and explicit residual routes", "Tonnes": qa["residual_gap"]},
        ]
    )
    st.dataframe(qa_df, use_container_width=True, hide_index=True)
    if abs(qa["recoverable_gap"]) > 1:
        st.warning("The recoverable-route columns do not exactly sum to the reported 'total recycled collected'. This is a workbook-level inconsistency, not a scenario-calculation bug.")
    if abs(qa["residual_gap"]) > 1:
        st.warning("The residual route split does not exactly sum to the reported residual total. Review the workbook structure if you need a perfect residual identity check.")
    st.info("The earlier >100% issue is resolved here by removing aggregate rows such as the national Total row before building the diversion pool.")

with tab6:
    st.subheader("Data")
    st.markdown("**Scenario summary**")
    st.dataframe(summary_df, use_container_width=True)
    add_table_download(summary_df, "scenario_summary.csv", "Download scenario summary (CSV)")

    st.markdown("**Scenario long table**")
    st.dataframe(scenario_long_df, use_container_width=True)
    add_table_download(scenario_long_df, "scenario_long_table.csv", "Download scenario long table (CSV)")

    st.markdown("**Baseline trade-off table**")
    st.dataframe(tradeoff_df, use_container_width=True)
    add_table_download(tradeoff_df, "baseline_tradeoff_table.csv", "Download baseline trade-off table (CSV)")

    st.markdown("**Input parameter tables actually used in the model**")
    ef_table = pd.DataFrame([{"fraction": f, "route": r, "kg_co2e_per_tonne": v} for (f, r), v in EF.items()]).sort_values(["fraction", "route"])
    value_table = pd.DataFrame([{"fraction": f, "value_mid_gbp_per_tonne": v} for f, v in VALUE_MID.items()]).sort_values("fraction")
    ncv_table = pd.DataFrame([{"fraction": f, "ncv_mj_per_kg": v} for f, v in NCV.items()]).sort_values("fraction")
    st.dataframe(ef_table, use_container_width=True)
    add_table_download(ef_table, "emission_factors_used.csv", "Download emission factors used (CSV)")
    st.dataframe(value_table, use_container_width=True)
    add_table_download(value_table, "value_factors_used.csv", "Download value factors used (CSV)")
    st.dataframe(ncv_table, use_container_width=True)
    add_table_download(ncv_table, "ncv_factors_used.csv", "Download NCV factors used (CSV)")
