# ============================================================
# TruckShield AI — Commercial Truck Insurance Dashboard
#
# A five-agent actuarial + ML pricing engine wrapped in a
# Streamlit dashboard. Works entirely offline with no API key.
# The AI Assistant tab (Tab 7) activates only when you provide
# a Gemini API key in the sidebar.
#
# Run with:  streamlit run truckshield_app.py
# ============================================================

# All imports in one place — nothing is imported inside functions
# except google.generativeai (loaded on demand since it's optional)
import io
import re
import json
import math
import datetime
import urllib.request
import urllib.parse

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, PageBreak, KeepTogether,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart

st.set_page_config(page_title="TruckShield AI", page_icon="🚛", layout="wide")

# ============================================================
# SECTION 1 — PRICING ENGINE
#
# SEM is the master configuration dictionary for the entire
# pricing and risk model. The four actuarial parameters at the
# top (expenses_loading, profit_margin, other_loading,
# deductible_pct) can be overridden at runtime from the
# sidebar's "Actuarial Parameters" panel — see Section 3 UI.
#
# Constraint enforced in the sidebar: the combined loading
# (expenses + profit + other) cannot exceed 100%, which would
# make the denominator in the gross premium formula zero or
# negative. Deductible has no such cap — it is a policyholder
# cost-share mechanism, not part of the loading stack.
# ============================================================

# These are the defaults. All four are overridable from the sidebar.
# The sidebar enforces that expenses_loading + profit_margin + other_loading ≤ 100%.
SEM = {
    "expenses_loading": 0.25,       # e.g., 25% covers acquisition, admin, reinsurance costs
    "profit_margin": 0.05,          # target underwriting profit margin
    "other_loading": 0.00,          # any additional loading (catastrophe margin, IBNR buffer, etc.)
    "deductible_pct": 0.10,         # policyholder absorbs this share per OD claim
    "policy_limit_pct_of_idv": 1.0,

    # OD pricing anchors — market-calibrated Own Damage rates as a percentage of IDV.
    # The base rate, floor, and cap are derived from IRDAI filings and insurer market surveys.
    "od_base_rate_of_idv":        0.025,
    "od_rate_floor_pct_of_idv":   0.015,
    "od_rate_cap_pct_of_idv":     0.09,
    "od_relativity_bounds":       (0.6, 3.0),
    "risk_score_reference_anchor": 5.0,
    "tp_assumed_loss_ratio":      0.70,    # TP is IRDAI-tariffed; this is the implied loss ratio

    # NCB (No Claim Bonus) schedule per IRDAI motor regulations.
    # Year 1 → 20%, Year 5+ → 50% discount on OD only.
    "ncb_grid": {0: 0.00, 1: 0.20, 2: 0.25, 3: 0.35, 4: 0.45, 5: 0.50},

    # Source: NDMA State Disaster Management Plans + IMD 3-year rolling avg
    "climate_risk_base": {
        "Rajasthan":        {"flood": 2, "fog": 4, "cyclone": 1},
        "Gujarat":          {"flood": 5, "fog": 2, "cyclone": 6},
        "Odisha":           {"flood": 7, "fog": 2, "cyclone": 9},
        "Kerala":           {"flood": 7, "fog": 1, "cyclone": 5},
        "Tamil Nadu":       {"flood": 5, "fog": 1, "cyclone": 7},
        "Haryana":          {"flood": 3, "fog": 8, "cyclone": 1},
        "Punjab":           {"flood": 3, "fog": 9, "cyclone": 1},
        "Delhi NCR":        {"flood": 4, "fog": 9, "cyclone": 1},
        "Uttar Pradesh":    {"flood": 6, "fog": 8, "cyclone": 1},
        "Bihar":            {"flood": 8, "fog": 7, "cyclone": 2},
        "West Bengal":      {"flood": 7, "fog": 4, "cyclone": 6},
        "Andhra Pradesh":   {"flood": 5, "fog": 1, "cyclone": 7},
        "Maharashtra":      {"flood": 5, "fog": 2, "cyclone": 3},
        "Himachal Pradesh": {"flood": 4, "fog": 6, "cyclone": 1},
        "Uttarakhand":      {"flood": 6, "fog": 5, "cyclone": 1},
        "Jharkhand":        {"flood": 5, "fog": 4, "cyclone": 1},
        "Madhya Pradesh":   {"flood": 4, "fog": 4, "cyclone": 1},
        "Karnataka":        {"flood": 4, "fog": 1, "cyclone": 3},
        "Telangana":        {"flood": 4, "fog": 1, "cyclone": 4},
        "Chhattisgarh":     {"flood": 4, "fog": 2, "cyclone": 1},
        "Assam":            {"flood": 9, "fog": 4, "cyclone": 2},
        "DEFAULT":          {"flood": 4, "fog": 3, "cyclone": 3},
    },

    "climate_zone_modifiers": {
        "Dense Fog Zone (North India)":    {"flood": 1.0, "fog": 1.40, "cyclone": 1.0},
        "Cyclone Prone Coast (East)":      {"flood": 1.1, "fog": 1.0,  "cyclone": 1.45},
        "Cyclone Prone Coast (West)":      {"flood": 1.1, "fog": 1.0,  "cyclone": 1.35},
        "Flood Plains (Indo-Gangetic)":    {"flood": 1.45, "fog": 1.1, "cyclone": 1.0},
        "Flash Flood (NE/Hill States)":    {"flood": 1.55, "fog": 1.0, "cyclone": 1.0},
        "Landslide Prone (Himalayan)":     {"flood": 1.3,  "fog": 1.2, "cyclone": 1.0},
        "Extreme Heat Belt (Central)":     {"flood": 0.9,  "fog": 0.9, "cyclone": 0.9},
        "Drought Prone (Deccan)":          {"flood": 0.8,  "fog": 0.8, "cyclone": 0.9},
        "Standard Low Risk":               {"flood": 1.0,  "fog": 1.0, "cyclone": 1.0},
    },

    "route_climate_risk": {
        "Hill/Mountain":        {"route_risk": 9.0, "flood_amp": 1.30, "cyclone_amp": 0.80},
        "Rural/Remote":         {"route_risk": 7.0, "flood_amp": 1.20, "cyclone_amp": 1.20},
        "Highway (Inter-City)": {"route_risk": 5.0, "flood_amp": 1.00, "cyclone_amp": 1.10},
        "Mixed":                {"route_risk": 5.0, "flood_amp": 1.10, "cyclone_amp": 1.00},
        "Urban (City)":         {"route_risk": 4.0, "flood_amp": 0.90, "cyclone_amp": 0.90},
    },

    "accident_prob_route": {
        "Hill/Mountain":        9.2,
        "Rural/Remote":         7.5,
        "Highway (Inter-City)": 5.8,
        "Mixed":                5.2,
        "Urban (City)":         3.8,
    },

    "accident_prob_state": {
        "Uttar Pradesh": 8.5, "Tamil Nadu": 8.0, "Madhya Pradesh": 7.8,
        "Rajasthan": 7.5, "Maharashtra": 7.0, "Karnataka": 6.8,
        "Andhra Pradesh": 6.5, "Gujarat": 6.3, "Telangana": 6.0,
        "West Bengal": 5.8, "Bihar": 5.5, "Haryana": 5.3, "Punjab": 5.0,
        "Odisha": 4.8, "Kerala": 4.5, "Jharkhand": 4.3, "Assam": 4.2,
        "Delhi NCR": 4.0, "Uttarakhand": 5.5, "Himachal Pradesh": 5.8,
        "Chhattisgarh": 4.5, "DEFAULT": 5.5,
    },

    "cargo_risk_map": {
        "Chemical/Hazardous": 9.0, "Petroleum Products": 8.5, "Livestock": 7.5,
        "Refrigerated/Perishable": 7.0, "Pharmaceuticals": 6.5,
        "Steel/Metal/Iron": 6.0, "Container/Intermodal": 5.5,
        "Coal/Minerals": 5.0, "Timber/Wood": 5.0, "Construction Material": 4.5,
        "Sand/Gravel/Aggregate": 4.0, "Agricultural Produce": 4.0,
        "FMCG/Retail": 3.5, "General Goods": 3.0,
    },

    "route_risk_map": {
        "Hill/Mountain": 9.0, "Rural/Remote": 7.0, "Highway (Inter-City)": 5.0,
        "Mixed": 5.0, "Urban (City)": 4.0,
    },

    "bs_norm_safety": {"BS-VI": 1.0, "BS-IV": 0.8, "BS-III": 0.6, "BS-II": 0.4},
    "fuel_safety":    {"Electric (EV)": 1.0, "CNG": 0.8, "LNG": 0.7, "Diesel": 0.5},

    "fatigue_monitor_safety": {
        "Advanced ADAS": 1.0,
        "Basic Alert":   0.65,
        "Not Installed": 0.20,
    },

    "driver_training_safety": {
        "Yes":     1.0,
        "Partial": 0.65,
        "No":      0.25,
    },

    "night_travel_risk": {
        "Primarily Night":    1.30,
        "Mixed (Day & Night)": 1.12,
        "Primarily Day":      1.00,
    },

    "tp_premium_by_gvw_slab": [
        (7500,   16049),   
        (12000,  20000),   
        (25000,  27186),   
        (40000,  35000),   
        (10**9,  44242),   
    ],
    "min_od_rate_pct_of_idv": 0.015,

    "vehicle_categories": ["LMV (< 3.5T)", "LCV (3.5-7.5T)", "MCV (7.5-12T)", "HCV (12-25T)", "Multi-Axle (>25T)"],
    "fuel_types":         ["Diesel", "CNG", "LNG", "Electric (EV)"],
    "bs_norms":           ["BS-II", "BS-III", "BS-IV", "BS-VI"],
    "permit_types":       ["State Permit", "National Permit", "All India Tourist", "Contract Carriage", "Goods Carriage", "Hazardous Goods"],
    "ownership_types":    ["Proprietorship", "Partnership", "Private Ltd Company", "Public Ltd Company", "Transport Corporation", "Individual Owner-Operator"],
    "route_types":        ["Rural/Remote", "Urban (City)", "Mixed", "Highway (Inter-City)", "Hill/Mountain"],
    "highway_usages":     ["Primarily NH (National Highway)", "Mixed NH+SH", "Primarily SH (State Highway)",
                           "Remote/Hill Routes", "Urban/City Routes"],
    "truck_models":       ["Tata LPT 2518", "Tata 407 LCV", "Ashok Leyland Boss", "Ashok Leyland Captain",
                           "Eicher Pro 3015", "Eicher Pro 6049", "Mahindra Furio 7", "Bharat Benz 1617R",
                           "Bharat Benz 2823C", "Tata Signa 4823"],
    "climate_zones": [
        "Standard Low Risk", "Dense Fog Zone (North India)", "Cyclone Prone Coast (East)",
        "Cyclone Prone Coast (West)", "Flood Plains (Indo-Gangetic)", "Flash Flood (NE/Hill States)",
        "Landslide Prone (Himalayan)", "Extreme Heat Belt (Central)", "Drought Prone (Deccan)",
    ],
    "fatigue_monitor_options": ["Not Installed", "Basic Alert", "Advanced ADAS"],
    "driver_training_options": ["No", "Partial", "Yes"],
    "travel_patterns":         ["Primarily Day", "Mixed (Day & Night)", "Primarily Night"],

    "valid_ranges": {
        "vehicle_age_yrs":              (0, 30),
        "gross_vehicle_weight_kg":      (500, 60000),
        "fleet_size":                   (1, 500),
        "years_in_business":            (0, 100),
        "idv_insured_declared_value":   (10000, 50000000),
        "overloading_incidents":        (0, 20),
        "average_driver_experience":    (0, 45),
        "driver_turnover_rate":         (0, 100),
        "at_fault_accidents":           (0, 20),
        "traffic_violations":           (0, 50),
        "total_claims_count":           (0, 50),
        "night_travel_frequency":       (0, 100),
        "day_travel_frequency":         (0, 100),
    },

    "required_quote_fields": ["vehicle_age_yrs", "vehicle_category", "truck_model", "goods_category"],

    "cat_cols": [
        "vehicle_category", "truck_model", "fuel_type", "bs_emission_norm",
        "permit_type", "ownership_type", "goods_category", "route_type",
        "highway_usage", "state", "rto_district",
        "fatigue_monitoring_system", "climate_zone",
        "driver_training_program", "travel_time_pattern",
    ],
}

STATES = list(SEM["climate_risk_base"].keys())
STATES.remove("DEFAULT")


# Column name mapping to handle uploaded files where Excel wraps long headers across
# two lines. This normalises them to the short snake_case names the engine expects.
_COL_RENAMES = {
    "average_driver_experience\n(years)":  "average_driver_experience",
    "driver_turnover_rate\n(% per yr)":    "driver_turnover_rate",
    "at_fault_accidents\n(last 3 yr)":     "at_fault_accidents",
    "traffic_violations\n(last 12m)":      "traffic_violations",
    "driver_training\nprogram":            "driver_training_program",
    "total_claims_count\n(3yr)":           "total_claims_count",
    "total_claim_amount_paid\n(INR)":      "total_claim_amount_paid",
    "night_travel_frequency\n(%)":         "night_travel_frequency",
    "day_travel_frequency\n(%)":           "day_travel_frequency",
}


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=_COL_RENAMES)


# Generates a realistic synthetic portfolio for demo and testing.
# Climate zones, travel patterns, and claim counts are correlated to their
# geographic context so the output doesn't look uniformly random.
def generate_sample_dataset(n=500, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rto_districts = ["Mumbai", "Pune", "Nagpur", "Lucknow", "Kanpur", "Jaipur", "Ahmedabad",
                     "Surat", "Chennai", "Coimbatore", "Bengaluru", "Hyderabad", "Patna",
                     "Kolkata", "Bhubaneswar", "Guwahati", "Chandigarh", "Dehradun"]
    major_cities = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata",
                    "Pune", "Ahmedabad", "Jaipur", "Lucknow", "Surat", "Nagpur",
                    "Indore", "Bhopal", "Chandigarh", "Patna", "Guwahati", "Kochi"]

    rows = []
    for i in range(n):
        inception = datetime.date(2023, 1, 1) + datetime.timedelta(days=int(rng.integers(0, 700)))
        expiry    = inception + datetime.timedelta(days=365)
        state     = rng.choice(STATES)
        route     = rng.choice(SEM["route_types"])

        fog_states  = {"Haryana", "Punjab", "Delhi NCR", "Uttar Pradesh", "Bihar"}
        cyc_east    = {"Odisha", "West Bengal", "Andhra Pradesh"}
        cyc_west    = {"Gujarat", "Kerala", "Tamil Nadu"}
        flood_plain = {"Bihar", "Uttar Pradesh", "Assam"}
        hill_states = {"Himachal Pradesh", "Uttarakhand"}
        if state in fog_states:
            climate_zone = rng.choice(["Dense Fog Zone (North India)", "Flood Plains (Indo-Gangetic)", "Standard Low Risk"],
                                       p=[0.55, 0.25, 0.20])
        elif state in cyc_east:
            climate_zone = rng.choice(["Cyclone Prone Coast (East)", "Standard Low Risk"], p=[0.70, 0.30])
        elif state in cyc_west:
            climate_zone = rng.choice(["Cyclone Prone Coast (West)", "Standard Low Risk"], p=[0.60, 0.40])
        elif state in flood_plain:
            climate_zone = rng.choice(["Flood Plains (Indo-Gangetic)", "Flash Flood (NE/Hill States)", "Standard Low Risk"],
                                       p=[0.55, 0.15, 0.30])
        elif state in hill_states:
            climate_zone = rng.choice(["Landslide Prone (Himalayan)", "Standard Low Risk"], p=[0.65, 0.35])
        else:
            climate_zone = rng.choice(SEM["climate_zones"],
                                       p=[0.35, 0.10, 0.08, 0.08, 0.10, 0.04, 0.05, 0.10, 0.10])

        night_pct = int(rng.integers(10, 91))
        day_pct   = 100 - night_pct
        if night_pct > 65:
            travel_pattern = "Primarily Night"
        elif night_pct < 35:
            travel_pattern = "Primarily Day"
        else:
            travel_pattern = "Mixed (Day & Night)"

        total_claims = int(rng.integers(0, 7))
        claim_amount = int(rng.integers(0, 300000)) if total_claims > 0 else 0
        last_claim   = (inception - datetime.timedelta(days=int(rng.integers(30, 1100)))).strftime("%Y-%m-%d") \
                       if total_claims > 0 else None

        dests = rng.choice(major_cities, size=3, replace=False).tolist()

        rows.append({
            "policy_id":                    f"POL{i+1:05d}",
            "vehicle_registration_number":  f"MH{rng.integers(1,50):02d}AB{rng.integers(1000,9999)}",
            "vehicle_age_yrs":              int(rng.integers(0, 21)),
            "vehicle_category":             rng.choice(SEM["vehicle_categories"]),
            "truck_model":                  rng.choice(SEM["truck_models"]),
            "gross_vehicle_weight_kg":      int(rng.integers(2000, 49000)),
            "fuel_type":                    rng.choice(SEM["fuel_types"], p=[0.7, 0.15, 0.05, 0.10]),
            "bs_emission_norm":             rng.choice(SEM["bs_norms"], p=[0.05, 0.15, 0.35, 0.45]),
            "permit_type":                  rng.choice(SEM["permit_types"]),
            "ownership_type":               rng.choice(SEM["ownership_types"]),
            "goods_category":               rng.choice(list(SEM["cargo_risk_map"].keys())),
            "route_type":                   route,
            "highway_usage":                rng.choice(SEM["highway_usages"]),
            "state":                        state,
            "rto_district":                 rng.choice(rto_districts),
            "fleet_size":                   int(rng.integers(1, 200)),
            "years_in_business":            int(rng.integers(0, 40)),
            "idv_insured_declared_value":   int(rng.integers(300000, 8000000)),
            "inception_date":               inception,
            "expiry_date":                  expiry,
            "fatigue_monitoring_system":    rng.choice(SEM["fatigue_monitor_options"], p=[0.60, 0.26, 0.14]),
            "overloading_incidents":        int(rng.integers(0, 4)),
            "climate_zone":                 climate_zone,
            "average_driver_experience":    round(float(rng.uniform(1, 23)), 1),
            "driver_turnover_rate":         round(float(rng.uniform(4, 50)), 1),
            "at_fault_accidents":           int(rng.integers(0, 6)),
            "traffic_violations":           int(rng.integers(0, 16)),
            "driver_training_program":      rng.choice(SEM["driver_training_options"], p=[0.42, 0.30, 0.28]),
            "top_destination_1":            dests[0],
            "top_destination_2":            dests[1],
            "top_destination_3":            dests[2],
            "total_claims_count":           total_claims,
            "total_claim_amount_paid":      claim_amount,
            "last_claim_date":              last_claim,
            "night_travel_frequency":       night_pct,
            "day_travel_frequency":         day_pct,
            "travel_time_pattern":          travel_pattern,
        })
    return pd.DataFrame(rows)


def infer_ncb_tier(total_claims_count: int, years_since_last_claim: float) -> int:
    """
    Infers the NCB tier from claims history. A policy with zero lifetime claims
    is assumed to be at tier 3 (35% discount) as a conservative starting point
    for new data where claim history may be incomplete.
    """
    if total_claims_count == 0:
        if years_since_last_claim >= 99:
            return 3   
        clean_years = int(min(years_since_last_claim, 5))
        return clean_years
    else:
        if years_since_last_claim < 1.0:
            return 0   
        clean_years = int(min(years_since_last_claim, 5))
        return clean_years

def ncb_discount_from_tier(ncb_tier: int) -> float:
    return SEM["ncb_grid"].get(min(ncb_tier, 5), 0.0)

def compute_premium_components(gvw: float, idv: float, risk_score: float,
                                accident_load: float, total_uw_loading: float,
                                ncb_tier: int = 0) -> dict:
    # TP is IRDAI-tariffed — fixed by GVW slab, not negotiable
    tp_gross = SEM["tp_premium_by_gvw_slab"][-1][1]
    for max_kg, premium in SEM["tp_premium_by_gvw_slab"]:
        if gvw <= max_kg:
            tp_gross = premium
            break
    tp_pure = round(tp_gross * SEM["tp_assumed_loss_ratio"], 2)

    relativity = risk_score / SEM["risk_score_reference_anchor"]
    lo, hi = SEM["od_relativity_bounds"]
    relativity = min(max(relativity, lo), hi)

    od_rate = SEM["od_base_rate_of_idv"] * relativity * accident_load
    od_rate = min(max(od_rate, SEM["od_rate_floor_pct_of_idv"]), SEM["od_rate_cap_pct_of_idv"])
    od_pure = round(od_rate * idv, 2)

    # The denominator converts pure premium to gross by stripping out all loadings.
    # other_loading is now user-configurable from the sidebar alongside expenses and profit.
    # We cap the denominator at 0.50 so the gross never blows up to absurd levels.
    denom = max(0.50, 1 - SEM["expenses_loading"] - SEM["profit_margin"]
                       - SEM.get("other_loading", 0.0) - total_uw_loading)
    od_gross = round(od_pure / denom, 2)

    ncb_discount = ncb_discount_from_tier(ncb_tier)
    od_gross_pre_ncb = od_gross
    od_gross_after_ncb = round(od_gross_pre_ncb * (1.0 - ncb_discount), 2)
    ncb_saving = round(od_gross_pre_ncb - od_gross_after_ncb, 2)

    total_pure  = round(tp_pure + od_pure, 2)
    total_gross = round(tp_gross + od_gross_after_ncb, 2)

    return {
        "tp_pure_premium_compulsory":    tp_pure,
        "tp_gross_premium_compulsory":   round(tp_gross, 2),
        "od_pure_premium_optional":      od_pure,
        "od_gross_pre_ncb":              od_gross_pre_ncb,
        "od_gross_premium_optional":     od_gross_after_ncb,
        "ncb_tier":                      ncb_tier,
        "ncb_discount_pct":             round(ncb_discount * 100, 1),
        "ncb_saving":                    ncb_saving,
        "total_pure_premium":            total_pure,
        "total_gross_premium":           total_gross,
        "relativity":                    round(relativity, 3),
        "od_rate_pct_of_idv":           round(od_rate * 100, 4),
    }

def compute_minimum_realistic_premium(gvw: float, idv: float) -> float:
    tp_floor = SEM["tp_premium_by_gvw_slab"][-1][1]
    for max_kg, premium in SEM["tp_premium_by_gvw_slab"]:
        if gvw <= max_kg:
            tp_floor = premium
            break
    od_floor = SEM["min_od_rate_pct_of_idv"] * idv
    return round(tp_floor + od_floor, 2)

def compute_route_risk_score(route_type: str, highway_usage: str,
                              top_dest_1: str = "", top_dest_2: str = "",
                              top_dest_3: str = "") -> float:
    route_base = SEM["route_climate_risk"].get(route_type, {}).get("route_risk", 5.0)

    hw_modifier = {
        "Remote/Hill Routes":               1.20,
        "Primarily SH (State Highway)":     1.05,
        "Mixed NH+SH":                      1.00,
        "Primarily NH (National Highway)":  0.95,
        "Urban/City Routes":                0.85,
    }.get(highway_usage, 1.0)

    dests = [d for d in [top_dest_1, top_dest_2, top_dest_3] if d and str(d).strip()]
    dest_penalty = 1.0 + (len(set(dests)) - 1) * 0.02  

    raw = route_base * hw_modifier * dest_penalty
    return round(min(10.0, raw), 2)


def compute_accident_probability(state, route_type, climate_flood, climate_fog, climate_cyclone,
                                  vehicle_age, gvw, cargo_category,
                                  night_travel_pct=50.0, overloading=0,
                                  at_fault=0, traffic_viol=0,
                                  travel_pattern="Mixed (Day & Night)") -> float:

    route_base = SEM["accident_prob_route"].get(route_type, 5.5)   
    state_base = SEM["accident_prob_state"].get(state, SEM["accident_prob_state"]["DEFAULT"])  
    combined_base = 0.55 * route_base + 0.45 * state_base         
    base_contrib  = (combined_base / 10.0) * 4.0                  

    age_contrib = min(vehicle_age, 20) / 20.0 * 1.0

    cargo_risk    = SEM["cargo_risk_map"].get(cargo_category, 4.0)
    cargo_contrib = (cargo_risk / 10.0) * 0.8

    climate_raw   = 0.04 * min(climate_fog, 10) + 0.02 * min(climate_flood, 10) + 0.01 * min(climate_cyclone, 10)
    climate_contrib = min(climate_raw, 0.8)

    night_norm    = min(night_travel_pct / 100.0, 1.0)
    travel_sc     = {"Primarily Night": 1.0, "Mixed (Day & Night)": 0.55, "Primarily Day": 0.15}.get(travel_pattern, 0.55)
    night_contrib = max(night_norm * 0.8, travel_sc) * 1.0   
    night_contrib = min(night_contrib, 1.0)

    overload_contrib = min(overloading, 5) / 5.0 * 0.7

    fault_contrib   = min(at_fault, 5) / 5.0 * 0.75
    viol_contrib    = min(traffic_viol, 10) / 10.0 * 0.45
    driver_contrib  = min(fault_contrib + viol_contrib, 1.2)

    gvw_contrib = min(gvw / 60000, 1.0) * 0.5

    raw = (base_contrib + age_contrib + cargo_contrib + climate_contrib
           + night_contrib + overload_contrib + driver_contrib + gvw_contrib)

    compressed = 9.0 * (1 - math.exp(-raw / 7.0))   

    return round(min(9.0, max(0.5, compressed)), 2)


def get_live_weather_traffic(location: str) -> dict:
    """Fetches live weather from Open-Meteo and derives a basic driving risk level."""
    result = {"location": location, "source": "Open-Meteo (live)", "error": None}
    try:
        geo_url = (f"https://geocoding-api.open-meteo.com/v1/search?name="
                   f"{requests.utils.quote(location)}&count=1&language=en&format=json")
        geo_data = requests.get(geo_url, timeout=5).json()
        if not geo_data.get("results"):
            result["error"] = f"Location '{location}' not found."
            return result

        loc = geo_data["results"][0]
        lat, lon, name = loc["latitude"], loc["longitude"], loc.get("name", location)
        result["resolved_name"] = name
        result["lat"], result["lon"] = lat, lon

        wx_url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                  f"&current=temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m,"
                  f"wind_gusts_10m,visibility,weather_code&timezone=Asia%2FKolkata&forecast_days=1")
        wx_data = requests.get(wx_url, timeout=5).json().get("current", {})

        wmo_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Fog", 48: "Icy fog", 51: "Light drizzle", 53: "Moderate drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
            80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
            95: "Thunderstorm", 96: "Thunderstorm+hail", 99: "Thunderstorm+heavy hail",
        }
        wcode = wx_data.get("weather_code", 0)
        result["temperature_c"]    = wx_data.get("temperature_2m")
        result["humidity_pct"]     = wx_data.get("relative_humidity_2m")
        result["precipitation_mm"] = wx_data.get("precipitation")
        result["wind_speed_kmh"]   = wx_data.get("wind_speed_10m")
        result["wind_gusts_kmh"]   = wx_data.get("wind_gusts_10m")
        result["visibility_m"]     = wx_data.get("visibility")
        result["weather_condition"] = wmo_codes.get(wcode, f"Code {wcode}")

        rain = result["precipitation_mm"] or 0
        vis  = result["visibility_m"] or 10000
        wind = result["wind_speed_kmh"] or 0
        if rain > 10 or vis < 500 or wind > 60:
            result["driving_risk"] = "HIGH"
        elif rain > 3 or vis < 2000 or wind > 35:
            result["driving_risk"] = "MODERATE"
        else:
            result["driving_risk"] = "LOW"

        hour = datetime.datetime.now().hour
        if   8 <= hour <= 10 or 17 <= hour <= 20: result["traffic_density"] = "HIGH (peak hours)"
        elif hour >= 22 or hour <= 5:              result["traffic_density"] = "LOW (off-peak/night)"
        else:                                       result["traffic_density"] = "MODERATE"
    except Exception as e:
        result["error"] = f"Weather fetch failed: {str(e)[:80]}"
    return result


def generate_premium_tips(params: dict, result: dict, sem: dict) -> dict:
    pr  = result["pricing"]
    rs  = result["risk_scores"]
    uw  = result["underwriting"]

    gvw      = params.get("gross_vehicle_weight_kg", 15000)
    idv      = params.get("idv_insured_declared_value", 1000000)
    age      = params.get("vehicle_age_yrs", 5)
    bs       = params.get("bs_emission_norm", "BS-IV")
    fuel     = params.get("fuel_type", "Diesel")
    fatigue  = params.get("fatigue_monitoring_system", "Not Installed")
    training = params.get("driver_training_program", "No")
    at_fault = params.get("at_fault_accidents", 0)
    viol     = params.get("traffic_violations", 0)
    overload = params.get("overloading_incidents", 0)
    night_pct= params.get("night_travel_frequency", 50)
    claims   = params.get("total_claims_count", 0)
    turnover = params.get("driver_turnover_rate", 25)
    route    = params.get("route_type", "Mixed")
    permit   = params.get("permit_type", "State Permit")
    cargo    = params.get("goods_category", "General Goods")
    fleet    = params.get("fleet_size", 1)

    od_gross = pr.get("od_gross_premium_optional", 0) or 0
    tp_gross = pr.get("tp_gross_premium_compulsory", 0) or 0
    total_gross = pr.get("total_gross_premium", 0) or 0
    composite = rs.get("composite_risk", 5.0)
    relativity = pr.get("relativity", 1.0)

    tp_tips, od_tips, general_tips = [], [], []

    tp_slabs = sem["tp_premium_by_gvw_slab"]
    current_tp = tp_gross
    current_slab_idx = len(tp_slabs) - 1
    for i, (max_kg, prem) in enumerate(tp_slabs):
        if gvw <= max_kg:
            current_slab_idx = i
            break
    if current_slab_idx > 0:
        prev_slab_max, prev_slab_prem = tp_slabs[current_slab_idx - 1]
        slab_min = tp_slabs[current_slab_idx - 2][0] if current_slab_idx > 1 else 0
        kg_to_drop = gvw - prev_slab_max
        tp_saving = current_tp - prev_slab_prem
        if kg_to_drop <= 3000 and tp_saving > 0:
            tp_tips.append({
                "icon": "⚖️",
                "title": "Reduce GVW to drop a tariff slab",
                "detail": (f"Your vehicle ({gvw:,} kg) is {kg_to_drop:,} kg above the "
                           f"{prev_slab_max:,} kg TP tariff boundary. Reconfiguring payload "
                           f"(lighter body, load redistribution) to stay under {prev_slab_max:,} kg "
                           f"GVW drops TP from ₹{current_tp:,.0f} to ₹{prev_slab_prem:,.0f}."),
                "saving_note": f"Potential TP saving: ₹{tp_saving:,.0f}/yr",
                "logic": ("IRDAI fixes TP rates by GVW slab, not by exact weight. "
                          "Dropping a slab is the only legal way to reduce TP premium.")
            })

    tp_tips.append({
        "icon": "📜",
        "title": "TP premium is non-negotiable — focus on OD to reduce total premium",
        "detail": (f"TP of ₹{tp_gross:,.0f} is fixed by the IRDAI/MoRTH GVW-slab tariff for "
                   f"your vehicle class. No insurer can offer a discount on this component. "
                   f"All premium savings come from reducing your OD component (currently ₹{od_gross:,.0f})."),
        "saving_note": "OD is insurer-priced — every tip below reduces your OD gross premium.",
        "logic": "Regulatory: Motor Vehicles Act 1988 mandates TP cover at IRDAI-set rates."
    })

    def estimate_od_saving_from_composite_drop(composite_drop: float) -> float:
        new_comp = max(0, composite - composite_drop)
        new_rel  = min(max(new_comp / 5.0, 0.6), 3.0)
        old_rel  = relativity
        if old_rel <= 0: return 0
        saving = od_gross * (1 - new_rel / old_rel)
        return max(0, saving)

    if fatigue == "Not Installed":
        saving_est = estimate_od_saving_from_composite_drop(0.6)
        od_tips.append({
            "icon": "🤖",
            "title": "Install an ADAS / Fatigue Monitoring System",
            "detail": ("Your vehicle has no fatigue monitoring. Installing an Advanced Driver Assistance "
                       "System (ADAS) or basic drowsiness alert improves safety_score by raising the "
                       "fatigue_monitor_safety weight from 0.20 → 1.0, which lowers composite risk "
                       "and therefore your OD relativity."),
            "saving_note": f"Estimated OD saving: ~₹{saving_est:,.0f}/yr. ADAS units available ₹25,000–₹80,000 ex-market.",
            "logic": ("fatigue_monitor_safety: Not Installed=0.20, Basic Alert=0.65, Advanced ADAS=1.0. "
                      "Safety score = 10×(0.40×BS + 0.35×fuel + 0.25×fatigue). Higher safety → lower composite risk.")
        })
    elif fatigue == "Basic Alert":
        saving_est = estimate_od_saving_from_composite_drop(0.3)
        od_tips.append({
            "icon": "🤖",
            "title": "Upgrade to Advanced ADAS from Basic Alert",
            "detail": ("Upgrading from Basic Alert to Advanced ADAS increases fatigue_monitor_safety "
                       "from 0.65 → 1.0, improving your safety score and reducing composite risk."),
            "saving_note": f"Estimated OD saving: ~₹{saving_est:,.0f}/yr.",
            "logic": "Same safety_score formula as above. Incremental but meaningful improvement."
        })

    if bs in ("BS-II", "BS-III"):
        saving_est = estimate_od_saving_from_composite_drop(0.5)
        od_tips.append({
            "icon": "🌿",
            "title": "Upgrade to BS-IV or BS-VI vehicle",
            "detail": (f"Your vehicle is {bs}. Upgrading to BS-VI raises bs_norm_safety from "
                       f"{sem['bs_norm_safety'].get(bs, 0.5)} → 1.0, improving safety score and "
                       f"reducing composite risk. BS-VI vehicles also qualify for better OD rates "
                       f"across most Indian insurers."),
            "saving_note": f"Estimated OD saving: ~₹{saving_est:,.0f}/yr on this vehicle.",
            "logic": "bs_norm_safety weights: BS-II=0.4, BS-III=0.6, BS-IV=0.8, BS-VI=1.0."
        })
    elif bs == "BS-IV" and fuel == "Diesel":
        saving_est = estimate_od_saving_from_composite_drop(0.35)
        od_tips.append({
            "icon": "🌿",
            "title": "Upgrade to BS-VI on next vehicle renewal",
            "detail": ("You're on BS-IV Diesel. Moving to BS-VI (especially with CNG or LNG) "
                       "significantly improves both BS-norm and fuel safety weights, giving the "
                       "largest single safety score improvement available."),
            "saving_note": f"Estimated OD saving at renewal: ~₹{saving_est:,.0f}/yr.",
            "logic": "fuel_safety: Diesel=0.5, CNG=0.8, LNG=0.7, EV=1.0. Both weights feed safety_score."
        })

    if fuel == "Diesel" and bs in ("BS-IV", "BS-VI"):
        saving_est = estimate_od_saving_from_composite_drop(0.25)
        od_tips.append({
            "icon": "⛽",
            "title": "Consider CNG/LNG conversion for OD premium benefit",
            "detail": ("Diesel fuel_safety weight is 0.5. Converting to CNG raises it to 0.8 "
                       "(+60%), improving safety score and reducing composite risk. CNG kits for "
                       "commercial trucks are available ₹1.5–3L and also reduce fuel costs."),
            "saving_note": f"Estimated OD saving: ~₹{saving_est:,.0f}/yr.",
            "logic": "fuel_safety feeds safety_score at 35% weight. CNG → diesel gap = 0.3 units × 3.5 pts."
        })

    if training == "No":
        saving_est = estimate_od_saving_from_composite_drop(0.7)
        od_tips.append({
            "icon": "🎓",
            "title": "Enrol drivers in a formal training programme",
            "detail": ("No driver training is currently in place. Implementing even a partial "
                       "programme raises driver_training_safety from 0.25 → 0.65, significantly "
                       "improving driver_behavior_score. Full certification pushes it to 1.0. "
                       "Government-accredited programmes available via AIMTC and state RTOs."),
            "saving_note": f"Estimated OD saving (full programme): ~₹{saving_est:,.0f}/yr.",
            "logic": "driver_training_safety: No=0.25, Partial=0.65, Yes=1.0. Weight = 25% of driver_behavior_score."
        })
    elif training == "Partial":
        saving_est = estimate_od_saving_from_composite_drop(0.3)
        od_tips.append({
            "icon": "🎓",
            "title": "Complete your driver training programme",
            "detail": ("Moving from Partial to full driver training raises driver_training_safety "
                       "from 0.65 → 1.0, further improving driver_behavior_score and composite risk."),
            "saving_note": f"Estimated OD saving: ~₹{saving_est:,.0f}/yr.",
            "logic": "Same formula. Completing training is the fastest actionable driver-score improvement."
        })

    if at_fault > 0:
        saving_est = estimate_od_saving_from_composite_drop(at_fault * 0.12)
        od_tips.append({
            "icon": "🛡️",
            "title": "Maintain a clean accident record for 3 years",
            "detail": (f"You have {at_fault} at-fault accident(s) in the last 3 years. Each "
                       f"at-fault incident adds a penalty to driver_behavior_score and accident_probability. "
                       f"A clean 3-year record removes these penalties entirely at renewal."),
            "saving_note": f"Estimated OD saving at next renewal: ~₹{saving_est:,.0f}/yr.",
            "logic": ("fault_penalty = at_fault/5, weighted 20% in driver_behavior_score. "
                      "Also feeds accident_probability directly via driver_amp.")
        })
    if viol > 2:
        saving_est = estimate_od_saving_from_composite_drop(viol * 0.04)
        od_tips.append({
            "icon": "🚦",
            "title": "Reduce traffic violations",
            "detail": (f"{viol} traffic violations in the last 12 months penalise driver_behavior_score. "
                       f"Implement speed discipline and safety checklists. Most violations drop off "
                       f"the 12-month lookback period if not repeated."),
            "saving_note": f"Estimated OD saving if violations drop to zero: ~₹{saving_est:,.0f}/yr.",
            "logic": "viol_penalty = violations/10, weighted 15% in driver_behavior_score."
        })

    if overload > 0:
        saving_est = estimate_od_saving_from_composite_drop(overload * 0.08)
        od_tips.append({
            "icon": "⚖️",
            "title": "Eliminate overloading incidents",
            "detail": (f"{overload} overloading incident(s) recorded. Overloading raises "
                       f"risk_exposure_score (13% weight) and directly increases accident_probability. "
                       f"NHAI data shows overloaded vehicles have 2× brake failure rates. "
                       f"Use weigh-bridges before departure and train loaders on GVW limits."),
            "saving_note": f"Estimated OD saving: ~₹{saving_est:,.0f}/yr.",
            "logic": "overload_norm feeds risk_exposure_score at 13% weight; also raises accident probability."
        })

    if night_pct > 60:
        saving_est = estimate_od_saving_from_composite_drop(0.4)
        od_tips.append({
            "icon": "🌙",
            "title": "Reduce night driving where operationally possible",
            "detail": (f"Currently {night_pct}% of travel is at night. MoRTH data shows 45% of "
                       f"fatal road accidents happen between 6 PM and 6 AM despite only ~30% of "
                       f"traffic being at night. Shifting even 20% of night trips to daytime "
                       f"meaningfully reduces night_travel_risk_score and accident_probability."),
            "saving_note": f"Estimated OD saving: ~₹{saving_est:,.0f}/yr.",
            "logic": "night_travel_risk_score feeds composite at 8% weight. Travel pattern also affects accident_probability."
        })

    if route in ("Hill/Mountain", "Rural/Remote"):
        saving_est = estimate_od_saving_from_composite_drop(0.5)
        od_tips.append({
            "icon": "🗺️",
            "title": "Use National Highway routes where possible",
            "detail": (f"Your primary route is {route} (risk score {sem['route_risk_map'].get(route, 5)}/10). "
                       f"Switching even a portion of trips to National Highway or Mixed routes "
                       f"lowers route_risk and risk_exposure_score. NHAI NH routes have better "
                       f"lighting, maintained surfaces, and emergency response."),
            "saving_note": f"Estimated OD saving: ~₹{saving_est:,.0f}/yr.",
            "logic": "route_risk_map: Hill=9, Rural=7, Highway=5, Mixed=5, Urban=4. Feeds risk_exposure at 22% weight."
        })

    if turnover > 40:
        saving_est = estimate_od_saving_from_composite_drop(0.2)
        od_tips.append({
            "icon": "👥",
            "title": "Reduce driver turnover rate",
            "detail": (f"Driver turnover of {turnover}% per year means frequent replacement "
                       f"with less experienced drivers. High turnover penalises driver_behavior_score. "
                       f"Better pay, rest facilities, and trip incentives reduce turnover and "
                       f"build an experienced, lower-risk driving pool."),
            "saving_note": f"Estimated OD saving: ~₹{saving_est:,.0f}/yr.",
            "logic": "turnover_penalty = turnover/50, weighted 10% in driver_behavior_score."
        })

    if claims > 0:
        claims_load_pct = uw.get("claims_loading_pct", 0)
        claims_load_amt = od_gross * (claims_load_pct / 100) / max(0.01, 1 - claims_load_pct / 100) if od_gross else 0
        general_tips.append({
            "icon": "📋",
            "title": "Build a claims-free record to eliminate claims loading",
            "detail": (f"You currently carry a claims history loading of {claims_load_pct:.1f}% on OD "
                       f"due to {claims} claim(s) in the past 3 years. This loading drops off "
                       f"automatically as claim-free years accumulate. Consider absorbing "
                       f"smaller losses (under ₹25,000–50,000) out-of-pocket to protect your "
                       f"claims record — the premium saving over 3 years typically exceeds the "
                       f"uninsured loss."),
            "saving_note": f"Claims loading currently adds ~₹{claims_load_amt:,.0f}/yr to your OD gross.",
            "logic": "claims_loading = min(10%, claims_count×1.5% + severity_factor×2%). Drops to zero with clean record."
        })

    general_tips.append({
        "icon": "🔧",
        "title": "Opt for a higher voluntary deductible",
        "detail": ("Most commercial vehicle policies allow you to opt for a higher voluntary "
                   "deductible (the amount you absorb per claim before insurance pays). "
                   "Increasing the voluntary deductible from ₹0 to ₹25,000–₹50,000 "
                   "typically reduces OD gross premium by 5–10% depending on insurer. "
                   "Best suited for fleet operators with consistent cash flow who can "
                   "self-fund small claims."),
        "saving_note": f"Indicative OD saving: ₹{od_gross * 0.06:,.0f}–₹{od_gross * 0.10:,.0f}/yr.",
        "logic": "Deductible reduces insurer's expected loss cost → passed through as premium reduction."
    })

    if cargo == "Chemical/Hazardous" and permit != "National Permit":
        general_tips.append({
            "icon": "📑",
            "title": "Obtain National Permit for Chemical/Hazardous cargo",
            "detail": ("Chemical/Hazardous cargo without a National Permit triggers automatic "
                       "REFER in underwriting, adding a 5%+ risk loading to your OD premium. "
                       "Securing the correct National Permit (and HMRT certification) removes "
                       "this flag and may move the policy to ACCEPT."),
            "saving_note": f"Risk loading of {uw.get('extra_loading_pct', 0):.1f}% currently applies to OD.",
            "logic": "Underwriting rule: Chemical/Hazardous without National Permit → REFER + mandatory 5%+ loading."
        })

    if fleet < 5:
        general_tips.append({
            "icon": "🚛",
            "title": "Add vehicles to fleet for group discount",
            "detail": ("Most commercial vehicle insurers in India offer 5–15% fleet discounts "
                       "for policies covering 5+ vehicles under a single floater or group policy. "
                       "If you plan to expand, grouping vehicles on a single policy is more "
                       "cost-effective than individual policies."),
            "saving_note": f"Indicative group discount on OD: ₹{od_gross * 0.07:,.0f}–₹{od_gross * 0.12:,.0f}/yr at 5+ vehicles.",
            "logic": "Group policies spread administrative cost; insurer passes ~50% of saving to policyholder."
        })
    elif fleet >= 5:
        general_tips.append({
            "icon": "🚛",
            "title": "Negotiate a fleet discount with your insurer",
            "detail": (f"With {fleet} vehicles, you qualify for fleet discount negotiations. "
                       f"Request a fleet discount of 8–15% on the OD component across all vehicles. "
                       f"Provide the insurer with your consolidated fleet loss run (claims history "
                       f"across all vehicles) — a good portfolio loss ratio strengthens your case."),
            "saving_note": f"Indicative fleet OD saving: ₹{od_gross * 0.08:,.0f}–₹{od_gross * 0.15:,.0f}/yr.",
            "logic": "IRDAI allows insurer discretion on OD rates; fleet scale gives negotiating leverage."
        })

    general_tips.append({
        "icon": "📡",
        "title": "Install AIS-140 compliant GPS with telematics data sharing",
        "detail": ("AIS-140 GPS tracking is mandatory for commercial vehicles in India. "
                   "Some insurers (New India Assurance, ICICI Lombard, Bajaj Allianz) offer "
                   "5–10% OD discounts for verified telematics data sharing — harsh braking, "
                   "speeding events, and geo-fence adherence are scored. Share your telematics "
                   "report at renewal to demonstrate safe driving behaviour."),
        "saving_note": f"Indicative telematics OD discount: ₹{od_gross * 0.07:,.0f}–₹{od_gross * 0.10:,.0f}/yr.",
        "logic": "Usage-based insurance (UBI) discounts telematics-verified safe drivers; industry standard globally."
    })

    general_tips.append({
        "icon": "📅",
        "title": "Consider a 3-year long-term policy",
        "detail": ("IRDAI permits 3-year bundled commercial vehicle policies. OD rates "
                   "are locked at inception, protecting against annual rate revisions. "
                   "Insurers typically offer a 5–8% discount for 3-year OD commitment. "
                   "TP is mandatorily 3 years for new vehicles under 2018 Supreme Court order."),
        "saving_note": f"Indicative 3-yr OD saving: ₹{od_gross * 0.055:,.0f}–₹{od_gross * 0.08:,.0f}/yr.",
        "logic": "Reduced admin cost + lower lapse risk → insurer shares benefit with policyholder."
    })

    if claims == 0:
        general_tips.append({
            "icon": "⭐",
            "title": "Protect your No Claim Bonus (NCB)",
            "detail": ("You have no claims this year — you are building or holding NCB. "
                       "Under IRDAI's NCB grid, 1 claim-free year = 20% OD discount, up to "
                       "50% at 5 consecutive claim-free years. Consider NCB Protect add-on "
                       "(typically ₹500–₹2,000/yr) to preserve NCB after a single minor claim."),
            "saving_note": f"Max NCB saves up to ₹{od_gross * 0.50:,.0f}/yr on OD at 5+ years claim-free.",
            "logic": "IRDAI NCB schedule: 20%-25%-35%-45%-50% for Years 1-5. NCB applies to OD only, not TP."
        })

    return {"tp": tp_tips, "od": od_tips, "general": general_tips}


# ============================================================
# PDF GENERATION ENGINE (REPORTLAB)
# ============================================================
def clean_for_pdf(text):
    """Removes emojis and cleans symbols to prevent standard font crashes in PDF."""
    text = str(text).replace('₹', 'Rs. ')
    return text.encode('ascii', 'ignore').decode()

def get_date_suffix(d):
    return 'th' if 11 <= d <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(d % 10, 'th')


class NumberedCanvas(canvas.Canvas):
    """Custom canvas wrapper to handle the 'Page X of Y' numbering reliably."""

    def __init__(self, *args, **kwargs):
        self.is_uw = kwargs.pop('is_uw', False)
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_watermark_and_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_watermark_and_footer(self, page_count):
        self.saveState()

        # Watermark: a simple vector truck + team name, rendered at 15% opacity
        # so it sits in the background without making the text unreadable.
        self.translate(300, 400)
        self.rotate(45)
        self.setFillAlpha(0.15)
        self.setFillColorRGB(0.3, 0.3, 0.3)
        self.rect(-80, 20, 100, 50, stroke=0, fill=1)  # trailer body
        self.rect(25, 20, 40, 40, stroke=0, fill=1)    # cab
        self.circle(-50, 20, 12, stroke=0, fill=1)
        self.circle(0, 20, 12, stroke=0, fill=1)
        self.circle(45, 20, 12, stroke=0, fill=1)

        self.setFont("Helvetica-Bold", 55)
        self.drawCentredString(0, -40, "SSSIA | Team 5")

        if self.is_uw:
            self.setFillColorRGB(0.8, 0.1, 0.1)  # Red tint for UW warning
            self.setFont("Helvetica-Bold", 35)
            self.drawCentredString(0, -85, "Underwriting & Actuarial Use Only")

        self.restoreState()

        # ── FOOTER ──
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.dimgray)

        # Force IST (+5:30) in the footer timestamp regardless of where the server is running
        ist = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        now = datetime.datetime.now(ist)
        now_str = f"{now.day}{get_date_suffix(now.day)} {now.strftime('%B %Y, %I:%M %p, IST')}"

        left_lines = [
            "Model- Commercial Truck Insurance Pricing & Underwriting Analysis Multi-Agent Model.",
            "Developer Team- Sri Sathya Sai Institute of Actuaries (SSSIA) | Team 5",
            f"Download on- {now_str}",
            "© All Right Reserved- SSSIA | Team 5"
        ]

        y_pos = 0.4 * inch
        for line in reversed(left_lines):
            self.drawString(0.5 * inch, y_pos, line)
            y_pos += 12

        page_num_str = f"Page No.- {self._pageNumber} of {page_count}"
        self.drawRightString(8 * inch, 0.5 * inch, page_num_str)
        self.restoreState()


def build_premium_pdf(params, result, tips, is_existing, policy_id, reg_no, is_uw=False):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=72)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=1, textColor=colors.darkblue,
                                 fontSize=18)
    heading_style = ParagraphStyle('HeadingStyle', parent=styles['Heading2'], textColor=colors.maroon, spaceAfter=10,
                                   spaceBefore=10)
    sub_style = ParagraphStyle('SubStyle', parent=styles['Heading3'], textColor=colors.darkslategray)
    normal_style = styles['Normal']

    # ── 1. TITLE & HEADER ──
    story.append(Paragraph("Single Policy Premium Quote", title_style))
    story.append(Spacer(1, 15))

    if is_existing:
        story.append(Paragraph("Single Policy Premium Quote for Existing Policyholder", heading_style))
        story.append(Paragraph(f"<b>Policy ID =</b> {policy_id}", normal_style))
        story.append(Paragraph(f"<b>Vehicle Registration Number =</b> {reg_no}", normal_style))
    else:
        story.append(Paragraph("Single Policy Premium Quote for New Policyholder", heading_style))

    story.append(Spacer(1, 20))

    # ── 2. INPUTS & PROFILE (All user-filled info) ──
    story.append(Paragraph("Inputs & Profile", heading_style))
    input_data = [["Parameter", "Value"]]
    for k, v in params.items():
        nice_key = str(k).replace('_', ' ').title()
        input_data.append([nice_key, clean_for_pdf(v)])

    input_table = Table(input_data, colWidths=[3.5 * inch, 3.5 * inch])
    input_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1A237E")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.aliceblue),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey)
    ]))
    story.append(input_table)
    story.append(Spacer(1, 20))

    # ── 3. UW, RISK SCORES & CLIMATE EXPOSURE (ONLY IF is_uw = True) ──
    if is_uw:
        story.append(PageBreak())
        story.append(Paragraph("Underwriting & Actuarial Analysis", title_style))

        # 3a. Underwriting Decision
        story.append(KeepTogether([
            Paragraph("Underwriting Decision & Loadings", heading_style),
            Table([
                ["Underwriting Decision", result['underwriting']['decision']],
                ["Risk Loading Applied", f"{result['underwriting']['extra_loading_pct']}%"],
                ["Claims Loading Applied", f"{result['underwriting']['claims_loading_pct']}%"]
            ], colWidths=[3.5 * inch, 3.5 * inch], style=TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#EFEBE9")),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
                ('TEXTCOLOR', (1, 0), (1, 0),
                 colors.darkred if result['underwriting']['decision'] != 'ACCEPT' else colors.darkgreen)
            ]))
        ]))
        story.append(Spacer(1, 15))

        # 3b. Risk Score Exposure
        rs = result['risk_scores']
        story.append(KeepTogether([
            Paragraph("Risk Score Exposure", heading_style),
            Table([
                ["Metric", "Score (out of 10)"],
                ["Cargo Risk", f"{rs['cargo_risk']:.2f}"],
                ["Risk Exposure", f"{rs['risk_exposure']:.2f}"],
                ["Safety Score", f"{rs['safety_score']:.2f}"],
                ["Driver Behavior Score", f"{rs['driver_behavior_score']:.2f}"],
                ["Night Travel Risk", f"{rs['night_travel_risk']:.2f}"],
                ["Accident Probability", f"{rs['accident_probability']:.2f}"],
                ["Composite Risk Score", f"{rs['composite_risk']:.2f}"]
            ], colWidths=[3.5 * inch, 3.5 * inch], style=TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#424242")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
            ]))
        ]))
        story.append(Spacer(1, 15))

        # 3c. Climate & Route Exposure
        cli = result['climate']
        story.append(KeepTogether([
            Paragraph("Climate & Route Exposure", heading_style),
            Table([
                ["Metric", "Value"],
                ["Climate Zone", cli['climate_zone']],
                ["Flood Risk", f"{cli['flood']:.1f} / 10"],
                ["Fog Risk", f"{cli['fog']:.1f} / 10"],
                ["Cyclone Risk", f"{cli['cyclone']:.1f} / 10"],
                ["Route Risk", f"{cli['route_risk']:.1f} / 10"],
                ["Climate Composite Score", f"{cli['climate_composite']:.2f} / 10"]
            ], colWidths=[3.5 * inch, 3.5 * inch], style=TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0277BD")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
            ]))
        ]))
        story.append(PageBreak())

        # ── 4. PREMIUM SUMMARY ──
    story.append(KeepTogether([
        Paragraph("Premium Summary", heading_style),
        Table([
            ["Total Gross Premium (Annual)", f"Rs. {result['pricing']['total_gross_premium']:,.0f}"],
            ["Total Pure Premium (Actuarial Loss Cost)", f"Rs. {result['pricing']['total_pure_premium']:,.0f}"],
            ["NCB Saving (Annual)", f"Rs. {result['pricing'].get('ncb_saving', 0):,.0f}"],
            ["Policy Limit (IDV)", f"Rs. {result['underwriting']['policy_limit']:,.0f}"]
        ], colWidths=[4.5 * inch, 2.5 * inch], style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2E7D32")),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#E8F5E9")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('PADDING', (0, 0), (-1, -1), 8)
        ]))
    ]))
    story.append(Spacer(1, 20))

    # ── 5. PREMIUM BREAKDOWN ──
    story.append(KeepTogether([
        Paragraph("Premium Breakdown - TP & OD (Up-to NCB Applied)", heading_style),
        Table([
            ["Component", "Amount"],
            ["Third Party (TP) Pure Premium", f"Rs. {result['pricing']['tp_pure_premium_compulsory']:,.0f}"],
            ["Third Party (TP) Gross Premium", f"Rs. {result['pricing']['tp_gross_premium_compulsory']:,.0f}"],
            ["Own Damage (OD) Pure Premium", f"Rs. {result['pricing']['od_pure_premium_optional']:,.0f}"],
            ["Own Damage (OD) Gross Premium (After NCB)", f"Rs. {result['pricing']['od_gross_premium_optional']:,.0f}"]
        ], colWidths=[4.5 * inch, 2.5 * inch], style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E65100")),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#FFF3E0")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold')
        ]))
    ]))

    # ── 5.5 PAPER VERIFIABLE PRICING STACK (ONLY IF is_uw = True) ──
    if is_uw:
        story.append(Spacer(1, 15))
        story.append(Paragraph("Paper-Verifiable Pricing Stack", heading_style))

        uw_data = result['underwriting']
        pr = result['pricing']

        denom_val = (1 - uw_data['expenses_loading_pct'] / 100 - uw_data['profit_margin_pct'] / 100
                     - uw_data.get('other_loading_pct', 0) / 100
                     - uw_data['extra_loading_pct'] / 100 - uw_data['claims_loading_pct'] / 100)

        stack_data = [
            ["Step 1: TP Premium (Compulsory)", ""],
            ["GVW-Slab IRDAI Tariff (Gross)", f"Rs. {pr['tp_gross_premium_compulsory']:,.0f}"],
            ["TP Pure Premium (70% of Gross)", f"Rs. {pr['tp_pure_premium_compulsory']:,.0f}"],

            ["Step 2: OD Premium (Optional, Pre-NCB)", ""],
            ["OD Pure Premium", f"Rs. {pr['od_pure_premium_optional']:,.0f}"],
            ["Expenses Loading", f"{uw_data['expenses_loading_pct']:.0f}%"],
            ["Profit Margin", f"{uw_data['profit_margin_pct']:.0f}%"]
        ]

        # Only add Other Loading row if it's > 0
        if uw_data.get('other_loading_pct', 0) > 0:
            stack_data.append(["Other Loading", f"{uw_data['other_loading_pct']:.0f}%"])

        stack_data.extend([
            ["Underwriting Risk Loading", f"{uw_data['extra_loading_pct']:.2f}%"],
            ["Claims History Loading", f"{uw_data['claims_loading_pct']:.2f}%"],
            ["Expense Denominator", f"{denom_val:.4f}"],
            ["OD Gross (Pre-NCB)", f"Rs. {pr['od_gross_pre_ncb']:,.0f}"],

            ["Step 3: NCB Discount (OD Only)", ""],
            [f"NCB Tier {pr['ncb_tier']} Discount", f"{pr['ncb_discount_pct']:.0f}%"],
            ["NCB Saving", f"- Rs. {pr.get('ncb_saving', 0):,.0f}"],
            ["OD Gross (After NCB)", f"Rs. {pr['od_gross_premium_optional']:,.0f}"],

            ["Step 4: Final Total", ""],
            ["Total Gross Premium", f"Rs. {pr['total_gross_premium']:,.0f}"]
        ])

        # Style the calculation stack table
        ts = [
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('PADDING', (0, 0), (-1, -1), 6)
        ]

        for i, row in enumerate(stack_data):
            if row[0].startswith("Step"):
                ts.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor("#D7CCC8")))
                ts.append(('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold'))
                ts.append(('SPAN', (0, i), (1, i)))
                ts.append(('ALIGN', (0, i), (1, i), 'LEFT'))
            if row[0] in ["Expense Denominator", "OD Gross (Pre-NCB)", "OD Gross (After NCB)", "Total Gross Premium"]:
                ts.append(('FONTNAME', (0, i), (-1, i), 'Helvetica-Bold'))
            if row[0] == "Total Gross Premium":
                ts.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor("#E8F5E9")))

        stack_table = Table(stack_data, colWidths=[4.5 * inch, 2.5 * inch])
        stack_table.setStyle(TableStyle(ts))

        story.append(KeepTogether([stack_table]))

    # ── 6. PREMIUM REDUCTION TIPS (EACH ON NEW PAGE) ──
    sections = [
        ("Third Party (TP) Tips", tips['tp']),
        ("Own Damage (OD) Tips", tips['od']),
        ("General / Structural Tips", tips['general'])
    ]

    for title, tip_list in sections:
        story.append(PageBreak())
        story.append(Paragraph("Single Policy Premium Quote", title_style))
        story.append(Paragraph("How to Reduce Your Premium", heading_style))
        story.append(Paragraph(title, sub_style))
        story.append(Spacer(1, 10))

        if not tip_list:
            story.append(Paragraph("No tips available for this section based on your current profile.", normal_style))

        for tip in tip_list:
            story.append(Paragraph(f"<b>{clean_for_pdf(tip['title'])}</b>", normal_style))
            story.append(Paragraph(clean_for_pdf(tip['detail']), normal_style))
            story.append(Paragraph(f"<i>Saving Note:</i> {clean_for_pdf(tip['saving_note'])}", normal_style))
            story.append(Spacer(1, 12))

    doc.build(story, canvasmaker=lambda *args, **kwargs: NumberedCanvas(*args, is_uw=is_uw, **kwargs))
    return buffer.getvalue()

def build_summary_stats_pdf(df, dataset_stats):
    """Generates an executive Summary Statistics Report for Underwriters & Actuaries."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=72)
    story = []
    styles = getSampleStyleSheet()

    # Custom Styles
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=1,
                                 textColor=colors.HexColor("#0D47A1"), fontSize=22, spaceAfter=20)
    heading1_style = ParagraphStyle('Heading1Style', parent=styles['Heading1'], textColor=colors.HexColor("#1565C0"),
                                    spaceAfter=15, spaceBefore=20)
    heading2_style = ParagraphStyle('Heading2Style', parent=styles['Heading2'], textColor=colors.HexColor("#B71C1C"),
                                    spaceAfter=10, spaceBefore=15)
    normal_style = ParagraphStyle('NormalStyle', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=10)
    commentary_style = ParagraphStyle('CommentaryStyle', parent=styles['Normal'], fontSize=10, leading=14,
                                      spaceAfter=15, textColor=colors.darkslategrey, leftIndent=20, rightIndent=20)

    # ==========================================
    # PAGE 1: COVER & EXECUTIVE SUMMARY
    # ==========================================
    story.append(Paragraph("Summary Statistic For Underwriters & Actuaries", title_style))
    story.append(Spacer(1, 20))

    story.append(Paragraph("1. Executive Portfolio Overview", heading1_style))
    story.append(Paragraph(
        "This document serves as the primary actuarial summary for the current commercial vehicle portfolio. It aggregates pricing adequate, risk concentration, underwriting funnel metrics, and deep-dive cross-tabulations across multiple exposure vectors. The data represents the fully scored output from the 5-Stage TruckShield Engine.",
        normal_style))

    total_records = len(df)
    accepted_df = df[df["uw_decision"] != "DECLINE"]

    acc_ct = (df['uw_decision'] == 'ACCEPT').sum()
    ref_ct = (df['uw_decision'] == 'REFER').sum()
    dec_ct = (df['uw_decision'] == 'DECLINE').sum()

    metrics_data = [
        ["Key Performance Indicator (KPI)", "Portfolio Value"],
        ["Total Policies Evaluated", f"{total_records:,}"],
        ["Total Active Exposure (IDV Sum)", f"Rs. {accepted_df['idv_insured_declared_value'].sum():,.0f}"],
        ["Total Gross Written Premium (GWP)", f"Rs. {accepted_df['total_gross_premium'].sum():,.0f}"],
        ["Average Premium Per Policy", f"Rs. {accepted_df['total_gross_premium'].mean():,.0f}"],
        ["Average Third-Party (TP) Portion", f"Rs. {accepted_df['tp_gross_premium_compulsory'].mean():,.0f}"],
        ["Average Own-Damage (OD) Portion", f"Rs. {accepted_df['od_gross_premium_optional'].mean():,.0f}"],
        ["Portfolio Average Risk Score", f"{accepted_df['composite_risk_score'].mean():.2f} / 10.0"],
        ["Overall Claims Frequency (Implied)", f"{accepted_df.get('claim_frequency', pd.Series([0])).mean():.4f}"]
    ]

    m_table = Table(metrics_data, colWidths=[4 * inch, 3 * inch])
    m_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0D47A1")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#F5F5F5")),
        ('PADDING', (0, 0), (-1, -1), 8)
    ]))
    story.append(m_table)

    # Underwriting Funnel Chart
    story.append(Paragraph("Underwriting Funnel Distribution", heading2_style))

    # Draw Native Pie Chart
    d = Drawing(400, 160)
    pc = Pie()
    pc.x = 150
    pc.y = 10
    pc.width = 140
    pc.height = 140
    pc.data = [acc_ct, ref_ct, dec_ct]
    pc.labels = [f"Accept ({acc_ct})", f"Refer ({ref_ct})", f"Decline ({dec_ct})"]
    pc.slices[0].fillColor = colors.HexColor("#2E7D32")  # Green
    pc.slices[1].fillColor = colors.HexColor("#F9A825")  # Yellow
    pc.slices[2].fillColor = colors.HexColor("#C62828")  # Red
    d.add(pc)
    story.append(d)

    story.append(PageBreak())

    # ==========================================
    # PAGE 2: PREMIUM COMPOSITION & DISTRIBUTION
    # ==========================================
    story.append(Paragraph("2. Premium & Risk Composition", heading1_style))

    # Premium Split Bar Chart (Top 5 States)
    if 'state' in accepted_df.columns:
        story.append(Paragraph("Average Premium by Top 5 States (TP vs OD)", heading2_style))
        state_grp = accepted_df.groupby("state")[
            ["tp_gross_premium_compulsory", "od_gross_premium_optional"]].mean().sort_values(
            "od_gross_premium_optional", ascending=False).head(5)

        d2 = Drawing(400, 200)
        bc = VerticalBarChart()
        bc.x = 50
        bc.y = 50
        bc.height = 125
        bc.width = 350
        bc.data = [
            tuple(state_grp["tp_gross_premium_compulsory"].tolist()),
            tuple(state_grp["od_gross_premium_optional"].tolist())
        ]
        bc.categoryAxis.categoryNames = [str(x)[:10] for x in state_grp.index.tolist()]
        bc.bars[0].fillColor = colors.HexColor("#0277BD")  # TP Blue
        bc.bars[1].fillColor = colors.HexColor("#E65100")  # OD Orange
        bc.valueAxis.valueMin = 0
        d2.add(bc)
        story.append(d2)

        story.append(Paragraph("<i>Chart Legend: Blue = Third Party (Fixed), Orange = Own Damage (Risk-Adjusted)</i>",
                               normal_style))
        story.append(Spacer(1, 20))

    # ==========================================
    # PAGES 3-10: DEEP-DIVE STRATIFICATION MATRICES
    # ==========================================
    story.append(Paragraph("3. Deep-Dive Exposure Stratification", heading1_style))
    story.append(Paragraph(
        "The following tables break down the portfolio across multiple actuarial dimensions. These matrices are critical for identifying risk concentrations, assessing pricing adequacy across segments, and guiding treaty reinsurance placement.",
        normal_style))

    # List of categories to generate full-page matrix reports for
    analysis_vectors = [
        ("vehicle_category", "Vehicle Class"),
        ("goods_category", "Cargo & Goods Carried"),
        ("route_type", "Route Typology"),
        ("climate_zone", "Environmental & Climate Zone"),
        ("driver_training_program", "Driver Training Engagement"),
        ("fatigue_monitoring_system", "Telematics & Fatigue Monitoring"),
        ("state", "Geographic Territory (State)"),
        ("travel_time_pattern", "Travel Time Pattern (Night vs Day)")
    ]

    for col_name, display_name in analysis_vectors:
        if col_name in accepted_df.columns:
            story.append(PageBreak())  # Start each major matrix on a new page
            story.append(Paragraph(f"Analysis Vector: {display_name}", heading2_style))

            # Generate Commentary
            story.append(Paragraph(
                f"<b>Actuarial Commentary:</b> This table illustrates the exposure and pricing adequacy segmented by {display_name}. Underwriters should monitor segments where the 'Avg Composite Risk' deviates significantly from the portfolio mean, ensuring that the 'Mean OD Gross' scales proportionately to cover the heightened loss propensity.",
                commentary_style))

            # Grouping Data
            grp = accepted_df.groupby(col_name).agg(
                count=("policy_id", "count"),
                avg_idv=("idv_insured_declared_value", "mean"),
                avg_risk=("composite_risk_score", "mean"),
                mean_tp=("tp_gross_premium_compulsory", "mean"),
                mean_od=("od_gross_premium_optional", "mean"),
                mean_total=("total_gross_premium", "mean")
            ).reset_index().sort_values("avg_risk", ascending=False)

            # Table Header
            table_data = [[
                display_name,
                "Units",
                "Avg IDV (Exposure)",
                "Avg Risk Score",
                "Mean OD Prem",
                "Mean Total Prem"
            ]]

            # Table Rows
            for _, r in grp.iterrows():
                table_data.append([
                    str(r[col_name])[:25],  # Truncate long names
                    f"{int(r['count'])}",
                    f"Rs. {r['avg_idv'] / 100000:,.1f}L",  # Represent IDV in Lakhs to save space
                    f"{r['avg_risk']:.2f}",
                    f"Rs. {r['mean_od']:,.0f}",
                    f"Rs. {r['mean_total']:,.0f}"
                ])

            seg_table = Table(table_data,
                              colWidths=[2.0 * inch, 0.6 * inch, 1.2 * inch, 1.0 * inch, 1.2 * inch, 1.2 * inch])
            seg_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#37474F")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#ECEFF1")),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#E3F2FD")]),
                # Alternating rows
                ('PADDING', (0, 0), (-1, -1), 6)
            ]))
            story.append(seg_table)

    # ==========================================
    # FINAL PAGE: UW DECISION MATRIX
    # ==========================================
    story.append(PageBreak())
    story.append(Paragraph("4. Underwriting & Loading Matrix", heading1_style))
    story.append(Paragraph(
        "Overview of policies flagged for referral or decline, alongside the average risk loadings applied to the Own-Damage component.",
        normal_style))

    uw_grp = df.groupby("uw_decision").agg(
        count=("policy_id", "count"),
        avg_risk=("composite_risk_score", "mean"),
        avg_uw_load=("uw_extra_loading", "mean"),
        avg_claim_load=("uw_claims_loading", "mean")
    ).reset_index()

    uw_data = [["UW Decision", "Total Units", "Avg Risk Score", "Avg UW Risk Load", "Avg Claims Load"]]
    for _, r in uw_grp.iterrows():
        uw_data.append([
            str(r["uw_decision"]),
            f"{int(r['count'])}",
            f"{r['avg_risk']:.2f} / 10",
            f"{r['avg_uw_load'] * 100:.1f}%",
            f"{r['avg_claim_load'] * 100:.1f}%"
        ])

    uw_table = Table(uw_data, colWidths=[1.5 * inch, 1.2 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch])
    uw_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#424242")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('PADDING', (0, 0), (-1, -1), 8)
    ]))
    story.append(uw_table)

    # Compile PDF
    doc.build(story, canvasmaker=lambda *args, **kwargs: NumberedCanvas(*args, is_uw=True, **kwargs))
    return buffer.getvalue()

def build_chat_history_pdf(chat_history):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=72)
    story = []
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], alignment=1,
                                 textColor=colors.HexColor("#0D47A1"), fontSize=20, spaceAfter=20)
    user_label_style = ParagraphStyle('UserLabel', fontName="Helvetica-Bold", fontSize=10,
                                      textColor=colors.HexColor("#0D47A1"), spaceAfter=2)
    asst_label_style = ParagraphStyle('AsstLabel', fontName="Helvetica-Bold", fontSize=10,
                                      textColor=colors.HexColor("#2E7D32"), spaceAfter=2)
    msg_style = ParagraphStyle('MsgStyle', fontName="Helvetica", fontSize=10, leading=14)

    # ── 1. MAIN TITLE & HEADER ──
    story.append(Paragraph("Chat With TruckShield AI", title_style))
    story.append(Paragraph("Official Conversation Transcript & Assistant Consultative Ledger", styles['Normal']))
    story.append(Spacer(1, 15))

    if not chat_history:
        story.append(Paragraph("<i>No prior conversation history found in this session.</i>", styles['Normal']))
    else:
        # ── 2. LOOP THROUGH LIVE MESSAGES ──
        for msg in chat_history:
            role = msg["role"]
            content = clean_for_pdf(msg["content"])

            # Format text blocks with soft background shades (Blue for user, Green for assistant)
            if role == "user":
                cell_content = [
                    Paragraph("👤 USER", user_label_style),
                    Paragraph(content, msg_style)
                ]
                bg_color = colors.HexColor("#E3F2FD")  # Soft Blue
                border_color = colors.HexColor("#90CAF9")
            else:
                cell_content = [
                    Paragraph("🤖 TRUCKSHIELD ASSISTANT", asst_label_style),
                    Paragraph(content, msg_style)
                ]
                bg_color = colors.HexColor("#E8F5E9")  # Soft Green
                border_color = colors.HexColor("#A5D6A7")

            # Render message block inside a safe multi-line padding box
            msg_table = Table([[cell_content]], colWidths=[7.0 * inch])
            msg_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), bg_color),
                ('BOX', (0, 0), (-1, -1), 1, border_color),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))

            # Wrap in a KeepTogether to prevent single chat blocks from cutting in half across pages
            story.append(KeepTogether([msg_table]))
            story.append(Spacer(1, 12))

    # Compile the final chat file using the existing watermark engine
    doc.build(story, canvasmaker=lambda *args, **kwargs: NumberedCanvas(*args, is_uw=False, **kwargs))
    return buffer.getvalue()

class TruckShieldEngine:
    def __init__(self):
        self.validation_log   = []
        self.error_row_count  = 0
        self.validation_detail_df = None
        self.full_export_df   = None
        self.excluded_row_count = 0
        self.validated_df     = None
        self.trained_models   = None
        self.mean_raw_pure_premium = None
        self.encoders         = None
        self.feature_cols     = None
        self.model_metrics    = {}
        self.climate_df       = None
        self.risk_df          = None
        self.underwriting_df  = None
        self.pricing_df       = None
        self.processed_df     = None
        self.dataset_stats    = None
        self.ready             = False

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║           AGENT 1 — DATA VALIDATION AGENT                       ║
    # ║  Checks every row for missing values, out-of-range numerics,    ║
    # ║  invalid category values, date-logic errors, and travel-        ║
    # ║  frequency consistency. Flags bad rows; clean rows proceed.     ║
    # ╚══════════════════════════════════════════════════════════════════╝
    def validate(self, df: pd.DataFrame):
        log = []
        df  = normalise_columns(df.copy())

        defaults = {
            "fatigue_monitoring_system": "Not Installed",
            "overloading_incidents":     0,
            "climate_zone":              "Standard Low Risk",
            "average_driver_experience": 8.0,
            "driver_turnover_rate":      25.0,
            "at_fault_accidents":        0,
            "traffic_violations":        0,
            "driver_training_program":   "No",
            "top_destination_1":         "",
            "top_destination_2":         "",
            "top_destination_3":         "",
            "total_claims_count":        0,
            "total_claim_amount_paid":   0,
            "last_claim_date":           None,
            "night_travel_frequency":    50,
            "day_travel_frequency":      50,
            "travel_time_pattern":       "Mixed (Day & Night)",
        }
        for col, val in defaults.items():
            if col not in df.columns:
                df[col] = val

        error_rows  = set()
        error_cells = []   

        def _flag(idx_list, col, reason):
            for r in idx_list:
                error_rows.add(r)
                error_cells.append({"row": int(r) + 1, "column": col, "issue": reason})

        nullable_ok_cols = {"last_claim_date"}

        null_check_cols = [c for c in df.columns if c not in nullable_ok_cols]
        rows_with_nulls = df.index[df[null_check_cols].isnull().any(axis=1)].tolist()
        for col in null_check_cols:
            bad = df.index[df[col].isnull()].tolist()
            if bad:
                _flag(bad, col, "missing value")
        if rows_with_nulls:
            log.append(f"MISSING VALUES in {len(rows_with_nulls)} rows.")

        for col, (lo, hi) in SEM["valid_ranges"].items():
            if col in df.columns:
                bad = df.index[df[col].notna() & ~df[col].between(lo, hi)].tolist()
                if bad:
                    _flag(bad, col, f"out of valid range ({lo}-{hi})")
                    log.append(f"OUT-OF-RANGE [{col}]: {len(bad)} rows")

        valid_cats = {
            "vehicle_category":        SEM["vehicle_categories"],
            "fuel_type":               SEM["fuel_types"],
            "bs_emission_norm":        SEM["bs_norms"],
            "permit_type":             SEM["permit_types"],
            "ownership_type":          SEM["ownership_types"],
            "route_type":              SEM["route_types"],
            "highway_usage":           SEM["highway_usages"],
            "fatigue_monitoring_system": SEM["fatigue_monitor_options"],
            "driver_training_program": SEM["driver_training_options"],
            "travel_time_pattern":     SEM["travel_patterns"],
        }
        for col, vals in valid_cats.items():
            if col in df.columns:
                bad = df.index[df[col].notna() & ~df[col].isin(vals)].tolist()
                if bad:
                    _flag(bad, col, "invalid category value")
                    log.append(f"INVALID CATEGORY [{col}]: {len(bad)} rows")

        if "expiry_date" in df.columns and "inception_date" in df.columns:
            bad = df.index[df["expiry_date"].notna() & df["inception_date"].notna() &
                            (df["expiry_date"] <= df["inception_date"])].tolist()
            if bad:
                _flag(bad, "expiry_date", "expiry_date is not after inception_date")
                log.append(f"DATE LOGIC ERRORS: {len(bad)} rows")

        if "night_travel_frequency" in df.columns and "day_travel_frequency" in df.columns:
            pct_sum = df["night_travel_frequency"] + df["day_travel_frequency"]
            bad = df.index[pct_sum.notna() & ~pct_sum.between(95, 105)].tolist()
            if bad:
                _flag(bad, "night_travel_frequency", "night % + day % travel frequency does not sum to ~100")
                _flag(bad, "day_travel_frequency",   "night % + day % travel frequency does not sum to ~100")
                log.append(f"TRAVEL FREQUENCY SUM ≠ 100: {len(bad)} rows")

        log.append(f"SUMMARY: {len(error_rows)} flagged row(s) out of {len(df)} total rows.")

        if error_cells:
            detail_df = (pd.DataFrame(error_cells)
                         .drop_duplicates()
                         .sort_values(["row", "column"])
                         .reset_index(drop=True))
        else:
            detail_df = pd.DataFrame(columns=["row", "column", "issue"])

        self.validation_log       = log
        self.error_row_count      = len(error_rows)
        self.validation_detail_df = detail_df

        return df, error_rows, detail_df, log

    def train_and_process(self, df_full: pd.DataFrame, error_rows: set, fast_mode: bool = True):
        error_rows = set(error_rows or [])
        df = df_full.drop(index=list(error_rows)) if error_rows else df_full.copy()

        df_clean = df.dropna(subset=["vehicle_age_yrs", "goods_category", "route_type",
                                      "gross_vehicle_weight_kg", "idv_insured_declared_value"]).copy()
        rng = np.random.default_rng(42)

        if "total_claims_count" in df_clean.columns and df_clean["total_claims_count"].sum() > 0:
            portfolio_mean_freq = (df_clean["total_claims_count"].astype(float) / 3.0).mean()
            k = 2.0  
            observed_freq = df_clean["total_claims_count"].astype(float) / 3.0
            credibility = df_clean["total_claims_count"].astype(float) / (df_clean["total_claims_count"].astype(float) + k)
            df_clean["claim_frequency"] = (credibility * observed_freq
                                            + (1 - credibility) * portfolio_mean_freq)
        else:
            cargo_factor = df_clean["goods_category"].map(SEM["cargo_risk_map"]).fillna(4.0) / 5.0
            route_factor = df_clean["route_type"].map(SEM["route_risk_map"]).fillna(5.0) / 5.0
            age_factor   = 1 + df_clean["vehicle_age_yrs"] * 0.04
            gvw_factor   = 1 + df_clean["gross_vehicle_weight_kg"] / 50000
            lambda_      = 0.08 * age_factor * gvw_factor * cargo_factor * route_factor
            df_clean["claim_frequency"] = rng.poisson(lambda_).astype(float)

        if "total_claim_amount_paid" in df_clean.columns and df_clean["total_claim_amount_paid"].sum() > 0:
            had_claim = df_clean["total_claims_count"] > 0
            df_clean["claim_severity"] = np.nan
            df_clean.loc[had_claim, "claim_severity"] = (
                df_clean.loc[had_claim, "total_claim_amount_paid"]
                / df_clean.loc[had_claim, "total_claims_count"]
            ).clip(5000)
            portfolio_mean_sev = df_clean.loc[had_claim, "claim_severity"].mean()
            df_clean["claim_severity"] = df_clean["claim_severity"].fillna(portfolio_mean_sev)
        else:
            idv_factor = df_clean["idv_insured_declared_value"] / df_clean["idv_insured_declared_value"].median()
            gvw_sev    = df_clean["gross_vehicle_weight_kg"] / df_clean["gross_vehicle_weight_kg"].median()
            cargo_factor = df_clean["goods_category"].map(SEM["cargo_risk_map"]).fillna(4.0) / 5.0
            mu_sev     = 85000 * idv_factor * gvw_sev * cargo_factor
            df_clean["claim_severity"] = rng.gamma(shape=3, scale=mu_sev / 3).clip(5000)

        encoders = {}
        for col in SEM["cat_cols"]:
            if col not in df_clean.columns:
                df_clean[col] = "Unknown"
            le = LabelEncoder()
            df_clean[col + "_enc"] = le.fit_transform(df_clean[col].astype(str))
            encoders[col] = le

        numeric_base = [
            "vehicle_age_yrs", "gross_vehicle_weight_kg", "idv_insured_declared_value",
            "fleet_size", "years_in_business",
        ]
        numeric_new = [
            "overloading_incidents", "average_driver_experience", "driver_turnover_rate",
            "at_fault_accidents", "traffic_violations",
            "night_travel_frequency",
        ]
        numeric_new = [c for c in numeric_new if c in df_clean.columns]

        feature_cols = numeric_base + numeric_new + [c + "_enc" for c in SEM["cat_cols"]]
        feature_cols = [c for c in feature_cols if c in df_clean.columns]

        X = df_clean[feature_cols].fillna(0)
        X_tr, X_te, yf_tr, yf_te, ys_tr, ys_te = train_test_split(
            X, df_clean["claim_frequency"], df_clean["claim_severity"],
            test_size=0.2, random_state=42)

        n_est = 80 if fast_mode else 200
        freq_model = XGBRegressor(objective="count:poisson", n_estimators=n_est, learning_rate=0.08,
                                   max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        freq_model.fit(X_tr, yf_tr)

        sev_model = XGBRegressor(objective="reg:gamma", n_estimators=n_est, learning_rate=0.08,
                                  max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
        sev_model.fit(X_tr, ys_tr)

        self.model_metrics  = {
            "freq_mae": mean_absolute_error(yf_te, freq_model.predict(X_te)),
            "sev_mae":  mean_absolute_error(ys_te, sev_model.predict(X_te)),
        }
        self.trained_models = {"frequency": freq_model, "severity": sev_model}
        self.encoders       = encoders
        self.feature_cols   = feature_cols
        self.validated_df   = df_clean

        raw_pure_all = freq_model.predict(X) * sev_model.predict(X)
        self.mean_raw_pure_premium = float(np.mean(raw_pure_all))

        df2 = self.run_climate(df_clean)
        df3 = self.run_risk_scoring(df2)
        df4 = self.run_underwriting(df3)
        df5 = self.run_pricing(df4)

        df5 = df5.copy()
        df5["data_validation_status"] = "PROCESSED"
        if error_rows:
            excluded_idx = [i for i in df_full.index if i in error_rows]
            excluded_df  = df_full.loc[excluded_idx].copy()
            excluded_df["data_validation_status"] = "EXCLUDED — wrong data type / missing value (see Data Validation Log)"
            full_export = pd.concat([df5, excluded_df], axis=0, sort=False)
            full_export = full_export.reindex(df_full.index)  
        else:
            full_export = df5.copy()

        self.full_export_df    = full_export
        self.excluded_row_count = len(error_rows)
        self.ready = True

        return df5

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║           AGENT 2 — CLIMATE RISK AGENT                          ║
    # ║  Calculates Flood, Fog, and Cyclone scores per state + route    ║
    # ║  + climate zone. Computes Route Risk Score (route type,         ║
    # ║  highway usage, multi-destination) and produces a Climate       ║
    # ║  Composite score fed into Risk Scoring.                         ║
    # ╚══════════════════════════════════════════════════════════════════╝
    def run_climate(self, df: pd.DataFrame) -> pd.DataFrame:
        base     = SEM["climate_risk_base"]
        zone_mod = SEM["climate_zone_modifiers"]
        df_c     = df.copy()

        flood_s, fog_s, cyclone_s, route_risk_s = [], [], [], []
        for _, row in df_c.iterrows():
            state      = row.get("state", "DEFAULT")
            route      = row.get("route_type", "Mixed")
            climate_zn = row.get("climate_zone", "Standard Low Risk")
            hw         = row.get("highway_usage", "Mixed NH+SH")
            d1         = str(row.get("top_destination_1", ""))
            d2         = str(row.get("top_destination_2", ""))
            d3         = str(row.get("top_destination_3", ""))

            p = base.get(state, base["DEFAULT"])
            rc = SEM["route_climate_risk"].get(route, {"flood_amp": 1.0, "cyclone_amp": 1.0})
            zm = zone_mod.get(climate_zn, zone_mod["Standard Low Risk"])

            flood   = min(10, round(p["flood"]   * rc["flood_amp"]   * zm["flood"],   2))
            fog     = min(10, round(p["fog"]      * zm["fog"],                         2))
            cyclone = min(10, round(p["cyclone"]  * rc["cyclone_amp"] * zm["cyclone"], 2))
            rr = compute_route_risk_score(route, hw, d1, d2, d3)

            flood_s.append(flood)
            fog_s.append(fog)
            cyclone_s.append(cyclone)
            route_risk_s.append(rr)

        df_c["climate_flood_score"]   = flood_s
        df_c["climate_fog_score"]     = fog_s
        df_c["climate_cyclone_score"] = cyclone_s
        df_c["route_risk_score"]      = route_risk_s  

        df_c["climate_composite"] = (
            0.38 * df_c["climate_flood_score"] +
            0.22 * df_c["climate_fog_score"]   +
            0.25 * df_c["climate_cyclone_score"] +
            0.15 * df_c["route_risk_score"]
        ).round(2)

        self.climate_df = df_c
        return df_c

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║           AGENT 3 — RISK SCORING AGENT                          ║
    # ║  Builds five sub-scores: Cargo Risk, Risk Exposure (GVW +       ║
    # ║  age + route + overloading), Safety (BS norm + fuel + ADAS),   ║
    # ║  Driver Behavior, and Night Travel Risk. Combines them with     ║
    # ║  Climate outputs into a single Composite Risk Score (0–10)      ║
    # ║  and an Accident Probability via the additive NCRB/MoRTH model.║
    # ╚══════════════════════════════════════════════════════════════════╝
    def run_risk_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        df_r = df.copy()

        df_r["cargo_risk_score"] = df_r["goods_category"].map(SEM["cargo_risk_map"]).fillna(4.0)

        gvw_norm  = (df_r["gross_vehicle_weight_kg"] / 50000).clip(0, 1)
        age_norm  = (df_r["vehicle_age_yrs"] / 20).clip(0, 1)
        route_raw = df_r["route_type"].map(SEM["route_risk_map"]).fillna(5.0) / 10
        overload_norm = (df_r.get("overloading_incidents", 0) / 5).clip(0, 1)
        df_r["risk_exposure_score"] = (
            10 * (0.35 * gvw_norm + 0.30 * age_norm + 0.22 * route_raw + 0.13 * overload_norm)
        ).round(2).clip(0, 10)

        bs_sc      = df_r["bs_emission_norm"].map(SEM["bs_norm_safety"]).fillna(0.5)
        fuel_sc    = df_r["fuel_type"].map(SEM["fuel_safety"]).fillna(0.5)
        fatigue_sc = df_r.get("fatigue_monitoring_system",
                               pd.Series("Not Installed", index=df_r.index)
                               ).map(SEM["fatigue_monitor_safety"]).fillna(0.2)
        df_r["safety_score"] = (10 * (0.40 * bs_sc + 0.35 * fuel_sc + 0.25 * fatigue_sc)).round(2).clip(0, 10)

        exp_norm       = (df_r.get("average_driver_experience", 8.0) / 20).clip(0, 1)
        training_sc    = df_r.get("driver_training_program",
                                   pd.Series("No", index=df_r.index)
                                   ).map(SEM["driver_training_safety"]).fillna(0.25)
        fault_penalty  = (df_r.get("at_fault_accidents", 0) / 5).clip(0, 1)
        viol_penalty   = (df_r.get("traffic_violations", 0) / 10).clip(0, 1)
        turnover_penalty = (df_r.get("driver_turnover_rate", 25) / 50).clip(0, 1)

        df_r["driver_behavior_score"] = (
            10 * (0.30 * exp_norm + 0.25 * training_sc
                  - 0.20 * fault_penalty - 0.15 * viol_penalty
                  - 0.10 * turnover_penalty)
        ).round(2).clip(0, 10)

        night_pct = df_r.get("night_travel_frequency", 50)
        night_norm = (night_pct / 100).clip(0, 1)
        travel_sc  = df_r.get("travel_time_pattern",
                               pd.Series("Mixed (Day & Night)", index=df_r.index)
                               ).map(SEM["night_travel_risk"]).fillna(1.1)
        df_r["night_travel_risk_score"] = (10 * night_norm * (travel_sc / 1.30)).round(2).clip(0, 10)

        df_r["accident_probability"] = df_r.apply(lambda row: compute_accident_probability(
            state=row["state"], route_type=row["route_type"],
            climate_flood=row["climate_flood_score"], climate_fog=row["climate_fog_score"],
            climate_cyclone=row["climate_cyclone_score"],
            vehicle_age=row["vehicle_age_yrs"], gvw=row["gross_vehicle_weight_kg"],
            cargo_category=row["goods_category"],
            night_travel_pct=row.get("night_travel_frequency", 50),
            overloading=row.get("overloading_incidents", 0),
            at_fault=row.get("at_fault_accidents", 0),
            traffic_viol=row.get("traffic_violations", 0),
            travel_pattern=row.get("travel_time_pattern", "Mixed (Day & Night)"),
        ), axis=1)

        df_r["composite_risk_score"] = (
            0.20 * df_r["cargo_risk_score"]
            + 0.20 * df_r["risk_exposure_score"]
            + 0.15 * (10 - df_r["safety_score"])
            + 0.15 * (10 - df_r["driver_behavior_score"])
            + 0.12 * df_r["accident_probability"]
            + 0.08 * df_r["night_travel_risk_score"]
            + 0.04 * df_r["climate_flood_score"]
            + 0.03 * df_r["climate_fog_score"]
            + 0.03 * df_r["climate_cyclone_score"]
        ).round(2).clip(0, 10)

        self.risk_df = df_r
        return df_r

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║           AGENT 4 — UNDERWRITING AGENT                          ║
    # ║  Makes Accept / Refer / Decline decisions based on Composite    ║
    # ║  Risk Score, vehicle age, cargo type, and permit status.        ║
    # ║  Applies risk loading and claims-history loading to the OD      ║
    # ║  premium, and sets the Policy Limit (= IDV).                    ║
    # ╚══════════════════════════════════════════════════════════════════╝
    def run_underwriting(self, df: pd.DataFrame) -> pd.DataFrame:
        df_u = df.copy()
        decisions, loadings, claim_loadings = [], [], []

        for _, row in df_u.iterrows():
            cr    = row["composite_risk_score"]
            age   = row["vehicle_age_yrs"]
            cargo = row["goods_category"]
            permit = row["permit_type"]

            claims_ct = row.get("total_claims_count", 0)
            claim_amt = row.get("total_claim_amount_paid", 0)
            severity_factor = min(claim_amt / 500000, 1.0) if claim_amt > 0 else 0
            claims_load = round(min(0.10, claims_ct * 0.015 + severity_factor * 0.02), 3)

            if cr > 8.5 and age > 17:
                decision, loading = "DECLINE", 0
            elif cr > 7.0 or (cargo == "Chemical/Hazardous" and permit != "National Permit"):
                decision = "REFER"
                loading  = round(0.05 + (cr - 7.0) * 0.015, 3)
            else:
                decision = "ACCEPT"
                loading  = round(max(0, (cr - 4.0) * 0.01), 3)

            decisions.append(decision)
            loadings.append(loading)
            claim_loadings.append(claims_load)

        df_u["uw_decision"]       = decisions
        df_u["uw_extra_loading"]  = loadings
        df_u["uw_claims_loading"] = claim_loadings   
        df_u["policy_limit"]      = (df_u["idv_insured_declared_value"] * SEM["policy_limit_pct_of_idv"]).round(0)

        self.underwriting_df = df_u
        return df_u

    # ╔══════════════════════════════════════════════════════════════════╗
    # ║           AGENT 5 — PRICING AGENT                               ║
    # ║  Uses XGBoost frequency & severity models to predict claim      ║
    # ║  behaviour. Computes TP premium (IRDAI GVW-slab tariff) and    ║
    # ║  OD premium (2.5% base rate × relativity × accident load /     ║
    # ║  expense denom). Applies NCB discount, premium floor, and       ║
    # ║  produces the final gross & pure premiums for every policy.     ║
    # ╚══════════════════════════════════════════════════════════════════╝
    def run_pricing(self, df: pd.DataFrame) -> pd.DataFrame:
        df_p = df.copy()

        for col in SEM["cat_cols"]:
            if col not in df_p.columns:
                df_p[col] = "Unknown"
            le    = self.encoders[col]
            known = set(le.classes_)
            df_p[col + "_enc"] = df_p[col].apply(lambda v: le.transform([v])[0] if v in known else 0)

        X_pred = df_p[self.feature_cols].fillna(0)
        df_p["pred_frequency"] = np.clip(self.trained_models["frequency"].predict(X_pred), 0, None)
        df_p["pred_severity"]  = np.clip(self.trained_models["severity"].predict(X_pred), 5000, None)
        df_p["accident_load"]  = 1.0 + df_p["accident_probability"] * 0.015

        today = datetime.date.today()
        def _years_since(row):
            lcd = row.get("last_claim_date", None)
            claims = row.get("total_claims_count", 0)
            if claims == 0 or lcd is None or str(lcd).strip() in ("", "None", "nan"):
                return 99.0   
            try:
                if isinstance(lcd, str):
                    lcd = datetime.date.fromisoformat(str(lcd)[:10])
                elif hasattr(lcd, 'date'):
                    lcd = lcd.date()
                return max(0.0, (today - lcd).days / 365.25)
            except Exception:
                return 99.0

        df_p["years_since_last_claim"] = df_p.apply(_years_since, axis=1)
        df_p["ncb_tier"] = df_p.apply(
            lambda r: infer_ncb_tier(
                int(r.get("total_claims_count", 0)),
                float(r["years_since_last_claim"])
            ), axis=1
        )

        PRICE_COLS = [
            "tp_pure_premium_compulsory", "tp_gross_premium_compulsory",
            "od_pure_premium_optional",   "od_gross_pre_ncb",
            "od_gross_premium_optional",  "ncb_tier", "ncb_discount_pct", "ncb_saving",
            "total_pure_premium",         "total_gross_premium",
            "relativity",                 "od_rate_pct_of_idv",
        ]

        def price_row(row):
            if row["uw_decision"] == "DECLINE":
                return pd.Series({c: np.nan for c in PRICE_COLS})
            total_loading = row["uw_extra_loading"] + row.get("uw_claims_loading", 0)
            comp = compute_premium_components(
                gvw=row["gross_vehicle_weight_kg"],
                idv=row["idv_insured_declared_value"],
                risk_score=row["composite_risk_score"],
                accident_load=row["accident_load"],
                total_uw_loading=total_loading,
                ncb_tier=int(row.get("ncb_tier", 0)),
            )
            return pd.Series(comp)

        priced = df_p.apply(price_row, axis=1)
        for col in PRICE_COLS:
            df_p[col] = priced[col]

        floor_vals = df_p.apply(lambda r: compute_minimum_realistic_premium(
            r["gross_vehicle_weight_kg"], r["idv_insured_declared_value"]), axis=1)
        df_p["premium_floor_applied"] = df_p["uw_decision"].ne("DECLINE") & \
                                        (df_p["total_gross_premium"] <= floor_vals + 0.01)
        df_p["total_gross_premium"] = np.where(
            df_p["uw_decision"].ne("DECLINE"),
            np.maximum(df_p["total_gross_premium"], floor_vals),
            np.nan
        )

        self.pricing_df   = df_p
        self.processed_df = df_p

        accepted = df_p[df_p["uw_decision"] != "DECLINE"]
        stats = {}
        for cat_col in ["vehicle_category", "goods_category", "state", "route_type",
                         "travel_time_pattern", "fatigue_monitoring_system"]:
            if cat_col not in df_p.columns:
                continue
            grp = accepted.groupby(cat_col).agg(
                count                   =("total_gross_premium",      "count"),
                mean_tp_pure            =("tp_pure_premium_compulsory", "mean"),
                mean_tp_gross           =("tp_gross_premium_compulsory","mean"),
                mean_od_pure            =("od_pure_premium_optional",   "mean"),
                mean_od_gross           =("od_gross_premium_optional",  "mean"),
                mean_total_pure         =("total_pure_premium",        "mean"),
                mean_total_gross        =("total_gross_premium",       "mean"),
                mean_acc_prob           =("accident_probability",      "mean"),
            ).round(2).reset_index()
            stats[cat_col] = grp
        self.dataset_stats = stats
        return df_p

    def run_full_pipeline(self, df: pd.DataFrame, fast_mode: bool = True):
        df_full, error_rows, _, _ = self.validate(df)
        return self.train_and_process(df_full, error_rows, fast_mode=fast_mode)

    def quote_single_policy(self, params: dict) -> dict:
        state      = params.get("state", "Maharashtra")
        route      = params.get("route_type", "Mixed")
        highway    = params.get("highway_usage", "Mixed NH+SH")
        climate_zn = params.get("climate_zone", "Standard Low Risk")
        d1, d2, d3 = params.get("top_destination_1", ""), params.get("top_destination_2", ""), params.get("top_destination_3", "")

        base    = SEM["climate_risk_base"]
        prof    = base.get(state, base["DEFAULT"])
        rc      = SEM["route_climate_risk"].get(route, {"flood_amp": 1.0, "cyclone_amp": 1.0})
        zm      = SEM["climate_zone_modifiers"].get(climate_zn, SEM["climate_zone_modifiers"]["Standard Low Risk"])

        flood   = min(10, round(prof["flood"]   * rc["flood_amp"]   * zm["flood"],   2))
        fog     = min(10, round(prof["fog"]      * zm["fog"],                         2))
        cyclone = min(10, round(prof["cyclone"]  * rc["cyclone_amp"] * zm["cyclone"], 2))
        route_risk = compute_route_risk_score(route, highway, d1, d2, d3)

        cargo   = params.get("goods_category", "General Goods")
        age     = params.get("vehicle_age_yrs", 5)
        gvw     = params.get("gross_vehicle_weight_kg", 15000)
        fuel    = params.get("fuel_type", "Diesel")
        bs      = params.get("bs_emission_norm", "BS-IV")
        fleet   = params.get("fleet_size", 5)
        exp_bz  = params.get("years_in_business", 5)
        permit  = params.get("permit_type", "State Permit")
        fatigue = params.get("fatigue_monitoring_system", "Not Installed")
        overload = params.get("overloading_incidents", 0)
        avg_exp  = params.get("average_driver_experience", 8.0)
        turnover = params.get("driver_turnover_rate", 25.0)
        at_fault = params.get("at_fault_accidents", 0)
        traffic_v = params.get("traffic_violations", 0)
        training = params.get("driver_training_program", "No")
        night_pct = params.get("night_travel_frequency", 50)
        travel_p  = params.get("travel_time_pattern", "Mixed (Day & Night)")
        claims_ct     = params.get("total_claims_count", 0)
        claim_amt     = params.get("total_claim_amount_paid", 0)
        yrs_since_lcm = params.get("years_since_last_claim", 99.0)

        ncb_tier_val  = infer_ncb_tier(int(claims_ct), float(yrs_since_lcm))
        ncb_disc      = ncb_discount_from_tier(ncb_tier_val)

        cargo_risk = SEM["cargo_risk_map"].get(cargo, 4.0)

        gvw_norm = min(gvw / 50000, 1.0)
        age_norm = min(age / 20, 1.0)
        route_raw = SEM["route_risk_map"].get(route, 5.0) / 10
        overload_n = min(overload / 5, 1.0)
        risk_exp = round(min(10.0, 10 * (0.35 * gvw_norm + 0.30 * age_norm + 0.22 * route_raw + 0.13 * overload_n)), 2)

        fatigue_sc = SEM["fatigue_monitor_safety"].get(fatigue, 0.2)
        safety = round(min(10.0, 10 * (0.40 * SEM["bs_norm_safety"].get(bs, 0.5)
                                       + 0.35 * SEM["fuel_safety"].get(fuel, 0.5)
                                       + 0.25 * fatigue_sc)), 2)

        exp_n = min(avg_exp / 20, 1.0)
        training_sc = SEM["driver_training_safety"].get(training, 0.25)
        fault_p = min(at_fault / 5, 1.0)
        viol_p = min(traffic_v / 10, 1.0)
        turnover_p = min(turnover / 50, 1.0)
        driver_beh = round(max(0.0, min(10.0, 10 * (0.30 * exp_n + 0.25 * training_sc
                                                    - 0.20 * fault_p - 0.15 * viol_p - 0.10 * turnover_p))), 2)

        night_risk_sc = round(
            min(10.0, 10 * min(night_pct / 100, 1.0) * (SEM["night_travel_risk"].get(travel_p, 1.1) / 1.30)), 2)

        acc_prob = compute_accident_probability(
            state, route, flood, fog, cyclone, age, gvw, cargo,
            night_travel_pct=night_pct, overloading=overload,
            at_fault=at_fault, traffic_viol=traffic_v, travel_pattern=travel_p)

        composite = round(min(10, (
                0.20 * cargo_risk + 0.20 * risk_exp
                + 0.15 * (10 - safety) + 0.15 * (10 - driver_beh)
                + 0.12 * acc_prob + 0.08 * night_risk_sc
                + 0.04 * flood + 0.03 * fog + 0.03 * cyclone
        )), 2)

        severity_factor = min(claim_amt / 500000, 1.0) if claim_amt > 0 else 0
        claims_load = round(min(0.10, claims_ct * 0.015 + severity_factor * 0.02), 3)

        if composite > 8.5 and age > 17:
            uw_decision, extra_load = "DECLINE", 0
        elif composite > 7.0 or (cargo == "Chemical/Hazardous" and permit != "National Permit"):
            uw_decision = "REFER"
            extra_load  = round(0.05 + (composite - 7.0) * 0.015, 3)
        else:
            uw_decision = "ACCEPT"
            extra_load  = round(max(0, (composite - 4.0) * 0.01), 3)

        idv     = params.get("idv_insured_declared_value", 1000000)
        pol_lim = round(idv * SEM["policy_limit_pct_of_idv"])

        row_dict = {
            "vehicle_age_yrs": age, "gross_vehicle_weight_kg": gvw,
            "idv_insured_declared_value": idv, "fleet_size": fleet,
            "years_in_business": exp_bz,
            "vehicle_category": params.get("vehicle_category", "MCV (7.5-12T)"),
            "truck_model": params.get("truck_model", "Tata 407 LCV"),
            "fuel_type": fuel, "bs_emission_norm": bs, "permit_type": permit,
            "ownership_type": params.get("ownership_type", "Proprietorship"),
            "goods_category": cargo, "route_type": route, "highway_usage": highway,
            "state": state, "rto_district": params.get("rto_district", "Mumbai"),
            "fatigue_monitoring_system": fatigue, "climate_zone": climate_zn,
            "driver_training_program": training, "travel_time_pattern": travel_p,
            "overloading_incidents": overload,
            "average_driver_experience": avg_exp,
            "driver_turnover_rate": turnover,
            "at_fault_accidents": at_fault,
            "traffic_violations": traffic_v,
            "total_claims_count": claims_ct,
            "total_claim_amount_paid": claim_amt,
            "night_travel_frequency": night_pct,
            "top_destination_1": d1,
            "top_destination_2": d2,
            "top_destination_3": d3,
        }

        if self.trained_models is None:
            raise RuntimeError("Models not trained yet. Run the pipeline first.")

        enc_row = {}
        for col in SEM["cat_cols"]:
            le = self.encoders[col]
            v  = str(row_dict.get(col, ""))
            enc_row[col + "_enc"] = le.transform([v])[0] if v in set(le.classes_) else 0
        for c in ["vehicle_age_yrs", "gross_vehicle_weight_kg", "idv_insured_declared_value",
                  "fleet_size", "years_in_business", "overloading_incidents",
                  "average_driver_experience", "driver_turnover_rate",
                  "at_fault_accidents", "traffic_violations",
                  "total_claims_count", "total_claim_amount_paid",
                  "night_travel_frequency"]:
            if c in row_dict:
                enc_row[c] = row_dict[c]

        X_s       = pd.DataFrame([enc_row])[[c for c in self.feature_cols if c in enc_row or c in pd.DataFrame([enc_row]).columns]].fillna(0)
        for fc in self.feature_cols:
            if fc not in X_s.columns:
                X_s[fc] = 0
        X_s = X_s[self.feature_cols]

        pred_freq  = max(0.0, float(self.trained_models["frequency"].predict(X_s)[0]))
        pred_sev   = max(5000, float(self.trained_models["severity"].predict(X_s)[0]))
        acc_load_v = round(1.0 + acc_prob * 0.015, 4)

        if uw_decision != "DECLINE":
            total_load = extra_load + claims_load
            comp = compute_premium_components(
                gvw=gvw, idv=idv, risk_score=composite,
                accident_load=acc_load_v, total_uw_loading=total_load,
                ncb_tier=ncb_tier_val,
            )
            floor_prem = compute_minimum_realistic_premium(gvw, idv)
            if comp["total_gross_premium"] < floor_prem:
                floor_diff   = floor_prem - comp["total_gross_premium"]
                od_gross_adj = round(comp["od_gross_premium_optional"] + floor_diff, 2)
                comp["od_gross_premium_optional"] = od_gross_adj
                comp["total_gross_premium"]       = round(comp["tp_gross_premium_compulsory"] + od_gross_adj, 2)
                comp["premium_floor_applied"]     = True
            else:
                comp["premium_floor_applied"] = False

            tp_pure_prem    = comp["tp_pure_premium_compulsory"]
            tp_gross_prem   = comp["tp_gross_premium_compulsory"]
            od_pure_prem    = comp["od_pure_premium_optional"]
            od_gross_pre_ncb= comp["od_gross_pre_ncb"]
            od_gross_prem   = comp["od_gross_premium_optional"]
            total_pure      = comp["total_pure_premium"]
            total_gross     = comp["total_gross_premium"]
            relativity      = comp["relativity"]
            od_rate_pct     = comp["od_rate_pct_of_idv"]
            ncb_saving_val  = comp["ncb_saving"]
        else:
            comp_dec = compute_premium_components(
                gvw=gvw, idv=idv, risk_score=composite,
                accident_load=acc_load_v, total_uw_loading=0.0,
                ncb_tier=0)
            tp_pure_prem    = comp_dec["tp_pure_premium_compulsory"]
            tp_gross_prem   = comp_dec["tp_gross_premium_compulsory"]
            od_pure_prem    = comp_dec["od_pure_premium_optional"]
            od_gross_pre_ncb= comp_dec["od_gross_pre_ncb"]
            od_gross_prem   = comp_dec["od_gross_premium_optional"]
            total_pure      = comp_dec["total_pure_premium"]
            total_gross     = None
            relativity      = comp_dec["relativity"]
            od_rate_pct     = comp_dec["od_rate_pct_of_idv"]
            ncb_saving_val  = 0

        return {
            "inputs": row_dict,
            "climate": {
                "flood": flood, "fog": fog, "cyclone": cyclone,
                "route_risk": route_risk,
                "climate_zone": climate_zn,
                "climate_composite": round(0.38*flood + 0.22*fog + 0.25*cyclone + 0.15*route_risk, 2),
            },
            "risk_scores": {
                "cargo_risk":            round(cargo_risk, 2),
                "risk_exposure":         round(risk_exp, 2),
                "safety_score":          round(safety, 2),
                "driver_behavior_score": round(driver_beh, 2),
                "night_travel_risk":     round(night_risk_sc, 2),
                "accident_probability":  acc_prob,
                "composite_risk":        composite,
            },
            "underwriting": {
                "decision":              uw_decision,
                "extra_loading_pct":     extra_load * 100,
                "claims_loading_pct":    claims_load * 100,
                "expenses_loading_pct":  SEM["expenses_loading"] * 100,
                "profit_margin_pct":     SEM["profit_margin"] * 100,
                "other_loading_pct":     SEM.get("other_loading", 0.0) * 100,
                "deductible_pct":        SEM["deductible_pct"] * 100,
                "policy_limit":          pol_lim,
            },
            "pricing": {
                "pred_frequency":                round(pred_freq, 4),
                "pred_severity":                 round(pred_sev, 2),
                "accident_load":                 acc_load_v,
                "relativity":                    relativity,
                "od_rate_pct_of_idv":            od_rate_pct,
                "tp_pure_premium_compulsory":    tp_pure_prem,
                "tp_gross_premium_compulsory":   tp_gross_prem,
                "od_pure_premium_optional":      od_pure_prem,
                "od_gross_pre_ncb":              od_gross_pre_ncb,
                "od_gross_premium_optional":     od_gross_prem,
                "ncb_tier":                      ncb_tier_val,
                "ncb_discount_pct":             round(ncb_disc * 100, 1),
                "ncb_saving":                    ncb_saving_val,
                "total_pure_premium":            total_pure,
                "total_gross_premium":           total_gross,
            },
        }

    def export_full_dataset_bytes(self) -> bytes:
        cols = [
            "policy_id", "vehicle_registration_number", "state", "vehicle_category",
            "truck_model", "goods_category", "route_type", "climate_zone",
            "idv_insured_declared_value",
            "climate_flood_score", "climate_fog_score", "climate_cyclone_score",
            "route_risk_score", "climate_composite",
            "cargo_risk_score", "risk_exposure_score", "safety_score",
            "driver_behavior_score", "night_travel_risk_score",
            "accident_probability", "composite_risk_score",
            "total_claims_count", "total_claim_amount_paid",
            "uw_decision", "uw_extra_loading", "uw_claims_loading",
            "ncb_tier", "ncb_discount_pct", "ncb_saving",
            "tp_pure_premium_compulsory", "tp_gross_premium_compulsory",
            "od_pure_premium_optional",   "od_gross_pre_ncb",
            "od_gross_premium_optional",  "total_pure_premium",
            "total_gross_premium",        "od_rate_pct_of_idv",
            "data_validation_status",
        ]
        source = self.full_export_df if self.full_export_df is not None else self.processed_df
        cols = [c for c in cols if c in source.columns]
        out  = source[cols].copy()
        buf  = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            out.to_excel(writer, sheet_name="Portfolio", index=False)
        return buf.getvalue()

def df_to_excel_bytes(df: pd.DataFrame, sheet_name="Result") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    return buf.getvalue()


# ============================================================
# SECTION 2 — OPTIONAL AI ASSISTANT (Gemini)
# ┌──────────────────────────────────────────────────────────┐
# │  AI ASSISTANT — TruckShield AI (Powered by Gemini)      │
# │  Provides a conversational interface for underwriters    │
# │  and analysts to query risk factors, climate zones,      │
# │  driver scoring methodology, and premium breakdowns.     │
# │  Requires a Gemini API key (entered in the sidebar).     │
# │  All five agent outputs are injected into the system     │
# │  prompt so the model can reason over live portfolio data.│
# │                                                          │
# │  OPEN SOURCE KNOWLEDGE: Uses the Wikipedia REST API     │
# │  (https://en.wikipedia.org/api/rest_v1/) for technical  │
# │  questions outside the model's embedded knowledge.      │
# │  No API key required — free and open to all.            │
# └──────────────────────────────────────────────────────────┘
# ============================================================

# ─────────────────────────────────────────────────────────────
# WIKIPEDIA OPEN-SOURCE KNOWLEDGE LOOKUP
# Uses the Wikipedia REST API — no key required, fully open.
# Called when the user asks a technical question that needs
# background knowledge beyond what is embedded in the model.
# Attribution is always included in the final answer.
# ─────────────────────────────────────────────────────────────
def _wikipedia_search(query: str, sentences: int = 5) -> dict:
    """
    Searches Wikipedia for a topic and returns a plain-text summary.
    Uses the Wikipedia REST API (https://en.wikipedia.org/api/rest_v1/).
    This is a free, open-source, publicly available knowledge base.
    No API key is required.

    Returns a dict with keys:
        'found'   : bool
        'title'   : str   — the matched Wikipedia article title
        'summary' : str   — the first N sentences of the article
        'url'     : str   — link to the full article
        'error'   : str   — set only if an error occurred
    """
    result = {"found": False, "title": "", "summary": "", "url": "", "error": ""}

    try:
        # Search for the closest matching article title
        search_url = (
            "https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={urllib.parse.quote(query)}&format=json&srlimit=1"
        )
        with urllib.request.urlopen(search_url, timeout=6) as resp:
            search_data = json.loads(resp.read().decode())

        hits = search_data.get("query", {}).get("search", [])
        if not hits:
            result["error"] = "No Wikipedia article found."
            return result

        title = hits[0]["title"]

        # Now pull the actual page summary
        summary_url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + urllib.parse.quote(title.replace(" ", "_"))
        )
        with urllib.request.urlopen(summary_url, timeout=6) as resp:
            page = json.loads(resp.read().decode())

        extract = page.get("extract", "").strip()
        if not extract:
            result["error"] = "Article found but summary is empty."
            return result

        # Trim to requested number of sentences
        raw_sentences = re.split(r'(?<=[.!?])\s+', extract)
        summary = " ".join(raw_sentences[:sentences])

        result["found"]   = True
        result["title"]   = page.get("title", title)
        result["summary"] = summary
        result["url"]     = page.get("content_urls", {}).get("desktop", {}).get("page", "")

    except Exception as exc:
        result["error"] = f"Wikipedia lookup failed: {str(exc)[:120]}"
        # Note: Even if Wikipedia is unreachable, get_chat_response continues normally.
        # Gemini will answer using its own training knowledge for the technical concept,
        # but without the explicit Wikipedia attribution in the answer.

    return result


# ─────────────────────────────────────────────────────────────
# TOPIC CLASSIFIER — decides whether a Wikipedia lookup is needed
# Returns one of: "project_internal", "technical_external", "out_of_scope"
# ─────────────────────────────────────────────────────────────
_TECHNICAL_TRIGGERS = [
    # ML / statistics / algorithms — needs Wikipedia for background explanation
    "xgboost", "gradient boosting", "poisson", "gamma distribution", "regression model",
    "machine learning", "ml model", "frequency model", "severity model",
    "train test split", "label encoder", "sklearn", "scikit",
    "random state", "subsample", "colsample", "n_estimators", "learning rate",
    "log compression", "log-compression", "actuarial science", "credibility theory",
    "tweedie", "compound model", "glm", "generalised linear",
    # LLM / AI technology — needs Wikipedia for background
    "gemini api", "google gemini", "google ai studio", "generativeai",
    "large language model", "llm", "neural network", "deep learning",
    "transformer model", "bert", "gpt", "open source platform",
    # Python libraries / tech stack — needs Wikipedia for background
    "streamlit framework", "plotly library", "reportlab", "openpyxl",
    "xgboost library", "sklearn library", "numpy library", "pandas library",
    # Regulatory bodies (general background needed) — not project formula detail
    "what is irdai", "about irdai", "what is morth", "about morth",
    "what is ncrb", "about ncrb", "what is nhai", "about nhai",
    # Vehicle/safety technology background
    "what is adas", "advanced driver assistance", "what is telematics",
    "vehicle telematics", "what is ais-140", "ais 140 standard",
    "what is bharat stage", "bs emission norms background",
]

_OUT_OF_SCOPE_TRIGGERS = [
    "stock", "crypto", "bitcoin", "forex", "investment", "share market",
    "politics", "election", "religion", "sports", "cricket", "football",
    "movie", "film", "music", "song", "recipe", "cooking", "food",
    "travel tips", "tourism", "hotel", "flight booking",
    "health advice", "medical", "medicine", "doctor",
    "personal loan", "home loan", "mortgage",
    "exam", "study tips", "homework", "assignment",
    "news", "breaking news", "current events",
    "weather forecast", "astrology", "horoscope",
    "legal advice", "court", "lawsuit",
    "hr policy", "salary negotiation", "job interview",
    "social media", "instagram", "facebook", "twitter",
]

_SELF_KNOWLEDGE_TRIGGERS = [
    "who are you", "what are you", "tell me about yourself", "your history",
    "about this model", "about truckshield", "about the model",
    "who built you", "who made you", "who developed you", "your developer",
    "who built this", "who created this", "who created you",
    "developer team", "team members", "mentor", "internship",
    "your capabilities", "what can you do", "how do you work",
    "how were you built", "how was this model built", "how was this built",
    "Sri Sathya Sai Institute of Actuaries Support Group", "Team 5", "Tesa Joby", "Khyati Bindal",
    "Lakshita Siwach", "Yashaswi Reddy", "Siva Ruben", "Aniket Kadam", "Dr. Rohan Yashraj Gupta, FIA, FIAI",
    "Sathya Sai Mudigonda",
]


def _classify_question(user_message: str) -> str:
    """
    Classifies the user message into one of three categories:
      - 'out_of_scope'       : unrelated to the project; should be refused
      - 'self_knowledge'     : questions about the model/team/history
      - 'technical_external' : needs Wikipedia for background context
      - 'project_internal'   : answerable from embedded model knowledge
    """
    msg_lower = user_message.lower()

    # Out-of-scope check first (highest priority guard)
    if any(t in msg_lower for t in _OUT_OF_SCOPE_TRIGGERS):
        return "out_of_scope"

    # Self-knowledge check
    if any(t in msg_lower for t in _SELF_KNOWLEDGE_TRIGGERS):
        return "self_knowledge"

    # Technical / external knowledge check
    if any(t in msg_lower for t in _TECHNICAL_TRIGGERS):
        return "technical_external"

    return "project_internal"


def _extract_wikipedia_query(user_message: str) -> str:
    """
    Extracts the best Wikipedia search query from the user's message.
    Maps recognised technical keywords to precise Wikipedia article titles
    to maximise relevance of the returned summary.
    """
    msg = user_message.lower()

    priority_topics = [
        # XGBoost / ML
        ("xgboost poisson", "XGBoost Poisson regression count data"),
        ("poisson regression", "Poisson regression"),
        ("gamma distribution", "Gamma distribution statistics"),
        ("gamma regressor", "Gamma distribution statistics"),
        ("xgboost", "XGBoost gradient boosting"),
        ("gradient boosting", "Gradient boosting"),
        ("tweedie", "Tweedie distribution actuarial"),
        ("credibility theory", "Credibility theory actuarial"),
        ("label encoder", "Feature engineering machine learning"),
        ("train test split", "Cross-validation machine learning"),
        ("frequency severity", "Frequency severity model actuarial"),
        ("compound model", "Compound Poisson distribution"),
        ("generalised linear", "Generalized linear model"),
        ("glm", "Generalized linear model"),
        ("log compression", "Logarithm mathematics"),
        # LLM / AI
        ("gemini api", "Google Gemini artificial intelligence"),
        ("google gemini", "Google Gemini artificial intelligence"),
        ("large language model", "Large language model"),
        ("llm", "Large language model"),
        ("generativeai", "Generative artificial intelligence"),
        ("neural network", "Artificial neural network"),
        ("transformer model", "Transformer machine learning model"),
        # Python libraries
        ("streamlit", "Streamlit Python framework"),
        ("plotly", "Plotly data visualization"),
        ("xgboost library", "XGBoost gradient boosting"),
        ("sklearn", "scikit-learn machine learning library"),
        # Regulatory bodies
        ("irdai", "Insurance Regulatory and Development Authority India"),
        ("morth", "Ministry of Road Transport and Highways India"),
        ("ncrb", "National Crime Records Bureau India"),
        ("nhai", "National Highways Authority of India"),
        # Vehicle technology
        ("adas", "Advanced driver-assistance systems"),
        ("telematics", "Vehicle telematics"),
        ("ais-140", "AIS 140 vehicle tracking India"),
        ("ais 140", "AIS 140 vehicle tracking India"),
        ("bharat stage", "Bharat Stage emission standards"),
        ("bs emission", "Bharat Stage emission standards"),
        # Actuarial / insurance
        ("actuarial science", "Actuarial science"),
        ("pure premium", "Pure premium insurance actuarial"),
        ("loss ratio", "Loss ratio insurance"),
        ("insured declared value", "Insured Declared Value vehicle insurance India"),
        ("gross vehicle weight", "Gross vehicle weight rating"),
    ]

    for keyword, wiki_query in priority_topics:
        if keyword in msg:
            return wiki_query

    # Fallback: use the raw message trimmed
    return user_message.strip()[:60]


# ─────────────────────────────────────────────────────────────
# SYSTEM PROMPT BUILDER
# ─────────────────────────────────────────────────────────────
_MODEL_SELF_KNOWLEDGE = """
═══════════════════════════════════════════════════════════════
ABOUT THIS MODEL — TRUCKSHIELD AI
═══════════════════════════════════════════════════════════════
Full Name   : TruckShield AI — Commercial Truck Insurance Pricing & Underwriting Model
Type        : Multi-Agent Actuarial & ML Pipeline with Conversational AI Assistant
Built By    : Team 5, under the mentorship of Sri Sathya Sai Institute of Actuaries (SSSIA)
Project     : Internship project — May 2025 to June 2025
Purpose     : Automated pricing, risk scoring, climate assessment, and underwriting
              for commercial truck insurance policies across India.

ARCHITECTURE — FIVE AGENTS IN SEQUENCE:
  Agent 1 — Data Validation Agent   : Validates uploaded policy data; flags bad rows
  Agent 2 — Climate Risk Agent       : Scores Flood/Fog/Cyclone risk per state + zone + route
  Agent 3 — Risk Scoring Agent       : Computes Cargo, Exposure, Safety, Driver Behavior, Night
                                       Travel, and Accident Probability → Composite Risk Score
  Agent 4 — Underwriting Agent       : Accept / Refer / Decline decisions + risk & claims loading
  Agent 5 — Pricing Agent            : XGBoost frequency × severity → TP + OD premium (with NCB)

AI ASSISTANT:
  Powered by Google Gemini 2.5 Flash (LLM).
  Augmented by Wikipedia open-source knowledge for technical questions.
  Scoped strictly to this project — out-of-scope questions are refused.

WHY GEMINI FOR THE AI ASSISTANT?
  Google Gemini 2.5 Flash was chosen because it is a state-of-the-art, fast, and
  cost-effective large language model with a simple Python SDK (google-generativeai).
  It has strong reasoning over structured actuarial prompts, supports long context
  windows (needed to inject full pipeline summaries), and requires only a single
  API key — making it accessible for a student internship project without complex
  infrastructure.

WHY XGBOOST WITH POISSON & GAMMA OBJECTIVES?
  Claim frequency (how often accidents occur) follows a Poisson distribution because:
    - Claims are rare, independent events over a fixed time period (the policy year)
    - The Poisson distribution is the canonical model for count data in actuarial science
    - XGBoost's objective="count:poisson" directly optimises the Poisson log-likelihood
  Claim severity (the cost per claim) follows a Gamma distribution because:
    - Severity is strictly positive and right-skewed (a few large claims, many small ones)
    - The Gamma distribution captures this shape naturally in GLM/boosting frameworks
    - XGBoost's objective="reg:gamma" optimises the Gamma deviance, not mean squared error,
      which prevents the model from being dominated by extreme outliers
  Together (frequency × severity) they form the compound model — standard actuarial practice
  for pure premium = E[frequency] × E[severity]. This is the Tweedie framework.

DATA SOURCES USED IN SCORING:
  - NDMA State Disaster Management Plans + IMD 3-year rolling avg → Climate risk base
  - NCRB (National Crime Records Bureau) + MoRTH 2023 → Accident probability calibration
  - IRDAI Motor Insurance Regulations → TP tariff slabs, NCB grid, OD rate bounds
  - NHAI data → Overloading risk factor (2× brake failure rate for overloaded trucks)

═══════════════════════════════════════════════════════════════
ABOUT DEVELOPER TEAM
═══════════════════════════════════════════════════════════════
MENTOR TEAM:
  • Sri Sathya Sai Institute of Actuaries (SSSIA) Support Group
    Official Website: https://sssia.org/
  • Sathya Sai Mudigonda
    LinkedIn: https://www.linkedin.com/in/satya-sai-mudigonda
  • Dr. Rohan Yashraj Gupta, FIA, FIAI
    LinkedIn: https://www.linkedin.com/in/rohanyashraj

DEVELOPER TEAM:
  • Tesa Joby           — Risk Analytics & Dataset Expert
    LinkedIn: https://www.linkedin.com/in/tesa-joby-02b28b243
  • Khyati Bindal        — Climate Analytics & Dataset Expert
    LinkedIn: https://www.linkedin.com/in/khyati-bindal
  • Lakshita Siwach      — Underwriting Analytics & Dataset Expert
    LinkedIn: https://www.linkedin.com/in/lakshita-siwach-952b77303
  • Yashaswi Reddy       — Pricing Analytics & Dataset Expert
    LinkedIn: https://www.linkedin.com/in/yashaswi-483a68261
  • Siva Ruben           — Underwriting Analytics & Dataset Expert
    LinkedIn: https://www.linkedin.com/in/siva-ruban-b80185367
  • Aniket Kadam         — Pricing Analytics & Modelling Expert
    LinkedIn: https://www.linkedin.com/in/aniket-kadam-1856362a1
"""


def _build_system_prompt(engine, history, wiki_context: str = "") -> str:
    pricing_summary = "Pricing not yet computed."
    if engine.pricing_df is not None:
        pr  = engine.pricing_df
        acc = pr[pr["uw_decision"] != "DECLINE"]
        if len(acc) > 0:
            pricing_summary = (
                f"{len(pr)} policies total. "
                f"Mean Total Pure ₹{acc['total_pure_premium'].mean():,.0f}, "
                f"Mean Total Gross ₹{acc['total_gross_premium'].mean():,.0f}. "
                f"Mean TP Gross ₹{acc['tp_gross_premium_compulsory'].mean():,.0f}, "
                f"Mean OD Gross ₹{acc['od_gross_premium_optional'].mean():,.0f}. "
                f"Accepted:{(pr['uw_decision']=='ACCEPT').sum()}, "
                f"Referred:{(pr['uw_decision']=='REFER').sum()}, "
                f"Declined:{(pr['uw_decision']=='DECLINE').sum()}."
            )

    stats_block = ""
    if engine.dataset_stats:
        lines = ["\nCATEGORY-WISE STATS:"]
        for cat, df in engine.dataset_stats.items():
            lines.append(f"  [{cat}]")
            for _, r in df.iterrows():
                lines.append(f"    {r[cat]}: count={r['count']}, "
                              f"mean_total_gross=₹{r['mean_total_gross']:,.0f}, "
                              f"mean_tp_gross=₹{r['mean_tp_gross']:,.0f}, "
                              f"mean_od_gross=₹{r['mean_od_gross']:,.0f}, "
                              f"mean_total_pure=₹{r['mean_total_pure']:,.0f}, "
                              f"mean_acc_prob={r['mean_acc_prob']:.2f}")
        stats_block = "\n".join(lines)

    history_block = "No prior conversation." if not history else "\n".join(
        f"{'User' if h['role']=='user' else 'Asst'}: {h['content'][:120]}" for h in history[-14:])

    wiki_section = ""
    if wiki_context:
        wiki_section = f"""
OPEN-SOURCE KNOWLEDGE (Wikipedia):
{wiki_context}
INSTRUCTION: When using the above Wikipedia content in your answer, you MUST explicitly state:
"According to Wikipedia (open-source knowledge)" before the relevant part of your answer.
Include the Wikipedia article URL if available. Then relate it back to this project's design choices.
"""

    exp_pct   = SEM["expenses_loading"]  * 100
    pm_pct    = SEM["profit_margin"]     * 100
    oth_pct   = SEM.get("other_loading", 0.0) * 100
    ded_pct   = SEM["deductible_pct"]   * 100
    total_pct = exp_pct + pm_pct + oth_pct

    return f"""You are TruckShield AI — an expert commercial truck insurance assistant for India.
You are the AI assistant built into the TruckShield AI dashboard, a 5-stage actuarial multi-agent model.

{_MODEL_SELF_KNOWLEDGE}

CONVERSATION HISTORY:
{history_block}

PIPELINE KNOWLEDGE:
- Climate Agent: Flood/Fog/Cyclone (0-10) per state+route+climate_zone, plus Route Risk Score.
  Route risk captures route type, highway usage, and multi-destination freight exposure.
  Climate composite = 0.38×Flood + 0.22×Fog + 0.25×Cyclone + 0.15×RouteRisk.
- Risk Scoring Agent: Cargo Risk, Exposure (GVW+age+route+overloading), Safety (BS+fuel+fatigue monitor),
  Driver Behavior (experience, training, at-fault, violations, turnover), Night Travel Risk,
  Accident Probability (NCRB/MoRTH additive model). By Driver Behavior.
- Underwriting Agent: Accept/Refer/Decline + extra risk loading + claims history loading.
  Claims loading = min(10%, claims_count×1.5% + severity_factor×2%).
- Pricing Agent: {pricing_summary}
{stats_block}

ACCIDENT PROBABILITY METHODOLOGY (v2 — additive weighted scoring):
  The model uses ADDITIVE factor scoring, not multiplicative amplifiers.
  Reason: multiplicative stacking caused near-universal saturation at 10/10 (meaningless).
  Calibration anchor: MoRTH 2023 reports ~4.5% of commercial vehicles involved in accidents per year.
  On a 0-10 scale, average-risk policy → ~4.5, best realistic → ~1.5, worst realistic → ~8.5.
  Score NEVER reaches 10; mathematical ceiling is 9.0 via log-compression.

  Factor contributions (max points each):
    Route+State base : 0–4.0 pts  (55% route × 45% state, rescaled)
    Vehicle age      : 0–1.0 pts
    Cargo risk       : 0–0.8 pts
    Climate          : 0–0.8 pts  (fog/flood/cyclone additive)
    Night travel     : 0–1.0 pts  (MoRTH: 45% of fatal accidents happen at night)
    Overloading      : 0–0.7 pts  (NHAI: overloaded vehicles have 2× brake failure rate)
    Driver behaviour : 0–1.2 pts
    GVW              : 0–0.5 pts
  Raw sum → log-compressed: compressed = 9.0 × (1 − e^(−raw/7)). Bounded [0.5, 9.0].

PREMIUM REDUCTION TIPS ENGINE:
  1. TP Tips: IRDAI-fixed. Only lever is GVW slab (shown if within 3,000 kg of lower slab).
  2. OD Tips: triggered by fatigue monitor, BS norm, fuel, driver training, at-fault, violations,
     overloading, night travel, route risk, driver turnover.
  3. General Tips: claims loading, voluntary deductible, National Permit, fleet discount,
     AIS-140 telematics, 3-year policy, NCB protect add-on.

NCB (NO CLAIM BONUS):
  Applied to OD gross ONLY. Never to TP.
  Tier 0=0%, Tier 1=20%, Tier 2=25%, Tier 3=35%, Tier 4=45%, Tier 5=50%.
  Resets on ANY claim. NCB protect add-on preserves after one claim/year.

FORMULAS:
  tp_gross  = GVW-slab IRDAI tariff (fixed)
  tp_pure   = tp_gross × 0.70
  relativity = composite_risk / 5.0  (bounded 0.60–3.00)
  od_rate    = 2.5% × relativity × accident_load  (bounded 1.5%–9.0% of IDV)
  od_pure    = od_rate × IDV
  od_gross   = od_pure / (1 − expenses_loading − profit_margin − other_loading − ExtraLoad − ClaimsLoad)
  od_gross_after_ncb = od_gross × (1 − ncb_discount)
  total_gross = tp_gross + od_gross_after_ncb
  Accident Load = 1 + accident_probability × 0.015
  Composite Risk = 0.20×Cargo + 0.20×Exposure + 0.15×(10−Safety) + 0.15×(10−DriverBehavior)
                   + 0.12×AccidentProb + 0.08×NightTravel + 0.04×Flood + 0.03×Fog + 0.03×Cyclone

CURRENT ACTUARIAL PARAMETERS (user-configured via sidebar):
  Expenses Loading : {exp_pct:.1f}%
  Profit Margin    : {pm_pct:.1f}%
  Other Loading    : {oth_pct:.1f}%
  Deductible       : {ded_pct:.1f}%
  Combined Loading : {total_pct:.1f}%  (must stay below 100%)
{wiki_section}

══════════════════════════════════════════════════════
STRICT BEHAVIOURAL RULES — YOU MUST FOLLOW ALL OF THESE
══════════════════════════════════════════════════════
1. SCOPE GUARD: You only answer questions related to this TruckShield AI project:
   commercial truck insurance, the five agents, actuarial methods, risk scoring,
   pricing formulas, the developer team, or technical concepts used in this model.
   If a question is about ANYTHING else (stocks, politics, general health, travel,
   movies, sports, recipes, personal advice, etc.), respond EXACTLY with:
   "⚠️ I'm sorry, that question is outside the scope of TruckShield AI. I can only
   assist with commercial truck insurance, this model's methodology, and the technology
   used to build it. Please ask something related to the project."

2. WIKIPEDIA ATTRIBUTION: If Wikipedia knowledge is provided above (in the
   OPEN-SOURCE KNOWLEDGE section), you MUST say "According to Wikipedia
   (open-source knowledge):" before quoting or paraphrasing it. Always include
   the Wikipedia URL. Then connect it to how this project uses the concept.

3. DETAILED ANSWERS: If a user asks for more detail, first use all available
   information in this prompt (formulas, pipeline knowledge, model self-knowledge).
   If deeper technical background is needed, state: "For further technical depth,
   I've drawn from Wikipedia (open-source knowledge)" and explain accordingly.

4. SELF-KNOWLEDGE: You can fully explain who you are, your history, your
   architecture (5 agents), your developer team, why Gemini was chosen, why
   XGBoost Poisson/Gamma was chosen, what data sources are used, and your
   project timeline (May–June 2025 internship, SSSIA mentorship, Team 5).

5. DO NOT FABRICATE: Never invent premium figures, policy decisions, or
   statistics not present in the pipeline data above.

6. FORMAT: Be concise for simple questions. Use tables for comparisons.
   Use bullet points for multi-part answers. Currency always in ₹ with commas.
   For specific premium quotes, direct user to the "Get a Quote" tab.
"""


# ─────────────────────────────────────────────────────────────
# MAIN CHAT RESPONSE FUNCTION
# Orchestrates: classify → Wikipedia lookup (if needed) → Gemini
# ─────────────────────────────────────────────────────────────
def get_chat_response(engine, user_message: str, api_key: str, history: list) -> str:
    # google.generativeai is an optional dependency — only imported here when the user
    # actually has a Gemini key. This keeps the base app working without installing it.
    import google.generativeai as genai

    # ── Step 1: Classify the question ──
    question_type = _classify_question(user_message)

    # ── Step 2: Hard refusal for out-of-scope questions ──
    if question_type == "out_of_scope":
        return (
            "⚠️ I'm sorry, that question is outside the scope of TruckShield AI. "
            "I can only assist with commercial truck insurance, this model's methodology, "
            "and the technology used to build it. Please ask something related to the project."
        )

    # ── Step 3: Wikipedia lookup for technical questions ──
    wiki_context = ""
    if question_type == "technical_external":
        wiki_query = _extract_wikipedia_query(user_message)
        wiki_result = _wikipedia_search(wiki_query, sentences=6)
        if wiki_result["found"]:
            wiki_context = (
                f"Article: {wiki_result['title']}\n"
                f"URL: {wiki_result['url']}\n"
                f"Summary: {wiki_result['summary']}"
            )
        # If Wikipedia lookup fails, proceed without it — Gemini still has embedded knowledge

    # ── Step 4: Build system prompt (with or without Wikipedia context) ──
    system_prompt = _build_system_prompt(engine, history, wiki_context=wiki_context)

    # ── Step 5: Call Gemini ──
    genai.configure(api_key=api_key)
    model      = genai.GenerativeModel("gemini-2.5-flash")
    full_prompt = f"{system_prompt}\n\nUSER: {user_message}\n\nYour response:"
    response   = model.generate_content(full_prompt)
    return response.text.strip()


# ============================================================
# SECTION 3 — STREAMLIT DASHBOARD UI
# ============================================================
if "engine" not in st.session_state:
    st.session_state.engine = TruckShieldEngine()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sample_df" not in st.session_state:
    st.session_state.sample_df = None
if "validation_pending" not in st.session_state:
    st.session_state.validation_pending = False     
if "validation_quit" not in st.session_state:
    st.session_state.validation_quit = False         
if "df_full_for_training" not in st.session_state:
    st.session_state.df_full_for_training = None
if "error_rows_for_training" not in st.session_state:
    st.session_state.error_rows_for_training = set()
if "fast_mode_for_training" not in st.session_state:
    st.session_state.fast_mode_for_training = True

engine: TruckShieldEngine = st.session_state.engine

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚛 TruckShield AI")
    st.caption("Commercial truck insurance — 5-stage pricing engine")

    st.markdown("### 1. Data source")
    source = st.radio("Choose dataset",
                       ["Use sample dataset (demo)", "Upload my own Excel file or csv file"],
                       label_visibility="collapsed")

    df_input = None
    if source == "Use sample dataset (demo)":
        n_rows = st.slider("Sample size", 500, 5000, 500, step=500)
        if st.button("🎲 Generate sample dataset", use_container_width=True, type="primary") or st.session_state.sample_df is None:
            st.session_state.sample_df = generate_sample_dataset(n=n_rows)
        df_input = st.session_state.sample_df
        st.caption(f"Synthetic dataset: {len(df_input)} policies, full schema.")
    else:
        uploaded   = st.file_uploader("Upload .xlsx or .csv", type=["xlsx", "csv"])
        header_row = st.number_input("Header row index (0 = first row is column names)", 0, 5, 0)
        if uploaded is not None:
            try:
                if uploaded.name.endswith(".csv"):
                    df_input = pd.read_csv(uploaded, header=header_row)
                else:
                    df_input = pd.read_excel(uploaded, header=header_row)
                df_input = normalise_columns(df_input)
                st.success(f"Loaded {len(df_input)} rows × {len(df_input.columns)} columns.")
            except Exception as e:
                st.error(f"Could not read file: {e}")

    st.markdown("---")

    # ── ACTUARIAL PARAMETERS ──────────────────────────────────
    # Placed before the pipeline run so users can dial in their
    # preferred loading assumptions first, then hit Run once.
    # The constraint below mirrors real actuarial practice:
    # expenses + profit + other loading must stay below 100%
    # combined, otherwise the gross premium formula breaks down
    # (denominator goes to zero or negative).
    # Deductible is a separate mechanism — it reduces the insurer's
    # expected loss cost and can be set independently.
    # ─────────────────────────────────────────────────────────────
    st.markdown("### 2. Actuarial Parameters")
    st.caption(
        "Set your pricing assumptions here **before** running the pipeline. "
        "**Expenses + Profit + Other ≤ 100%** is enforced — deductible is uncapped."
    )

    # Read current values from SEM defaults
    _cur_exp  = int(round(SEM["expenses_loading"] * 100))
    _cur_pm   = int(round(SEM["profit_margin"]    * 100))
    _cur_oth  = int(round(SEM.get("other_loading", 0.0) * 100))
    _cur_ded  = int(round(SEM["deductible_pct"]   * 100))

    exp_loading = st.slider(
        "Expense Loading (%)",
        min_value=0, max_value=100, value=_cur_exp, step=1,
        help="Covers acquisition costs, admin overheads, and reinsurance cession charges.",
    )

    # Profit margin ceiling is whatever budget remains after expenses
    _pm_cap = max(0, 100 - exp_loading)
    profit_margin = st.slider(
        "Profit Margin (%)",
        min_value=0, max_value=_pm_cap, value=min(_cur_pm, _pm_cap), step=1,
        help="Target underwriting profit. Cannot push combined loadings past 100%.",
    )

    # Other loading ceiling is whatever is left after expenses + profit
    _oth_cap = max(0, 100 - exp_loading - profit_margin)
    other_loading = st.slider(
        "Other Loading (%) — e.g. cat margin, IBNR buffer",
        min_value=0, max_value=_oth_cap, value=min(_cur_oth, _oth_cap), step=1,
        help="Any additional loading — catastrophe buffer, unexpired risk reserve, etc.",
    )

    # Show the remaining pricing bandwidth so it's obvious to the user
    _total_loading = exp_loading + profit_margin + other_loading
    _remaining     = 100 - _total_loading
    if _remaining > 0:
        st.success(f"Combined loading: **{_total_loading}%** — {_remaining}% bandwidth remaining.")
    elif _remaining == 0:
        st.warning("Combined loading is exactly 100% — gross premium will equal pure premium.")
    else:
        st.error("⚠️ Combined loading exceeds 100%. Adjust the sliders.")

    deductible_pct = st.slider(
        "Deductible (%)",
        min_value=0, max_value=100, value=_cur_ded, step=1,
        help=(
            "The share of each OD claim absorbed by the policyholder before the insurer pays. "
            "Higher deductible = lower OD premium. No cap applies here — it is not part of the "
            "expense/profit loading stack."
        ),
    )

    # Push the new values into the live SEM dictionary so pricing picks them up immediately
    SEM["expenses_loading"] = exp_loading  / 100.0
    SEM["profit_margin"]    = profit_margin / 100.0
    SEM["other_loading"]    = other_loading / 100.0
    SEM["deductible_pct"]   = deductible_pct / 100.0

    st.markdown("---")
    st.markdown("### 3. Run the pipeline")
    fast_mode = st.checkbox("⚡ Fast mode (fewer trees)", value=True,
                             help="80 trees instead of 200 — premiums still realistic, trains in seconds.")
    run_clicked = st.button("▶️ Run 5-Agent Pipeline", type="primary", use_container_width=True,
                             disabled=df_input is None)

    if run_clicked and df_input is not None:
        required_cols = ["vehicle_age_yrs", "vehicle_category", "truck_model", "goods_category",
                          "gross_vehicle_weight_kg", "fleet_size", "years_in_business",
                          "idv_insured_declared_value", "fuel_type", "bs_emission_norm",
                          "permit_type", "ownership_type", "route_type", "highway_usage",
                          "state", "rto_district"]
        df_chk = normalise_columns(df_input)
        missing = [c for c in required_cols if c not in df_chk.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            st.session_state.validation_pending = False
            st.session_state.validation_quit     = False

            with st.spinner("Agent 1 — validating data…"):
                df_full, error_rows, detail_df, log = engine.validate(df_input)

            if error_rows:
                st.session_state.validation_pending     = True
                st.session_state.df_full_for_training    = df_full
                st.session_state.error_rows_for_training = error_rows
                st.session_state.fast_mode_for_training  = fast_mode
            else:
                with st.spinner("Training XGBoost models, scoring risk, pricing…"):
                    engine.train_and_process(df_full, set(), fast_mode=fast_mode)
                st.success("Pipeline complete!")

    if engine.ready:
        st.success("✅ Engine ready")
        st.caption(f"{len(engine.processed_df)} policies processed")
        if engine.excluded_row_count:
            st.caption(f"⚠️ {engine.excluded_row_count} row(s) excluded — see Data Validation Log "
                       f"under the Underwriting tab.")
    elif st.session_state.validation_pending:
        st.warning("⏸️ Waiting for your decision — see the box in the main panel.")
    else:
        st.info("Run the pipeline to unlock all tabs.")

    st.markdown("---")
    st.markdown("### 4. AI Assistant (Gemini API Key)")
    gemini_key = st.text_input("Gemini API key", type="password",
                                help="Get one at https://aistudio.google.com/apikey")
    if gemini_key:
        st.session_state["gemini_key"] = gemini_key

    st.markdown("---")
    st.markdown("### 5. About Model")
    st.markdown(
        "This is a **Commercial Truck Insurance Pricing & Underwriting Model** built by **Team 5** "
        "under the mentorship of the **Sri Sathya Sai Institute of Actuaries Support Team** as an "
        "internship project during the period **May 2025 – June 2025**. "
        "It is a multi-agent model consisting of five agents: "
        "**Data Validation Agent**, **Climate Risk Agent**, **Risk Scoring Agent**, "
        "**Underwriting Agent**, and **Pricing Agent** — with **TruckShield AI** as the AI Assistant."
    )

    st.markdown("---")
    st.markdown("### 6. About Developer Team")
    st.markdown("**Mentor Team**")
    st.markdown(
        "- [Sri Sathya Sai Institute of Actuaries Support Group](https://sssia.org/)\n"
        "- [Satya Sai Mudigonda](https://www.linkedin.com/in/satya-sai-mudigonda)\n"
        "- [Dr. Rohan Yashraj Gupta, FIA, FIAI](https://www.linkedin.com/in/rohanyashraj)"
    )
    st.markdown("**Developer Team (Team 5)**")
    st.markdown(
        "- [Tesa Joby](https://www.linkedin.com/in/tesa-joby-02b28b243) — Risk Analytics & Dataset Expert\n"
        "- [Khyati Bindal](https://www.linkedin.com/in/khyati-bindal) — Climate Analytics & Dataset Expert\n"
        "- [Lakshita Siwach](https://www.linkedin.com/in/lakshita-siwach-952b77303) — Underwriting Analytics & Dataset Expert\n"
        "- [Yashaswi Reddy](https://www.linkedin.com/in/yashaswi-483a68261) — Pricing Analytics & Dataset Expert\n"
        "- [Siva Ruben](https://www.linkedin.com/in/siva-ruban-b80185367) — Underwriting Analytics & Dataset Expert\n"
        "- [Aniket Kadam](https://www.linkedin.com/in/aniket-kadam-1856362a1) — Pricing Analytics & Modelling Expert"
    )

    # ==========================================
    st.markdown("---")
    st.markdown("### 7. Advanced Version")

    st.link_button(
        label="🔗 Open Advanced Version",
        url="https://commercialtruckinsuranceinternship-bvk6aj57kgcvqjejvoimpm.streamlit.app/",
        type="primary",
        use_container_width=True
    )

    st.markdown(
        "⚠️ **Note:** It is strongly advised to free tier Gemini API Key Users for not to use Advanced Version "
        "because it can exhaust your free tier Gemini API Key Limit within a single click.\n\n"
        "**Reason-** In advance version each Agent get there on thinking ability to transfer/calculate the output. "
        "Along with that- few subagents are spawned under the main agent. All this thing is done through the Gemini "
        "resulting it uses your Gemini API Key Limit."
    )

# ─────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────
st.title("🚛 TruckShield AI — Commercial Truck Insurance Pricing and Underwriting Analysis Dashboard")
st.caption("© All Rights Reserved — Sri Sathya Sai Institute of Actuaries (SSSIA) | Team 5")

if st.session_state.validation_pending:
    n_bad   = len(st.session_state.error_rows_for_training)
    n_total = len(st.session_state.df_full_for_training)
    n_good  = n_total - n_bad

    st.warning(
        f"⚠️ **Sorry for interrupting, but there are a total of {n_bad} row(s) that have "
        f"wrong data entry type or missing values.**\n\n"
        f"If you still want to continue, the model will be trained & tested on the "
        f"remaining **{n_good}** row(s). For that, just press **RUN**.\n\n"
        f"If you press **QUIT**, you will be exited from the running process. If you want to "
        f"review the rows with wrong data type entries or missing values, please check the "
        f"**Data Validation Log** under the **Underwriting** option."
    )

    bcol1, bcol2, _ = st.columns([1, 1, 3])
    with bcol1:
        run_anyway = st.button("▶️ RUN", type="primary", use_container_width=True, key="hitl_run")
    with bcol2:
        quit_clicked = st.button("⛔ QUIT", use_container_width=True, key="hitl_quit")

    if run_anyway:
        with st.spinner(f"Training XGBoost models, scoring risk, pricing… ({n_good} valid policies)"):
            engine.train_and_process(
                st.session_state.df_full_for_training,
                st.session_state.error_rows_for_training,
                fast_mode=st.session_state.fast_mode_for_training,
            )
        st.session_state.validation_pending = False
        st.session_state.validation_quit    = False
        st.success(f"Pipeline complete — trained & tested on {n_good} valid polic{'y' if n_good == 1 else 'ies'}.")
        st.rerun()

    if quit_clicked:
        st.session_state.validation_pending = False
        st.session_state.validation_quit    = True
        st.rerun()

    st.stop()

if st.session_state.validation_quit:
    st.error("🚫 Pipeline run cancelled. No policies were trained, scored, or priced.")
    st.subheader("✅ Underwriting — Data Validation Log")
    st.caption("Rows below have a wrong data type entry or a missing value. Fix them in your "
               "source file, then re-upload and run the pipeline again.")
    detail_df = engine.validation_detail_df
    if detail_df is not None and not detail_df.empty:
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
        lines = [f"[{r}, {c}]" for r, c in zip(detail_df["row"], detail_df["column"])]
        st.code("\n".join(lines))
        st.download_button(
            "⬇️ Download validation log (Excel)",
            data=df_to_excel_bytes(detail_df, "Validation Log"),
            file_name="datavalidation_log.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    else:
        st.info("No issues found.")
    st.markdown("---")
    for line in engine.validation_log:
        st.text(line)
    st.stop()

if not engine.ready:
    st.markdown("""
Welcome to **TruckShield AI** — the full dataset is integrated across all agents for pricing & underwriting analysis.

| Stage | Agent | What's in Each Agent Schema|
|---|---|---|
| 1 | **Data Validation** | Validates all dataset columns; uses real claims data for XGBoost targets |
| 2 | **Climate Risk** | Climate zone modifiers + Route Risk Score (route type × highway × destinations) |
| 3 | **Risk Scoring** | Driver Behavior Score; Night Travel Risk score added |
| 4 | **Underwriting** | Claims history loading (count × severity) added to pricing |
| 5 | **Pricing** | 6-field premium split: tp_pure/tp_gross (compulsory) + od_pure/od_gross (optional) + totals |

👈 Pick a data source in the sidebar and click **Run 5-Agent Pipeline**.

👈 To get the access to **AI Assistant- TruckShield AI**, you have to input **Gemini API Key**
""")
    st.stop()

sem        = SEM
pricing_df = engine.pricing_df
risk_df    = engine.risk_df

tab_names = ["💰 Get a Quote", "📊 Portfolio Analytics", "🌧️ Climate & Route Risk",
             "👤 Driver & Travel Risk", "✅ Underwriting", "📊 TP vs OD Analysis", "📥 Export Center"]
if st.session_state.get("gemini_key"):
    tab_names.append("💬 AI Assistant")

tabs = st.tabs(tab_names)

# ════════════════════════════════════════════════════════════
# TAB 0 — GET A QUOTE  
# ════════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Get an instant premium quote")
    st.caption("All fields included — driver profile, climate zone, travel pattern, claims history.")

    if "quote_defaults" not in st.session_state:
        st.session_state.quote_defaults = {}

    st.markdown("### For Existing Policyholder")
    st.caption("Enter Policy ID or Vehicle Registration Number")

    col_search1, col_search2 = st.columns([3, 1])
    search_query = col_search1.text_input(
        "Search Query",
        label_visibility="collapsed",
        placeholder="e.g., CTI-IN-12345678 or MH-12-AB-1234"
    )

    if col_search2.button("CHECK", use_container_width=True, type= "primary"):
        if search_query:
            df_search = engine.full_export_df
            if df_search is not None:
                match = df_search[
                    (df_search["policy_id"].astype(str).str.strip().str.lower() == search_query.strip().lower()) |
                    (df_search["vehicle_registration_number"].astype(
                        str).str.strip().str.lower() == search_query.strip().lower())
                    ]

                if not match.empty:
                    row = match.iloc[0]
                    val_status = str(row.get("data_validation_status", "PROCESSED"))

                    if "EXCLUDED" in val_status:
                        st.error(
                            "You are an existing policyholder in our data but there are some data entry errors in the dataset so I am unable to load your data. Perhaps you can enter your data below in the mentioned option & get an idea of your premium quote.\n\nI am really sorry for your inconvenience.")
                        st.session_state.quote_defaults = {}  
                    else:
                        st.success("Your information found in dataset!")
                        st.session_state.quote_defaults = row.to_dict()
                else:
                    st.error("Invalid Entry! Please Try Again")
                    st.session_state.quote_defaults = {}
            else:
                st.warning("Please run the pipeline first to load the dataset.")
        else:
            st.warning("Please enter a Policy ID or Vehicle Registration Number.")

    st.markdown("---")
    st.markdown("### For New Policyholder")

    defaults = st.session_state.quote_defaults

    def get_def(key, fallback):
        val = defaults.get(key)
        return val if pd.notna(val) else fallback

    def get_str(key, fallback):
        val = defaults.get(key)
        if pd.notna(val) and str(val).strip().lower() not in ["", "nan", "none"]:
            return str(val).strip()
        return fallback

    def get_idx(options, key, fallback_idx=0):
        val = defaults.get(key)
        if pd.notna(val) and val in options:
            return options.index(val)
        return fallback_idx

    with st.form("quote_form"):
        st.markdown("**🚛 Vehicle & Policy**")
        c1, c2, c3 = st.columns(3)
        with c1:
            vehicle_age_yrs = st.number_input("Vehicle age (years)", 0, 30, int(get_def("vehicle_age_yrs", 5)))
            vehicle_category = st.selectbox("Vehicle category", sem["vehicle_categories"],
                                            index=get_idx(sem["vehicle_categories"], "vehicle_category", 0))
            truck_model = st.selectbox("Truck model", sem["truck_models"],
                                       index=get_idx(sem["truck_models"], "truck_model", 0))
            gross_vehicle_weight_kg = st.number_input("Gross vehicle weight (kg)", 500, 60000,
                                                      int(get_def("gross_vehicle_weight_kg", 15000)), step=500)
            fuel_type = st.selectbox("Fuel type", sem["fuel_types"], index=get_idx(sem["fuel_types"], "fuel_type", 0))
            fatigue_monitoring = st.selectbox("Fatigue monitoring system", sem["fatigue_monitor_options"],
                                              index=get_idx(sem["fatigue_monitor_options"], "fatigue_monitoring_system",
                                                            0))
        with c2:
            bs_emission_norm = st.selectbox("BS emission norm", sem["bs_norms"],
                                            index=get_idx(sem["bs_norms"], "bs_emission_norm", 2))
            permit_type = st.selectbox("Permit type", sem["permit_types"],
                                       index=get_idx(sem["permit_types"], "permit_type", 0))
            ownership_type = st.selectbox("Ownership type", sem["ownership_types"],
                                          index=get_idx(sem["ownership_types"], "ownership_type", 0))

            goods_opts = list(sem["cargo_risk_map"].keys())
            goods_category = st.selectbox("Goods category", goods_opts, index=get_idx(goods_opts, "goods_category", 0))

            route_type = st.selectbox("Route type", sem["route_types"],
                                      index=get_idx(sem["route_types"], "route_type", 0))
            overloading_inc = st.number_input("Overloading incidents (last yr)", 0, 10,
                                              int(get_def("overloading_incidents", 0)))
        with c3:
            highway_usage = st.selectbox("Highway usage", sem["highway_usages"],
                                         index=get_idx(sem["highway_usages"], "highway_usage", 0))
            state_opts = sorted(STATES)
            state = st.selectbox("State", state_opts, index=get_idx(state_opts, "state", 0))
            rto_district = st.text_input("RTO district", get_str("rto_district", "Mumbai"))
            climate_zone = st.selectbox("Climate zone", sem["climate_zones"],
                                        index=get_idx(sem["climate_zones"], "climate_zone", 0))
            fleet_size = st.number_input("Fleet size", 1, 500, int(get_def("fleet_size", 5)))
            years_in_biz = st.number_input("Years in business", 0, 100, int(get_def("years_in_business", 5)))
            idv = st.number_input("IDV (₹)", 10000, 50000000, int(get_def("idv_insured_declared_value", 1000000)),
                                  step=50000)

        st.markdown("**👤 Driver Profile**")
        d1, d2, d3, d4 = st.columns(4)
        avg_exp = d1.number_input("Avg driver experience (yrs)", 0.0, 45.0,
                                  float(get_def("average_driver_experience", 8.0)), step=0.5)
        turnover = d2.number_input("Driver turnover rate (% / yr)", 0.0, 100.0,
                                   float(get_def("driver_turnover_rate", 25.0)), step=0.5)
        at_fault = d3.number_input("At-fault accidents (last 3 yr)", 0, 20, int(get_def("at_fault_accidents", 0)))
        traffic_v = d4.number_input("Traffic violations (last 12m)", 0, 50, int(get_def("traffic_violations", 0)))
        dr_training = st.selectbox("Driver training program", sem["driver_training_options"],
                                   index=get_idx(sem["driver_training_options"], "driver_training_program", 0))

        st.markdown("**🗺️ Travel Pattern & Destinations**")
        t1, t2, t3 = st.columns(3)
        night_pct = t1.slider("Night travel frequency (%)", 0, 100, int(get_def("night_travel_frequency", 50)))
        travel_patt = t2.selectbox("Travel time pattern", sem["travel_patterns"],
                                   index=get_idx(sem["travel_patterns"], "travel_time_pattern", 0))
        dest1 = t3.text_input("Top destination 1", get_str("top_destination_1", ""))
        dest_extra = st.columns(2)
        dest2 = dest_extra[0].text_input("Top destination 2", get_str("top_destination_2", ""))
        dest3 = dest_extra[1].text_input("Top destination 3", get_str("top_destination_3", ""))

        st.markdown("**📋 Claims History & NCB**")
        cl1, cl2, cl3 = st.columns(3)
        claims_ct = cl1.number_input("Total claims count (last 3 yr)", 0, 20, int(get_def("total_claims_count", 0)))
        claims_amt = cl2.number_input("Total claim amount paid (₹)", 0, 5000000,
                                      int(get_def("total_claim_amount_paid", 0)), step=10000)
        yrs_since = cl3.number_input(
            "Years since last claim",
            min_value=0.0, max_value=99.0, value=float(get_def("years_since_last_claim", 99.0)), step=0.5,
            help="How many years ago was your most recent claim? Enter 99 if you have never made a claim."
        )

        submitted = st.form_submit_button("🧮 Calculate Premium", type="primary", use_container_width=True)

    if submitted:
        params = dict(
            vehicle_age_yrs=vehicle_age_yrs, vehicle_category=vehicle_category,
            truck_model=truck_model, gross_vehicle_weight_kg=gross_vehicle_weight_kg,
            fuel_type=fuel_type, bs_emission_norm=bs_emission_norm,
            permit_type=permit_type, ownership_type=ownership_type,
            goods_category=goods_category, route_type=route_type,
            highway_usage=highway_usage, state=state, rto_district=rto_district,
            fleet_size=fleet_size, years_in_business=years_in_biz,
            idv_insured_declared_value=idv,
            fatigue_monitoring_system=fatigue_monitoring,
            overloading_incidents=overloading_inc,
            climate_zone=climate_zone,
            average_driver_experience=avg_exp,
            driver_turnover_rate=turnover,
            at_fault_accidents=at_fault,
            traffic_violations=traffic_v,
            driver_training_program=dr_training,
            top_destination_1=dest1, top_destination_2=dest2, top_destination_3=dest3,
            total_claims_count=claims_ct,
            total_claim_amount_paid=claims_amt,
            years_since_last_claim=yrs_since,
            night_travel_frequency=night_pct,
            day_travel_frequency=100 - night_pct,
            travel_time_pattern=travel_patt,
        )
        result = engine.quote_single_policy(params)
        uw  = result["underwriting"]
        pr  = result["pricing"]
        rs  = result["risk_scores"]
        cli = result["climate"]

        decision_color = {"ACCEPT": "🟢", "REFER": "🟡", "DECLINE": "🔴"}[uw["decision"]]
        st.markdown(f"## {decision_color} Underwriting Decision: **{uw['decision']}**")

        if uw["decision"] == "DECLINE":
            st.error("This risk profile falls outside acceptable underwriting criteria "
                     "(high composite risk with an aged vehicle).")
        else:
            st.markdown("### 💰 Total Premium Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Gross Premium (annual)", f"₹{pr['total_gross_premium']:,.0f}",
                      help="What the customer pays: TP tariff + OD gross (after NCB discount)")
            m2.metric("Total Pure Premium",           f"₹{pr['total_pure_premium']:,.0f}",
                      help="Actuarial loss cost: TP pure (70% of tariff) + OD pure (rate × IDV)")
            ncb_label = f"NCB Tier {pr['ncb_tier']} ({pr['ncb_discount_pct']:.0f}% OD discount)" if pr.get('ncb_tier',0)>0 else "NCB: None (Tier 0)"
            m3.metric("NCB Saving (annual)",
                      f"₹{pr.get('ncb_saving', 0):,.0f}",
                      help=ncb_label)
            m4.metric("Policy Limit (IDV)",           f"₹{uw['policy_limit']:,.0f}")

            if uw["decision"] == "REFER":
                st.warning(f"Needs underwriter review — risk loading {uw['extra_loading_pct']:.1f}% "
                           f"+ claims loading {uw['claims_loading_pct']:.1f}%.")

            st.markdown("---")
            st.markdown("### 📋 Premium Breakdown — TP & OD (Paper-Verifiable)")
            denom_disp = (1 - SEM['expenses_loading'] - SEM['profit_margin']
                            - SEM.get('other_loading', 0.0)
                            - uw['extra_loading_pct']/100 - uw['claims_loading_pct']/100)
            _oth_pct = SEM.get('other_loading', 0.0) * 100
            st.caption(
                "**How to verify on paper (4 steps):** "
                "① TP = GVW-slab IRDAI tariff (fixed). "
                "② OD rate = 2.5% × relativity × accident_load, bounded 1.5–9% of IDV. "
                f"Relativity = {rs['composite_risk']:.2f}/5.0 = **{pr['relativity']:.3f}**. "
                f"OD rate = **{pr['od_rate_pct_of_idv']:.4f}%** × IDV. "
                f"③ OD gross (pre-NCB) = OD pure / (1 − {SEM['expenses_loading']*100:.0f}% exp "
                f"− {SEM['profit_margin']*100:.0f}% profit"
                + (f" − {_oth_pct:.0f}% other" if _oth_pct > 0 else "")
                + f" − {uw['extra_loading_pct']:.1f}% UW − {uw['claims_loading_pct']:.1f}% claims)"
                f" = OD pure / {denom_disp:.4f}. "
                f"④ NCB discount = **{pr['ncb_discount_pct']:.0f}%** on OD gross (Tier {pr['ncb_tier']}). "
                f"Total gross = TP gross + OD gross after NCB."
            )

            col_tp, col_od, col_total = st.columns(3)

            with col_tp:
                st.markdown("#### 🔵 Third Party (Compulsory)")
                st.info("Regulated by IRDAI/MoRTH. Rate is fixed by GVW slab. No insurer discretion.")
                st.metric("TP Pure Premium (Compulsory)",
                          f"₹{pr['tp_pure_premium_compulsory']:,.0f}",
                          help=f"= TP Tariff × 70% = ₹{pr['tp_gross_premium_compulsory']:,.0f} × 0.70")
                st.metric("TP Gross Premium (Compulsory)",
                          f"₹{pr['tp_gross_premium_compulsory']:,.0f}",
                          help="Fixed GVW-slab tariff. This is the amount charged to customer for TP cover.")

            with col_od:
                st.markdown("#### 🟠 Own Damage (Optional)")
                st.info("Insurer-priced. Based on IDV, risk score relativity, and accident loading.")
                st.metric("OD Pure Premium (Optional)",
                          f"₹{pr['od_pure_premium_optional']:,.0f}",
                          help=f"= OD Rate ({pr['od_rate_pct_of_idv']:.3f}%) × IDV (₹{params['idv_insured_declared_value']:,.0f})")
                st.metric("OD Gross Premium (Optional)",
                          f"₹{pr['od_gross_premium_optional']:,.0f}",
                          help="= OD Pure / (1 − Expenses − Profit − UW Loadings)")

            with col_total:
                st.markdown("#### 🟢 Total (TP + OD)")
                st.success("Sum of both components. What appears on the policy schedule.")
                st.metric("Total Pure Premium",
                          f"₹{pr['total_pure_premium']:,.0f}",
                          help="= TP Pure + OD Pure (combined actuarial loss cost)")
                st.metric("Total Gross Premium",
                          f"₹{pr['total_gross_premium']:,.0f}",
                          help="= TP Gross + OD Gross (amount customer is charged)")

            if pr.get("ncb_tier", 0) > 0:
                st.success(
                    f"⭐ **NCB Applied: {pr['ncb_discount_pct']:.0f}% OD discount "
                    f"(Tier {pr['ncb_tier']} — {pr['ncb_tier']} consecutive claim-free year"
                    f"{'s' if pr['ncb_tier'] > 1 else ''})** | "
                    f"OD saving: ₹{pr['ncb_saving']:,.0f}"
                )
            else:
                st.info("ℹ️ No NCB applied — claim in last 12 months or new policy. "
                        "Build a clean record to earn up to 50% OD discount.")

            with st.expander("📐 Full pricing stack — how your premium is built step by step"):
                _oth_loading = SEM.get('other_loading', 0.0)
                denom_val = (1 - SEM['expenses_loading'] - SEM['profit_margin']
                               - _oth_loading
                               - uw['extra_loading_pct']/100 - uw['claims_loading_pct']/100)
                _other_row = (f"| Other Loading | {_oth_loading*100:.0f}% |\n"
                              if _oth_loading > 0 else "")
                st.markdown(f"""
**Step 1 — TP Premium (Compulsory)**

| Item | Value |
|---|---|
| GVW-slab IRDAI tariff | ₹{pr['tp_gross_premium_compulsory']:,.0f} |
| TP Pure (tariff × 70%) | ₹{pr['tp_pure_premium_compulsory']:,.0f} |

---

**Step 2 — OD Premium (Optional, before NCB)**

| Item | Value |
|---|---|
| OD Pure Premium | ₹{pr['od_pure_premium_optional']:,.0f} |
| Expenses Loading | {uw['expenses_loading_pct']:.0f}% |
| Profit Margin | {uw['profit_margin_pct']:.0f}% |
{_other_row}| UW Risk Loading | {uw['extra_loading_pct']:.2f}% |
| Claims History Loading | {uw['claims_loading_pct']:.2f}% |
| **Expense Denominator** | **{denom_val:.4f}** |
| **OD Gross (pre-NCB)** | **₹{pr['od_gross_pre_ncb']:,.0f}** |

---

**Step 3 — NCB Discount (OD only)**

| Item | Value |
|---|---|
| NCB Tier | {pr['ncb_tier']} ({pr['ncb_discount_pct']:.0f}% discount) |
| OD Gross pre-NCB | ₹{pr['od_gross_pre_ncb']:,.0f} |
| NCB Saving | −₹{pr['ncb_saving']:,.0f} |
| **OD Gross after NCB** | **₹{pr['od_gross_premium_optional']:,.0f}** |

---

**Step 4 — Total**

| Item | Value |
|---|---|
| TP Gross (compulsory) | ₹{pr['tp_gross_premium_compulsory']:,.0f} |
| OD Gross after NCB | ₹{pr['od_gross_premium_optional']:,.0f} |
| **Total Gross Premium** | **₹{pr['total_gross_premium']:,.0f}** |
""")

        st.markdown("#### Risk score breakdown")
        sc_cols = st.columns(7)
        labels = [
            ("Cargo Risk",        rs["cargo_risk"]),
            ("Exposure",          rs["risk_exposure"]),
            ("Safety (↑=safer)", rs["safety_score"]),
            ("Driver Behavior ↑", rs["driver_behavior_score"]),
            ("Night Travel Risk", rs["night_travel_risk"]),
            ("Accident Prob.",    rs["accident_probability"]),
            ("Composite Risk",   rs["composite_risk"]),
        ]
        for col, (label, val) in zip(sc_cols, labels):
            col.metric(label, f"{val:.2f}/10")

        with st.expander("🌧️ Climate & Route exposure"):
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Flood Risk",   f"{cli['flood']:.1f}/10")
            cc2.metric("Fog Risk",     f"{cli['fog']:.1f}/10")
            cc3.metric("Cyclone Risk", f"{cli['cyclone']:.1f}/10")
            cc4.metric("Route Risk",   f"{cli['route_risk']:.1f}/10")
            st.caption(f"Climate Zone: **{cli['climate_zone']}** | Climate Composite: {cli['climate_composite']:.2f}/10")


        # ══════════════════════════════════════════════════════
        # PDF EXPORT (POLICYHOLDER & UW)
        # ══════════════════════════════════════════════════════
        st.markdown("---")
        st.markdown("### 📥 Download Quote Reports")

        is_existing = bool(st.session_state.quote_defaults)
        pol_id = st.session_state.quote_defaults.get("policy_id", "N/A") if is_existing else "N/A"
        reg_no = st.session_state.quote_defaults.get("vehicle_registration_number", "N/A") if is_existing else "N/A"

        tips = generate_premium_tips(params, result, SEM)

        pdf_policyholder = build_premium_pdf(
            params, result, tips, is_existing, pol_id, reg_no, is_uw=False
        )

        pdf_uw = build_premium_pdf(
            params, result, tips, is_existing, pol_id, reg_no, is_uw=True
        )

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                label="📄 Download PDF (For Policyholder)",
                data=pdf_policyholder,
                file_name=f"Quote_{pol_id}_Policyholder.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )
        with col_dl2:
            st.download_button(
                label="🔒 Download PDF (Underwriting & Actuaries)",
                data=pdf_uw,
                file_name=f"Quote_{pol_id}_Actuarial.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary"
            )

        # ══════════════════════════════════════════════════════
        # PREMIUM REDUCTION TIPS
        # ══════════════════════════════════════════════════════
        if uw["decision"] != "DECLINE":
            st.markdown("---")
            st.markdown("## 💡 How to Reduce Your Premium")
            st.caption(
                "Tips are generated from your specific inputs. Each shows the pricing logic "
                "behind it so you can verify the saving estimate independently."
            )

            tip_tab_tp, tip_tab_od, tip_tab_gen = st.tabs([
                "🔵 TP (Compulsory) Tips",
                "🟠 OD (Optional) Tips",
                "🟢 General / Structural Tips",
            ])

            with tip_tab_tp:
                st.caption("Third Party premium is IRDAI-regulated. Saving opportunities here are limited but exist.")
                for tip in tips["tp"]:
                    with st.expander(f"{tip['icon']} {tip['title']}"):
                        st.markdown(tip["detail"])
                        st.success(f"💰 {tip['saving_note']}")
                        st.caption(f"📐 Pricing logic: {tip['logic']}")

            with tip_tab_od:
                st.caption(
                    f"OD premium (currently ₹{pr['od_gross_premium_optional']:,.0f}) is where the biggest "
                    f"savings are possible. Your OD relativity is **{pr['relativity']:.3f}** "
                    f"(1.0 = average risk). Lowering composite risk score reduces this directly."
                )
                for tip in tips["od"]:
                    with st.expander(f"{tip['icon']} {tip['title']}"):
                        st.markdown(tip["detail"])
                        st.success(f"💰 {tip['saving_note']}")
                        st.caption(f"📐 Pricing logic: {tip['logic']}")

            with tip_tab_gen:
                st.caption("Policy structuring and claims management tips that reduce overall premium.")
                for tip in tips["general"]:
                    with st.expander(f"{tip['icon']} {tip['title']}"):
                        st.markdown(tip["detail"])
                        st.success(f"💰 {tip['saving_note']}")
                        st.caption(f"📐 Pricing logic: {tip['logic']}")

# ════════════════════════════════════════════════════════════
# TAB 1 — PORTFOLIO ANALYTICS
# ════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Portfolio analytics")
    accepted = pricing_df[pricing_df["uw_decision"] != "DECLINE"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Policies",         f"{len(pricing_df):,}")
    k2.metric("Accept Rate",            f"{(pricing_df['uw_decision']=='ACCEPT').mean()*100:.1f}%")
    k3.metric("Mean Total Gross",       f"₹{accepted['total_gross_premium'].mean():,.0f}")
    k4.metric("Mean TP Gross",          f"₹{accepted['tp_gross_premium_compulsory'].mean():,.0f}")
    k5.metric("Mean OD Gross",          f"₹{accepted['od_gross_premium_optional'].mean():,.0f}")

    st.markdown("---")
    available_groups = [c for c in ["vehicle_category", "goods_category", "state", "route_type",
                                     "travel_time_pattern", "fatigue_monitoring_system"]
                        if c in engine.dataset_stats]
    group_choice = st.selectbox("Break down by", available_groups,
                                 format_func=lambda x: x.replace("_", " ").title())
    grp = engine.dataset_stats[group_choice]
    fig = px.bar(grp, x=group_choice, y="mean_total_gross", color="mean_acc_prob",
                 color_continuous_scale="OrRd", text_auto=".2s",
                 labels={"mean_total_gross": "Mean Total Gross Premium (₹)",
                         group_choice: group_choice.replace("_", " ").title(),
                         "mean_acc_prob": "Avg Accident Prob."})
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(grp, use_container_width=True, hide_index=True)

    st.markdown("---")
    cdist1, cdist2 = st.columns(2)
    with cdist1:
        st.markdown("##### Underwriting decision split")
        dec_counts = pricing_df["uw_decision"].value_counts().reset_index()
        dec_counts.columns = ["Decision", "Count"]
        fig2 = px.pie(dec_counts, names="Decision", values="Count", hole=0.45,
                      color="Decision",
                      color_discrete_map={"ACCEPT": "#2e7d32", "REFER": "#f9a825", "DECLINE": "#c62828"})
        st.plotly_chart(fig2, use_container_width=True)
    with cdist2:
        st.markdown("##### Total gross premium distribution")
        fig3 = px.histogram(accepted, x="total_gross_premium", nbins=40,
                             labels={"total_gross_premium": "Total Gross Premium (₹)"})
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("##### Risk score vs. total gross premium")
    fig4 = px.scatter(accepted, x="composite_risk_score", y="total_gross_premium", color="uw_decision",
                       hover_data=["policy_id", "state", "vehicle_category"],
                       color_discrete_map={"ACCEPT": "#2e7d32", "REFER": "#f9a825"},
                       labels={"composite_risk_score": "Composite Risk Score",
                               "total_gross_premium": "Total Gross Premium (₹)"})
    st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 2 — CLIMATE & ROUTE RISK
# ════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Climate & Route risk explorer")

    cc1, cc2 = st.columns([2, 1])
    with cc1:
        st.markdown("##### State-wise climate risk (base methodology)")
        climate_rows = []
        for s, v in sem["climate_risk_base"].items():
            if s == "DEFAULT":
                continue
            climate_rows.append({"State": s, "Flood": v["flood"], "Fog": v["fog"], "Cyclone": v["cyclone"]})
        climate_df_disp = pd.DataFrame(climate_rows).sort_values("Flood", ascending=False)
        fig5 = px.bar(climate_df_disp, x="State", y=["Flood", "Fog", "Cyclone"], barmode="group")
        fig5.update_layout(height=420, xaxis_tickangle=-45)
        st.plotly_chart(fig5, use_container_width=True)

    with cc2:
        st.markdown("##### Live weather lookup")
        st.caption("Open-Meteo — free, no API key required.")
        location = st.text_input("City / location", "Mumbai")
        if st.button("🌦️ Check live weather"):
            with st.spinner("Fetching…"):
                wx = get_live_weather_traffic(location)
            if wx.get("error"):
                st.error(wx["error"])
            else:
                st.write(f"**{wx.get('resolved_name', location)}**")
                st.metric("Condition", wx.get("weather_condition"))
                wcol1, wcol2 = st.columns(2)
                wcol1.metric("Temperature", f"{wx.get('temperature_c')}°C")
                wcol2.metric("Humidity",    f"{wx.get('humidity_pct')}%")
                st.metric("Driving Risk", wx.get("driving_risk"))
                st.caption(f"Traffic: {wx.get('traffic_density')}")

    st.markdown("---")
    st.markdown("##### Climate zone distribution in portfolio")
    if "climate_zone" in risk_df.columns:
        zone_counts = risk_df["climate_zone"].value_counts().reset_index()
        zone_counts.columns = ["Climate Zone", "Count"]
        fig_zone = px.bar(zone_counts, x="Climate Zone", y="Count",
                           color="Count", color_continuous_scale="Blues")
        fig_zone.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig_zone, use_container_width=True)

    st.markdown("##### Route risk score across portfolio")
    if "route_risk_score" in risk_df.columns:
        fig_rr = px.histogram(risk_df, x="route_risk_score", nbins=20,
                               labels={"route_risk_score": "Route Risk Score (0-10)"},
                               color_discrete_sequence=["#e65100"])
        st.plotly_chart(fig_rr, use_container_width=True)

    st.markdown("---")
    st.markdown("##### Accident probability explorer (NCRB / MoRTH-based)")
    ac1, ac2 = st.columns(2)
    with ac1:
        route_rows = [{"Route Type": k, "Accident Score": v} for k, v in sem["accident_prob_route"].items()]
        rdf = pd.DataFrame(route_rows).sort_values("Accident Score", ascending=False)
        fig6 = px.bar(rdf, x="Route Type", y="Accident Score", color="Accident Score",
                      color_continuous_scale="Reds")
        st.plotly_chart(fig6, use_container_width=True)
    with ac2:
        state_rows = [{"State": k, "Accident Score": v} for k, v in sem["accident_prob_state"].items() if k != "DEFAULT"]
        sdf = pd.DataFrame(state_rows).sort_values("Accident Score", ascending=False)
        fig7 = px.bar(sdf, x="State", y="Accident Score", color="Accident Score",
                      color_continuous_scale="Reds")
        fig7.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig7, use_container_width=True)

    st.markdown("##### Accident probability across current portfolio (by state)")
    fig8 = px.box(risk_df, x="state", y="accident_probability", points=False)
    fig8.update_layout(xaxis_tickangle=-45, height=420)
    st.plotly_chart(fig8, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 3 — DRIVER & TRAVEL RISK  
# ════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("Driver & Travel risk analysis")
    st.caption("Explores driver behavior score, night travel risk, and travel pattern distribution.")

    dr1, dr2 = st.columns(2)
    with dr1:
        if "driver_behavior_score" in risk_df.columns:
            st.markdown("##### Driver behavior score distribution")
            fig_db = px.histogram(risk_df, x="driver_behavior_score", nbins=25,
                                   labels={"driver_behavior_score": "Driver Behavior Score (0-10, higher=safer)"},
                                   color_discrete_sequence=["#1565c0"])
            st.plotly_chart(fig_db, use_container_width=True)

    with dr2:
        if "night_travel_risk_score" in risk_df.columns:
            st.markdown("##### Night travel risk score distribution")
            fig_nt = px.histogram(risk_df, x="night_travel_risk_score", nbins=25,
                                   labels={"night_travel_risk_score": "Night Travel Risk Score (0-10)"},
                                   color_discrete_sequence=["#4a148c"])
            st.plotly_chart(fig_nt, use_container_width=True)

    st.markdown("---")
    if "travel_time_pattern" in risk_df.columns:
        st.markdown("##### Travel time pattern vs. accident probability")
        fig_tp = px.box(risk_df, x="travel_time_pattern", y="accident_probability",
                         color="travel_time_pattern",
                         labels={"accident_probability": "Accident Probability",
                                 "travel_time_pattern": "Travel Pattern"})
        st.plotly_chart(fig_tp, use_container_width=True)

    st.markdown("---")
    if "fatigue_monitoring_system" in risk_df.columns:
        st.markdown("##### Fatigue monitoring system vs. safety score")
        fig_fm = px.box(risk_df, x="fatigue_monitoring_system", y="safety_score",
                         color="fatigue_monitoring_system",
                         labels={"safety_score": "Safety Score (0-10)",
                                 "fatigue_monitoring_system": "Fatigue System"})
        st.plotly_chart(fig_fm, use_container_width=True)

    st.markdown("---")
    driver_cols = [c for c in ["policy_id", "state", "vehicle_category",
                                "fatigue_monitoring_system", "overloading_incidents",
                                "average_driver_experience", "driver_turnover_rate",
                                "at_fault_accidents", "traffic_violations",
                                "driver_training_program", "travel_time_pattern",
                                "night_travel_frequency", "driver_behavior_score",
                                "night_travel_risk_score", "accident_probability"] if c in risk_df.columns]
    st.markdown("##### Portfolio detail — driver & travel columns")
    st.dataframe(risk_df[driver_cols].head(200), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════
# TAB 4 — UNDERWRITING
# ════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("Underwriting decisions")

    f1, f2, f3 = st.columns(3)
    decision_filter = f1.multiselect("Decision", ["ACCEPT", "REFER", "DECLINE"],
                                      default=["ACCEPT", "REFER", "DECLINE"])
    state_filter    = f2.multiselect("State", sorted(pricing_df["state"].unique()))
    cargo_filter    = f3.multiselect("Goods category", sorted(pricing_df["goods_category"].unique()))

    filtered = pricing_df[pricing_df["uw_decision"].isin(decision_filter)]
    if state_filter: filtered = filtered[filtered["state"].isin(state_filter)]
    if cargo_filter: filtered = filtered[filtered["goods_category"].isin(cargo_filter)]

    st.caption(f"{len(filtered)} policies match your filters")

    show_cols = [c for c in [
        "policy_id", "state", "vehicle_category", "goods_category", "route_type",
        "climate_zone", "composite_risk_score", "driver_behavior_score",
        "accident_probability", "total_claims_count",
        "uw_decision", "uw_extra_loading", "uw_claims_loading",
        "tp_pure_premium_compulsory", "tp_gross_premium_compulsory",
        "od_pure_premium_optional",   "od_gross_premium_optional",
        "total_pure_premium",         "total_gross_premium",
    ] if c in filtered.columns]
    st.dataframe(filtered[show_cols], use_container_width=True, hide_index=True)

    st.download_button("⬇️ Download filtered results (Excel)",
                        data=df_to_excel_bytes(filtered[show_cols], "Underwriting"),
                        file_name="underwriting_filtered.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("---")
    st.markdown("##### 🧐 Data Validation Log")
    if engine.excluded_row_count:
        st.warning(f"{engine.excluded_row_count} row(s) from the uploaded file had a wrong data "
                   f"type entry or a missing value, and were **excluded** from training/testing "
                   f"and from all scoring & pricing agents. They're listed below by row number and "
                   f"column name, e.g. `[9, vehicle_category]`. You can still see their originally-"
                   f"uploaded values in the Export Center download — every agent-computed column is "
                   f"left blank for them.")
    else:
        st.success("No wrong data type entries or missing values were found in the uploaded data.")

    detail_df = engine.validation_detail_df
    if detail_df is not None and not detail_df.empty:
        st.dataframe(detail_df, use_container_width=True, hide_index=True)
        lines = [f"[{r}, {c}]" for r, c in zip(detail_df["row"], detail_df["column"])]
        st.code("\n".join(lines))
        st.download_button(
            "⬇️ Download validation log (Excel)",
            data=df_to_excel_bytes(detail_df, "Validation Log"),
            file_name="datavalidation_log.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="uw_validation_log_dl",
        )
    with st.expander("Full validation summary"):
        for line in engine.validation_log:
            st.text(line)

# ════════════════════════════════════════════════════════════
# TAB 5 — TP vs OD PREMIUM ANALYSIS  
# ════════════════════════════════════════════════════════════
with tabs[5]:
    st.subheader("TP vs OD Premium Analysis")
    st.caption(
        "Third Party (TP) is compulsory and IRDAI-tariffed (fixed by GVW slab). "
        "Own Damage (OD) is optional and insurer-priced (2.5% base rate of IDV × relativity × accident load)."
    )

    if "total_gross_premium" not in pricing_df.columns:
        st.info("Run the pipeline to view premium data.")
    else:
        accepted = pricing_df[pricing_df["uw_decision"] != "DECLINE"]

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Portfolio Mean Total Gross", f"₹{accepted['total_gross_premium'].mean():,.0f}")
        e2.metric("Mean TP Gross (Compulsory)", f"₹{accepted['tp_gross_premium_compulsory'].mean():,.0f}")
        e3.metric("Mean OD Gross (Optional)",   f"₹{accepted['od_gross_premium_optional'].mean():,.0f}")
        e4.metric("Mean OD Pure (Loss Cost)",   f"₹{accepted['od_pure_premium_optional'].mean():,.0f}")

        st.markdown("---")

        dist1, dist2 = st.columns(2)
        with dist1:
            st.markdown("##### Total gross premium distribution")
            fig_dist = px.histogram(accepted, x="total_gross_premium", nbins=30,
                                    labels={"total_gross_premium": "Total Gross Premium (₹)"},
                                    color_discrete_sequence=["#006064"])
            st.plotly_chart(fig_dist, use_container_width=True)
        with dist2:
            st.markdown("##### OD gross premium distribution")
            fig_od = px.histogram(accepted, x="od_gross_premium_optional", nbins=30,
                                  labels={"od_gross_premium_optional": "OD Gross Premium (₹)"},
                                  color_discrete_sequence=["#e65100"])
            st.plotly_chart(fig_od, use_container_width=True)

        st.markdown("##### TP vs OD gross premium by state")
        if "state" in accepted.columns:
            state_prem = accepted.groupby("state").agg(
                TP_Gross=("tp_gross_premium_compulsory", "mean"),
                OD_Gross=("od_gross_premium_optional",   "mean"),
                Total_Gross=("total_gross_premium",      "mean"),
            ).reset_index().sort_values("Total_Gross", ascending=False)
            fig_state = px.bar(
                state_prem.melt(id_vars="state", value_vars=["TP_Gross", "OD_Gross"],
                                var_name="Component", value_name="Mean Premium"),
                x="state", y="Mean Premium", color="Component", barmode="stack",
                color_discrete_map={"TP_Gross": "#0277bd", "OD_Gross": "#e65100"},
                labels={"Mean Premium": "Mean Premium (₹)", "state": "State"}
            )
            fig_state.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_state, use_container_width=True)

        st.markdown("##### OD pure vs OD gross by goods category")
        if "goods_category" in accepted.columns:
            cat_prem = accepted.groupby("goods_category").agg(
                OD_Pure=("od_pure_premium_optional",  "mean"),
                OD_Gross=("od_gross_premium_optional","mean"),
            ).reset_index().sort_values("OD_Gross", ascending=False)
            fig_cat = px.bar(
                cat_prem.melt(id_vars="goods_category", value_vars=["OD_Pure", "OD_Gross"],
                               var_name="Component", value_name="Mean Premium"),
                x="goods_category", y="Mean Premium", color="Component", barmode="group",
                color_discrete_map={"OD_Pure": "#388e3c", "OD_Gross": "#e65100"},
                labels={"Mean Premium": "Mean Premium (₹)", "goods_category": "Goods Category"}
            )
            st.plotly_chart(fig_cat, use_container_width=True)

# ════════════════════════════════════════════════════════════
# TAB 6 — EXPORT CENTER
# ════════════════════════════════════════════════════════════
with tabs[6]:
    st.subheader("Export center")
    st.caption("Full processed dataset includes all scored columns.")

    st.markdown("##### Full priced portfolio")
    st.download_button("⬇️ Download full dataset (Excel)",
                        data=engine.export_full_dataset_bytes(),
                        file_name="TruckShield_Full_Dataset.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary")

    stats_pdf_data = build_summary_stats_pdf(engine.processed_df, engine.dataset_stats)
    st.download_button("📊 Download Summary Statistic Report (PDF)",
                       data=stats_pdf_data,
                       file_name="Summary_Statistic_Report.pdf",
                       mime="application/pdf",
                       type="primary",
                       use_container_width=False)

    st.markdown("##### Category-wise statistics")
    available_stats = list(engine.dataset_stats.keys())
    stat_choice = st.selectbox("Choose a breakdown",
                                available_stats,
                                format_func=lambda x: x.replace("_", " ").title(),
                                key="export_stat_choice")
    stat_df = engine.dataset_stats[stat_choice]
    st.dataframe(stat_df, use_container_width=True, hide_index=True)
    st.download_button(f"⬇️ Download {stat_choice} stats (Excel)",
                        data=df_to_excel_bytes(stat_df, stat_choice[:30]),
                        file_name=f"stats_{stat_choice}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.markdown("##### Validation log")
    st.code("\n".join(engine.validation_log) or "No issues found.")
    if engine.model_metrics:
        st.caption(f"Model test MAE — Frequency: {engine.model_metrics['freq_mae']:.4f} | "
                   f"Severity: ₹{engine.model_metrics['sev_mae']:,.0f}")

# ════════════════════════════════════════════════════════════
# TAB 7 — AI ASSISTANT  (optional)
# ════════════════════════════════════════════════════════════
if st.session_state.get("gemini_key"):
    with tabs[7]:
        st.subheader("💬 AI Assistant")
        st.caption("Powered by Gemini 2.5 Flash. For exact premium quotes, use the **Get a Quote** tab.")

        chat_col1, chat_col2 = st.columns(2)

        with chat_col1:
            if st.button("Clear chat", type="primary", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        with chat_col2:
            if st.session_state.chat_history:
                pdf_chat_data = build_chat_history_pdf(st.session_state.chat_history)
                st.download_button(
                    label="📥 Download Chat Transcript (PDF)",
                    data=pdf_chat_data,
                    file_name="TruckShield_AI_Chat_Transcript.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    type= "primary"
                )
            else:
                st.button("📥 Download Chat Transcript (PDF Locked)", disabled=True, use_container_width=True, type= "primary")

        st.markdown("---")

        truck_pic = "https://img.icons8.com/color/96/truck.png"  # truck icon URL
        human_pic = "https://img.icons8.com/color/96/user.png"  # user icon URL

        for msg in st.session_state.chat_history:
            avatar_pic = truck_pic if msg["role"] == "assistant" else human_pic
            with st.chat_message(msg["role"], avatar=avatar_pic):
                st.markdown(msg["content"])

        user_msg = st.chat_input("Ask about risk factors, climate zones, driver scoring, or methodology…")

        if user_msg:
            st.session_state.chat_history.append({"role": "user", "content": user_msg})

            with st.chat_message("user", avatar=human_pic):
                st.markdown(user_msg)

            with st.chat_message("assistant", avatar=truck_pic):
                with st.spinner("Thinking…"):
                    try:
                        reply = get_chat_response(engine, user_msg,
                                                  st.session_state["gemini_key"],
                                                  st.session_state.chat_history)
                    except Exception as e:
                        reply = (f"⚠️ Could not reach Gemini ({e}). "
                                 f"All other tabs work without it.")
                    st.markdown(reply)

            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()