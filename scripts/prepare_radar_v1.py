#!/usr/bin/env python
"""
prepare_radar_v1.py — Prepare data for the Radar section of the dashboard

This script implements the plano_radar_v1.md specification:
- Painel 1 (Alerta Origem): Z-scores for BR ports + historical risk profile
- Painel 2 (Alerta Destino): Z-scores for destination countries + exposed chains

Outputs:
- radar_alerta_origem.json → dashboard/data/
- radar_alerta_destino.json → dashboard/data/
"""

import json
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parents[2]  # TCC MBA ENAP
DATA = ROOT / "v3" / "data"
PROCESSED = DATA / "processed"
OUTPUT = DATA / "output"
ALERTS = DATA / "alerts"
RAW = DATA / "raw"
AUX = DATA / "_ux"
DASHBOARD = ROOT / "dashboard" / "data"

# ════════════════════════════════════════════════════════════════════════════
# PORTWATCH API CONFIG
# ════════════════════════════════════════════════════════════════════════════

PORTWATCH_API = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/"
    "Daily_Ports_Data/FeatureServer/0/query"
)
MAX_RECORDS = 1000
RATE_LIMIT_SLEEP = 1.0

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════

ROLLING_WINDOW = 52  # weeks for z-score baseline (captures seasonality)
MIN_PERIODS = 26     # at least half a year of data
Z_ALERT = 2.0        # |z| > 2 → alert
Z_WATCH = 1.5        # |z| > 1.5 → watch

# Top destinations to monitor (by FOB importance to Brazil)
TOP_N_DESTINATIONS = 20

# ════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════════════════

def compute_zscore_rolling(series: pd.Series, window: int = 52, min_periods: int = 26) -> pd.Series:
    """Compute rolling z-score excluding current observation (shift 1)."""
    rmean = series.rolling(window, min_periods=min_periods).mean().shift(1)
    rstd = series.rolling(window, min_periods=min_periods).std().shift(1)
    return (series - rmean) / rstd.replace(0, np.nan)


def classify_alert(z: float) -> str:
    """Classify alert level based on absolute z-score."""
    if pd.isna(z):
        return "unknown"
    az = abs(z)
    if az >= Z_ALERT:
        return "alert"
    if az >= Z_WATCH:
        return "watch"
    return "normal"


def fmt_fob(v):
    """Format FOB value for display."""
    if v >= 1e9:
        return f"US$ {v/1e9:.1f} bi"
    if v >= 1e6:
        return f"US$ {v/1e6:.0f} mi"
    return f"US$ {v:,.0f}"


def fix_double_encoding(s):
    """Fix double-encoded UTF-8 strings (UTF-8 bytes read as latin1, then re-encoded)."""
    if not isinstance(s, str):
        return s
    try:
        return s.encode('latin1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return s


def query_portwatch_api(port_ids: list[str], start_date: str, end_date: str = None) -> pd.DataFrame:
    """
    Query PortWatch API for daily port data by port IDs.
    
    Args:
        port_ids: List of port IDs (e.g., ["port1160", "port885"])
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format (default: today)
        
    Returns:
        DataFrame with columns: portid, portname, country, ISO3, date, portcalls, export, import
    """
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    
    # Build WHERE clause for port IDs
    port_list = ",".join(f"'{p}'" for p in port_ids)
    where = f"portid IN ({port_list}) AND date >= '{start_date}' AND date <= '{end_date}'"
    
    fields = "portid,portname,country,ISO3,date,portcalls,export,import"
    
    all_records = []
    offset = 0
    
    print(f"   Querying PortWatch API: {len(port_ids)} ports, {start_date} → {end_date}")
    
    while True:
        params = {
            "where": where,
            "outFields": fields,
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": MAX_RECORDS,
        }
        
        data = None
        for _attempt in range(3):
            try:
                resp = requests.get(PORTWATCH_API, params=params, timeout=90)
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.RequestException as e:
                wait = (_attempt + 1) * 5
                print(f"   ! API error (attempt {_attempt+1}/3): {e} — retrying in {wait}s")
                time.sleep(wait)
        if data is None:
            print(f"   ! Giving up after 3 attempts")
            break
            
        features = data.get("features", [])
        if not features:
            break
            
        for feat in features:
            all_records.append(feat["attributes"])
            
        print(f"   ... fetched {len(all_records)} records", end="\r")
        
        if len(features) < MAX_RECORDS:
            break
            
        offset += MAX_RECORDS
        time.sleep(RATE_LIMIT_SLEEP)
    
    print(f"   Fetched {len(all_records)} records from PortWatch API")
    
    if not all_records:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    
    # Convert date string to datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    
    return df


def query_portwatch_by_country(iso3_codes: list[str], start_date: str, end_date: str = None, max_ports_per_country: int = 10) -> pd.DataFrame:
    """
    Query PortWatch API for daily port data by country ISO3 codes.
    Fetches only top N ports per country for efficiency.
    
    Args:
        iso3_codes: List of ISO3 country codes (e.g., ["CHN", "USA", "ARG"])
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format (default: today)
        max_ports_per_country: Maximum ports to query per country (default: 10)
        
    Returns:
        DataFrame with columns: portid, portname, country, ISO3, date, portcalls, export, import
    """
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    
    # First, get a list of top ports per country (by recent activity)
    print(f"   Discovering top {max_ports_per_country} ports per country...")
    
    # Query recent data to find most active ports per country
    recent_start = (datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    iso3_list = ",".join(f"'{c}'" for c in iso3_codes)
    
    # Get port activity summary
    params = {
        "where": f"ISO3 IN ({iso3_list}) AND date >= '{recent_start}'",
        "outFields": "portid,ISO3,portcalls",
        "f": "json",
        "resultOffset": 0,
        "resultRecordCount": 2000,
        "groupByFieldsForStatistics": "portid,ISO3",
        "outStatistics": json.dumps([
            {"statisticType": "sum", "onStatisticField": "portcalls", "outStatisticFieldName": "total_calls"}
        ]),
        "orderByFields": "total_calls DESC"
    }
    
    try:
        resp = requests.get(PORTWATCH_API, params=params, timeout=60)
        resp.raise_for_status()
        summary = resp.json()
    except requests.RequestException as e:
        print(f"   ! Error getting port summary: {e}")
        return pd.DataFrame()
        
    if not summary.get("features"):
        print(f"   ! No port summary available")
        return pd.DataFrame()
    
    # Select top N ports per country
    port_counts = {}
    selected_ports = []
    for feat in summary["features"]:
        attr = feat["attributes"]
        iso3 = attr["ISO3"]
        if port_counts.get(iso3, 0) < max_ports_per_country:
            selected_ports.append(attr["portid"])
            port_counts[iso3] = port_counts.get(iso3, 0) + 1
    
    print(f"   Selected {len(selected_ports)} ports from {len(iso3_codes)} countries")
    
    # Query full data for selected ports in batches (to avoid URL length limits)
    BATCH_SIZE = 15  # ports per batch (smaller for reliability)
    fields = "portid,portname,country,ISO3,date,portcalls,export,import"
    all_records = []
    
    for batch_idx in range(0, len(selected_ports), BATCH_SIZE):
        batch_ports = selected_ports[batch_idx:batch_idx + BATCH_SIZE]
        port_list = ",".join(f"'{p}'" for p in batch_ports)
        where = f"portid IN ({port_list}) AND date >= '{start_date}' AND date <= '{end_date}'"
        
        offset = 0
        while True:
            params = {
                "where": where,
                "outFields": fields,
                "f": "json",
                "resultOffset": offset,
                "resultRecordCount": MAX_RECORDS,
            }
            
            data = None
            for attempt in range(3):
                try:
                    resp = requests.get(PORTWATCH_API, params=params, timeout=90)
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except (requests.RequestException, Exception) as e:
                    wait = (attempt + 1) * 3
                    print(f"   ! API error (attempt {attempt+1}/3): {e} — retrying in {wait}s")
                    time.sleep(wait)
            
            if data is None:
                print(f"   ! Skipping batch after 3 failures")
                break
                
            features = data.get("features", [])
            if not features:
                break
                
            for feat in features:
                all_records.append(feat["attributes"])
                
            print(f"   ... fetched {len(all_records)} records (batch {batch_idx // BATCH_SIZE + 1})", end="\r")
            
            if len(features) < MAX_RECORDS:
                break
                
            offset += MAX_RECORDS
            time.sleep(RATE_LIMIT_SLEEP)
        
        time.sleep(RATE_LIMIT_SLEEP * 2)  # longer delay between batches
    
    print(f"   Fetched {len(all_records)} records from PortWatch API{' ' * 20}")
    
    if not all_records:
        return pd.DataFrame()
    
    df = pd.DataFrame(all_records)
    
    # Convert date string to datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    
    return df


# ════════════════════════════════════════════════════════════════════════════
# LOAD BASE DATA
# ════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("RADAR v1 — Prepare Data")
print("=" * 70)

# 1. Port mapping (ANTAQ → PortWatch)
mapeamento = pd.read_csv(PROCESSED / "mapeamento_portos.csv")
# Filter out "NÃO ENCONTRADO" mappings
mapeamento = mapeamento[mapeamento["portid_portwatch"] != "NÃO ENCONTRADO"].copy()
portid_to_antaq = dict(zip(mapeamento["portid_portwatch"], mapeamento["porto_antaq"]))
antaq_to_portid = dict(zip(mapeamento["porto_antaq"], mapeamento["portid_portwatch"]))
print(f"1. Port mapping: {len(mapeamento)} BR ports mapped")

# 2. PortWatch data from API (daily → weekly aggregation)
print("2. Fetching PortWatch data from API...")
# Calculate date range: need ROLLING_WINDOW + MIN_PERIODS to have full z-scores for last 52 weeks
end_date = datetime.now()
start_date = end_date - timedelta(weeks=ROLLING_WINDOW + MIN_PERIODS + 4)  # 52 + 26 + 4 = 82 weeks

# Get all mapped port IDs
port_ids = list(portid_to_antaq.keys())
pw = query_portwatch_api(port_ids, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

if pw.empty:
    print("   ! ERROR: No data from API. Exiting.")
    exit(1)

pw["week"] = pw["date"].dt.to_period("W").dt.start_time
print(f"   Date range: {pw['date'].min().date()} → {pw['date'].max().date()}")
print(f"   Total ports: {pw['portid'].nunique()}")

# 3. Historical anomalies and fingerprints
anomalias = pd.read_parquet(OUTPUT / "anomalias_classificadas.parquet")
fingerprints = pd.read_parquet(OUTPUT / "fingerprints.parquet")
print(f"3. Historical anomalies: {len(anomalias)} records, {anomalias['porto'].nunique()} ports")

# 4. Vulnerability data
hhi_porto = pd.read_csv(OUTPUT / "comex_v2_hhi_porto.csv")
ranking_portos = pd.read_csv(OUTPUT / "comex_v2_ranking_portos.csv")
perfil_cuci_porto = pd.read_csv(OUTPUT / "comex_v2_perfil_cuci_porto.csv")
uf_cuci_porto = pd.read_csv(OUTPUT / "comex_v2_uf_cuci_porto.csv")
hhi_cadeia = pd.read_csv(OUTPUT / "comex_v2_hhi_cadeia.csv")
fob_cadeia_porto_regiao = pd.read_csv(OUTPUT / "comex_v2_fob_cadeia_porto_regiao.csv")
print(f"4. Vulnerability data loaded")

# 5. Country mapping (ComexStat CO_PAIS → ISO3)
pais_map = pd.read_csv(AUX / "NCM_PAIS.csv", sep=";", encoding="latin1")
pais_map = pais_map[["CO_PAIS", "CO_PAIS_ISOA3", "NO_PAIS", "NO_PAIS_ING"]].copy()
pais_map.columns = ["co_pais", "iso3", "nome_pt", "nome_en"]
pais_map["co_pais"] = pais_map["co_pais"].astype(str)
print(f"5. Country mapping: {len(pais_map)} countries")

# 6. Build country-level FOB by chain × port (EXP_2025 → NCM→CUCI, URF→porto, CO_PAIS→ISO3)
print("\n6. Building country-level FOB by chain × port...")
_exp25 = pd.read_csv(
    RAW / "comexstat" / "EXP_2025.csv", sep=";",
    dtype={"CO_NCM": str, "CO_URF": str, "CO_PAIS": str, "CO_VIA": str},
    usecols=["CO_NCM", "CO_PAIS", "CO_VIA", "CO_URF", "VL_FOB"],
)
_exp25 = _exp25[_exp25["CO_VIA"] == "01"]  # maritime only
print(f"   EXP_2025 maritime: {len(_exp25):,} rows")

# NCM → CUCI Grupo
_ncm_tab = pd.read_csv(AUX / "NCM.csv", sep=";", dtype=str, encoding="latin-1")
for col in _ncm_tab.columns:
    _ncm_tab[col] = _ncm_tab[col].str.strip('"').str.strip()
_ncm_cuci_map = _ncm_tab[["CO_NCM", "CO_CUCI_ITEM"]].drop_duplicates()
_ncm_cuci_map["CO_NCM"] = _ncm_cuci_map["CO_NCM"].str.zfill(8)

_cuci_tab = pd.read_csv(AUX / "NCM_CUCI.csv", sep=";", dtype=str, encoding="latin-1")
for col in _cuci_tab.columns:
    _cuci_tab[col] = _cuci_tab[col].str.strip('"').str.strip()
_cuci_grupo = _cuci_tab[["CO_CUCI_ITEM", "CO_CUCI_GRUPO", "NO_CUCI_GRUPO"]].drop_duplicates()
_ncm_grupo = _ncm_cuci_map.merge(_cuci_grupo, on="CO_CUCI_ITEM", how="left")

_exp25 = _exp25.merge(
    _ncm_grupo[["CO_NCM", "CO_CUCI_GRUPO", "NO_CUCI_GRUPO"]].drop_duplicates(subset="CO_NCM"),
    on="CO_NCM", how="left",
)

# URF → porto
_urf_porto = pd.read_csv(RAW / "de_para_urf_porto.csv", dtype={"CO_URF": str})
_urf_porto["CO_URF"] = _urf_porto["CO_URF"].str.zfill(7)
_exp25["CO_URF"] = _exp25["CO_URF"].str.zfill(7)
_exp25 = _exp25.merge(_urf_porto[["CO_URF", "porto"]], on="CO_URF", how="inner")

# CO_PAIS → ISO3
_exp25["CO_PAIS"] = _exp25["CO_PAIS"].astype(str)
_exp25 = _exp25.merge(
    pais_map[["co_pais", "iso3"]].drop_duplicates(),
    left_on="CO_PAIS", right_on="co_pais", how="left",
)

# Aggregate: FOB by (iso3, CUCI_GRUPO, porto)
fob_pais_cadeia_porto = (
    _exp25[_exp25["iso3"].notna() & _exp25["CO_CUCI_GRUPO"].notna()]
    .groupby(["iso3", "CO_CUCI_GRUPO", "NO_CUCI_GRUPO", "porto"])
    .agg(fob=("VL_FOB", "sum"))
    .reset_index()
)
print(f"   FOB by country × chain × port: {len(fob_pais_cadeia_porto):,} rows")
print(f"   Countries: {fob_pais_cadeia_porto['iso3'].nunique()}, Chains: {fob_pais_cadeia_porto['CO_CUCI_GRUPO'].nunique()}")
del _exp25, _ncm_tab, _ncm_cuci_map, _cuci_tab, _cuci_grupo, _ncm_grupo, _urf_porto


# ════════════════════════════════════════════════════════════════════════════
# PAINEL 1 — ALERTA ORIGEM (BR PORTS)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PAINEL 1 — ALERTA ORIGEM (Portos BR)")
print("=" * 70)

# 1a. Filter PortWatch to mapped BR ports
br_portids = list(mapeamento["portid_portwatch"])
pw_br = pw[pw["portid"].isin(br_portids)].copy()
print(f"\n1a. BR ports in PortWatch: {pw_br['portid'].nunique()} / {len(br_portids)} mapped")

# 1b. Weekly aggregation for BR ports
pw_br_weekly = (
    pw_br.groupby(["portid", "week"])
    .agg(
        export_vol=("export", "sum"),
        portcalls=("portcalls", "sum"),
        n_dias=("date", "nunique"),
    )
    .reset_index()
    .sort_values(["portid", "week"])
)
# Drop the last week per port if it is incomplete (< 7 days — PortWatch has daily data incl. weekends)
_last_wk = pw_br_weekly.groupby("portid")["week"].transform("max")
_incomplete = (pw_br_weekly["week"] == _last_wk) & (pw_br_weekly["n_dias"] < 7)
if _incomplete.any():
    print(f"   ⚠ Dropping incomplete last week for {_incomplete.sum()} port(s)")
pw_br_weekly = pw_br_weekly[~_incomplete].copy()
print(f"1b. Weekly series: {len(pw_br_weekly)} rows")

# 1c. Compute z-scores
z_results = []
for portid, grp in pw_br_weekly.groupby("portid"):
    grp = grp.copy().sort_values("week")
    grp["z_export"] = compute_zscore_rolling(grp["export_vol"], ROLLING_WINDOW, MIN_PERIODS)
    grp["z_portcalls"] = compute_zscore_rolling(grp["portcalls"], ROLLING_WINDOW, MIN_PERIODS)
    grp["porto_antaq"] = portid_to_antaq.get(portid, portid)
    z_results.append(grp)

pw_br_z = pd.concat(z_results, ignore_index=True)
print(f"1c. Z-scores computed")

# 1d. Get latest week status per port
latest_week = pw_br_z["week"].max()
pw_latest = pw_br_z[pw_br_z["week"] == latest_week].copy()
pw_latest["status"] = pw_latest["z_export"].apply(classify_alert)
pw_latest["direction"] = pw_latest["z_export"].apply(lambda z: "positive" if z > 0 else "negative" if z < 0 else "neutral")

print(f"\n   Latest week: {latest_week.date()}")
print(f"   Ports with alert: {(pw_latest['status'] == 'alert').sum()}")
print(f"   Ports with watch: {(pw_latest['status'] == 'watch').sum()}")

# 1e. Compile historical risk profile per port
def get_port_profile(porto):
    """Compile historical profile for a port."""
    # Anomaly stats
    port_anom = anomalias[anomalias["porto"] == porto]
    n_anomalias = len(port_anom)
    pct_global = (port_anom["classificacao"] == "global").mean() if n_anomalias > 0 else 0
    pct_nacional = (port_anom["classificacao"] == "nacional").mean() if n_anomalias > 0 else 0
    pct_isolado = (port_anom["classificacao"] == "isolado").mean() if n_anomalias > 0 else 0
    
    # Fingerprint (average intensity per dimension)
    port_fp = fingerprints[fingerprints["porto"] == porto]
    fp_dims = {}
    for dim in ["atracacoes", "tonelagem_exp", "t1_mediano", "tatracado_mediano"]:
        if dim in port_fp.columns:
            # Normalize to 0-1 scale (intensity values are already proportions)
            fp_dims[dim] = port_fp[dim].mean() if len(port_fp) > 0 else 0
    
    # HHI and vulnerability
    port_hhi = hhi_porto[hhi_porto["porto"] == porto]
    hhi_val = port_hhi["hhi_pauta"].iloc[0] if len(port_hhi) > 0 else np.nan
    fob_total = port_hhi["fob_total"].iloc[0] if len(port_hhi) > 0 else 0
    
    # Top chains (UF + CUCI group) - format: "Produto - UF"
    port_uf = uf_cuci_porto[uf_cuci_porto["porto"] == porto].nlargest(5, "fob")
    # Calculate total FOB for this port to get percentages
    port_total_fob = port_uf["fob"].sum() if len(port_uf) > 0 else 1
    top_chains = [
        {
            "uf": row["SG_UF_NCM"],
            "cuci": row["CO_CUCI_GRUPO"],
            "nome": fix_double_encoding(row["NO_CUCI_GRUPO"]),
            "fob": int(row["fob"]),
            "pct": row["fob"] / port_total_fob if port_total_fob > 0 else 0,
        }
        for _, row in port_uf.iterrows()
    ]
    
    return {
        "n_anomalias": int(n_anomalias),
        "pct_global": round(pct_global * 100, 1),
        "pct_nacional": round(pct_nacional * 100, 1),
        "pct_isolado": round(pct_isolado * 100, 1),
        "fingerprint": {k: round(v, 3) for k, v in fp_dims.items()},
        "hhi_pauta": round(hhi_val, 4) if not pd.isna(hhi_val) else None,
        "fob_total": int(fob_total),
        "top_chains": top_chains,
    }


# Build full alert data for each port
alerta_origem = []
for _, row in pw_latest.iterrows():
    porto = row["porto_antaq"]
    profile = get_port_profile(porto)
    
    # Timeline (last 52 weeks)
    port_ts = pw_br_z[pw_br_z["porto_antaq"] == porto].tail(52)
    timeline = [
        {
            "week": r["week"].strftime("%Y-%m-%d"),
            "export_vol": float(r["export_vol"]) if not pd.isna(r["export_vol"]) else None,
            "z_export": round(r["z_export"], 2) if not pd.isna(r["z_export"]) else None,
        }
        for _, r in port_ts.iterrows()
    ]
    
    alerta_origem.append({
        "porto": porto,
        "portid_pw": row["portid"],
        "week": latest_week.strftime("%Y-%m-%d"),
        "export_vol": float(row["export_vol"]),
        "z_export": round(row["z_export"], 2) if not pd.isna(row["z_export"]) else None,
        "z_portcalls": round(row["z_portcalls"], 2) if not pd.isna(row["z_portcalls"]) else None,
        "status": row["status"],
        "direction": row["direction"],
        **profile,
        "timeline": timeline,
    })

# Normalize fingerprint values across all ports so max per dimension = 1.0
fp_dims_list = ["atracacoes", "tonelagem_exp", "t1_mediano", "tatracado_mediano"]
fp_maxes = {dim: max((a["fingerprint"].get(dim, 0) for a in alerta_origem), default=1) for dim in fp_dims_list}
for a in alerta_origem:
    for dim in fp_dims_list:
        raw = a["fingerprint"].get(dim, 0)
        max_val = fp_maxes[dim] if fp_maxes[dim] > 0 else 1
        a["fingerprint"][dim] = round(raw / max_val, 3)

# Sort by absolute z-score (most anomalous first)
alerta_origem.sort(key=lambda x: abs(x["z_export"]) if x["z_export"] else 0, reverse=True)

print(f"\n1e. Risk profiles compiled for {len(alerta_origem)} ports")
for a in alerta_origem[:5]:
    z = a["z_export"]
    print(f"   {a['status'].upper():6s} {a['porto']:20s} z={z:+.2f}" if z else f"   {a['status'].upper():6s} {a['porto']:20s} z=N/A")


# ════════════════════════════════════════════════════════════════════════════
# PAINEL 2 — ALERTA DESTINO (DESTINATION COUNTRIES)
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PAINEL 2 — ALERTA DESTINO (Países de Destino)")
print("=" * 70)

# 2a. Load ComexStat 2025 to get top destinations by FOB
print("\n2a. Loading ComexStat 2025 for destination analysis...")
comex_2025 = pd.read_csv(RAW / "comexstat" / "EXP_2025.csv", sep=";", usecols=["CO_PAIS", "VL_FOB"], dtype={"CO_PAIS": str})
fob_by_country = comex_2025.groupby("CO_PAIS")["VL_FOB"].sum().reset_index()
fob_by_country = fob_by_country.merge(pais_map, left_on="CO_PAIS", right_on="co_pais", how="left")
fob_by_country = fob_by_country.dropna(subset=["iso3"])
fob_by_country = fob_by_country.sort_values("VL_FOB", ascending=False)
fob_total = fob_by_country["VL_FOB"].sum()
fob_by_country["pct_export"] = (fob_by_country["VL_FOB"] / fob_total * 100).round(2)

top_destinations = fob_by_country.head(TOP_N_DESTINATIONS).copy()
print(f"   Top {TOP_N_DESTINATIONS} destinations by FOB (2025):")
for _, r in top_destinations.iterrows():
    print(f"     {r['iso3']} {r['nome_pt'][:20]:20s} {fmt_fob(r['VL_FOB']):>14s} ({r['pct_export']:.1f}%)")

# 2b. Get PortWatch data for destination countries from API
dest_iso3 = list(top_destinations["iso3"])
print(f"\n2b. Fetching PortWatch data for destination countries...")
pw_dest = query_portwatch_by_country(dest_iso3, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), max_ports_per_country=5)

if pw_dest.empty:
    print("   ! WARNING: No data for destination countries")
else:
    pw_dest["week"] = pw_dest["date"].dt.to_period("W").dt.start_time
    print(f"   PortWatch data for destinations: {pw_dest['portid'].nunique()} ports in {pw_dest['ISO3'].nunique()} countries")

# 2c. Weekly aggregation by country (sum of portcalls and import volume across top ports)
pw_dest_weekly = (
    pw_dest.groupby(["ISO3", "week"])
    .agg(
        portcalls=("portcalls", "sum"),
        import_vol=("import", "sum"),
        n_ports=("portid", "nunique"),
        n_dias=("date", "nunique"),
    )
    .reset_index()
    .sort_values(["ISO3", "week"])
)
# Drop the last week per country if it is incomplete (< 7 days — PortWatch has daily data incl. weekends)
_last_wk_d = pw_dest_weekly.groupby("ISO3")["week"].transform("max")
_incomplete_d = (pw_dest_weekly["week"] == _last_wk_d) & (pw_dest_weekly["n_dias"] < 7)
if _incomplete_d.any():
    print(f"   ⚠ Dropping incomplete last week for {_incomplete_d.sum()} country/ies")
pw_dest_weekly = pw_dest_weekly[~_incomplete_d].copy()

# 2d. Compute z-scores per country
z_dest_results = []
for iso3, grp in pw_dest_weekly.groupby("ISO3"):
    grp = grp.copy().sort_values("week")
    grp["z_portcalls"] = compute_zscore_rolling(grp["portcalls"], ROLLING_WINDOW, MIN_PERIODS)
    grp["z_import"] = compute_zscore_rolling(grp["import_vol"], ROLLING_WINDOW, MIN_PERIODS)
    z_dest_results.append(grp)

pw_dest_z = pd.concat(z_dest_results, ignore_index=True)
print(f"2c-d. Z-scores computed for {pw_dest_z['ISO3'].nunique()} countries")

# 2e. Latest week status per country
dest_latest = pw_dest_z[pw_dest_z["week"] == latest_week].copy()
dest_latest["status"] = dest_latest["z_portcalls"].apply(classify_alert)
dest_latest["direction"] = dest_latest["z_portcalls"].apply(lambda z: "positive" if z > 0 else "negative" if z < 0 else "neutral")

# Merge with FOB data
dest_latest = dest_latest.merge(
    top_destinations[["iso3", "nome_pt", "nome_en", "VL_FOB", "pct_export"]],
    left_on="ISO3",
    right_on="iso3",
    how="left",
)

print(f"\n   Destinations with alert: {(dest_latest['status'] == 'alert').sum()}")
print(f"   Destinations with watch: {(dest_latest['status'] == 'watch').sum()}")

# 2f. For each destination in alert/watch, identify exposed BR chains
def get_exposed_chains(iso3):
    """Get Brazilian chains exposed to a specific destination country.
    
    Uses country-level FOB data (EXP_2025) instead of regional aggregation,
    so each country shows its own export profile.
    """
    country_data = fob_pais_cadeia_porto[fob_pais_cadeia_porto["iso3"] == iso3]
    if len(country_data) == 0:
        return []
    
    # For each chain (CUCI): total FOB to this country + principal port
    chain_agg = (
        country_data.groupby(["CO_CUCI_GRUPO", "NO_CUCI_GRUPO"])
        .apply(lambda g: pd.Series({
            "fob_regional": g["fob"].sum(),
            "porto_principal": g.loc[g["fob"].idxmax(), "porto"],
        }))
        .reset_index()
    )
    
    # Top chains by FOB to this specific country
    chain_agg = chain_agg.sort_values("fob_regional", ascending=False).head(10)
    
    # Add HHI from vulnerability data (optional enrichment)
    chain_agg = chain_agg.merge(
        hhi_cadeia[["CO_CUCI_GRUPO", "hhi_portuario"]].drop_duplicates(),
        on="CO_CUCI_GRUPO", how="left",
    )
    
    return [
        {
            "cuci": row["CO_CUCI_GRUPO"],
            "nome": fix_double_encoding(row["NO_CUCI_GRUPO"]),
            "fob_regional": int(row["fob_regional"]),
            "hhi_portuario": round(row["hhi_portuario"], 3) if not pd.isna(row.get("hhi_portuario")) else None,
            "porto_principal": row["porto_principal"],
        }
        for _, row in chain_agg.iterrows()
    ]


# Build alert data for destinations
alerta_destino = []
for _, row in dest_latest.iterrows():
    iso3 = row["ISO3"]
    nome = row.get("nome_pt", iso3)
    
    # Timeline (last 52 weeks)
    dest_ts = pw_dest_z[pw_dest_z["ISO3"] == iso3].tail(52)
    timeline = [
        {
            "week": r["week"].strftime("%Y-%m-%d"),
            "portcalls": int(r["portcalls"]),
            "z_portcalls": round(r["z_portcalls"], 2) if not pd.isna(r["z_portcalls"]) else None,
        }
        for _, r in dest_ts.iterrows()
    ]
    
    # Exposed chains
    exposed = get_exposed_chains(iso3)
    
    alerta_destino.append({
        "iso3": iso3,
        "nome": nome,
        "nome_en": row.get("nome_en", nome),
        "week": latest_week.strftime("%Y-%m-%d"),
        "portcalls": int(row["portcalls"]),
        "n_ports": int(row["n_ports"]),
        "z_portcalls": round(row["z_portcalls"], 2) if not pd.isna(row["z_portcalls"]) else None,
        "z_import": round(row["z_import"], 2) if not pd.isna(row["z_import"]) else None,
        "status": row["status"],
        "direction": row["direction"],
        "fob_total": int(row["VL_FOB"]) if not pd.isna(row["VL_FOB"]) else 0,
        "pct_export": float(row["pct_export"]) if not pd.isna(row["pct_export"]) else 0,
        "exposed_chains": exposed,
        "timeline": timeline,
    })

# Sort by FOB importance
alerta_destino.sort(key=lambda x: x["fob_total"], reverse=True)

print(f"\n2f. Alert data compiled for {len(alerta_destino)} destinations")
for a in alerta_destino[:5]:
    z = a["z_portcalls"]
    print(f"   {a['status'].upper():6s} {a['nome'][:20]:20s} z={z:+.2f} FOB={fmt_fob(a['fob_total'])}" if z else f"   {a['status']}")


# ════════════════════════════════════════════════════════════════════════════
# EXPORT
# ════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXPORT")
print("=" * 70)

# Summary stats
n_br_alert = sum(1 for a in alerta_origem if a["status"] == "alert")
n_br_watch = sum(1 for a in alerta_origem if a["status"] == "watch")
n_dest_alert = sum(1 for a in alerta_destino if a["status"] == "alert")
n_dest_watch = sum(1 for a in alerta_destino if a["status"] == "watch")

# Combined summary
radar_summary = {
    "week_reference": latest_week.strftime("%Y-%m-%d"),
    "portwatch_latest": pw["date"].max().strftime("%Y-%m-%d"),
    "br_ports_total": len(alerta_origem),
    "br_ports_alert": n_br_alert,
    "br_ports_watch": n_br_watch,
    "destinations_total": len(alerta_destino),
    "destinations_alert": n_dest_alert,
    "destinations_watch": n_dest_watch,
}

# Painel 1 output
output_origem = {
    "summary": radar_summary,
    "alerts": alerta_origem,
}
out_path_origem = DASHBOARD / "radar_alerta_origem.json"
with open(out_path_origem, "w", encoding="utf-8") as f:
    json.dump(output_origem, f, indent=2, ensure_ascii=False, default=str)
print(f"\n✓ {out_path_origem.name}: {out_path_origem.stat().st_size:,} bytes")

# Painel 2 output
output_destino = {
    "summary": radar_summary,
    "alerts": alerta_destino,
}
out_path_destino = DASHBOARD / "radar_alerta_destino.json"
with open(out_path_destino, "w", encoding="utf-8") as f:
    json.dump(output_destino, f, indent=2, ensure_ascii=False, default=str)
print(f"✓ {out_path_destino.name}: {out_path_destino.stat().st_size:,} bytes")

# Also save to alerts folder
for src, name in [(output_origem, "radar_alerta_origem.json"), (output_destino, "radar_alerta_destino.json")]:
    with open(ALERTS / name, "w", encoding="utf-8") as f:
        json.dump(src, f, indent=2, ensure_ascii=False, default=str)

print("\n" + "=" * 70)
print("RADAR v1 — Complete!")
print("=" * 70)
print(f"""
Summary:
  Week reference: {radar_summary['week_reference']}
  BR ports:       {radar_summary['br_ports_total']} total, {radar_summary['br_ports_alert']} alert, {radar_summary['br_ports_watch']} watch
  Destinations:   {radar_summary['destinations_total']} total, {radar_summary['destinations_alert']} alert, {radar_summary['destinations_watch']} watch
""")
