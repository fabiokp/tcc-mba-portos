"""
Pipeline completo: Preparação (NB1) + Análise (NB2) — modo headless.

Lê dados ANTAQ de zips anuais, gera séries temporais, features,
índice global, modela com XGBoost, aplica AR(1), detecta anomalias
com ensemble, classifica com dual score, gera fingerprints.

Uso:
    conda activate port_analysis
    python artigo/run_pipeline.py

Outputs:
    artigo/data/processed/  → parquets intermediários (NB1)
    artigo/data/output/     → anomalias, fingerprints, rankings (NB2)
"""

import sys, os, time, zipfile, io, re as re_mod, warnings
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# ── Config ──
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))
from config import *

t0 = time.time()

# Diretório de saída NB2
OUT = ROOT / "data" / "output"
OUT.mkdir(parents=True, exist_ok=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  PARTE 1 — PREPARAÇÃO (NB1)                                ║
# ╚══════════════════════════════════════════════════════════════╝

print("=" * 70)
print("PARTE 1 — PREPARAÇÃO")
print("=" * 70)

# ── 1.1 Carregar ANTAQ dos zips ──

def load_antaq_yearly(file_suffix, zip_dir=ANTAQ_RAW_ZIPS, fallback_dir=ANTAQ_RAW_REAL):
    """Carrega ANTAQ de zips anuais; fallback para .txt extraídos."""
    dfs = []
    zip_files = sorted(zip_dir.glob("[0-9]*.zip")) if zip_dir.exists() else []
    if zip_files:
        for zf_path in zip_files:
            year = zf_path.stem
            target = f"{year}{file_suffix}"
            try:
                with zipfile.ZipFile(zf_path) as zf:
                    candidates = [n for n in zf.namelist()
                                  if n.endswith(target) or n == target]
                    if not candidates:
                        continue
                    with zf.open(candidates[0]) as f:
                        raw_bytes = f.read()
                        for enc in [ANTAQ_ENCODING, "latin-1"]:
                            try:
                                text = raw_bytes.decode(enc)
                                break
                            except UnicodeDecodeError:
                                continue
                        df = pd.read_csv(io.StringIO(text), sep=ANTAQ_SEP,
                                         low_memory=False)
                        dfs.append(df)
            except Exception as e:
                print(f"  AVISO: {zf_path.name}/{target} — {e}")
        if dfs:
            print(f"  → {len(dfs)} zips para *{file_suffix}")
            return pd.concat(dfs, ignore_index=True)

    if fallback_dir.exists():
        all_files = sorted(fallback_dir.glob(f"*{file_suffix}"))
        all_files = [f for f in all_files if re_mod.match(r'^\d{4}', f.name)]
        for f in all_files:
            try:
                df = pd.read_csv(f, sep=ANTAQ_SEP, encoding=ANTAQ_ENCODING,
                                 low_memory=False)
                dfs.append(df)
            except Exception as e:
                print(f"  ERRO: {f.name} — {e}")
        if dfs:
            print(f"  → {len(dfs)} .txt (fallback) para *{file_suffix}")
            return pd.concat(dfs, ignore_index=True)

    raise ValueError(f"Nenhum arquivo para *{file_suffix}")


print("Carregando Atracacao...")
raw_atrac = load_antaq_yearly("Atracacao.txt")
print(f"  {len(raw_atrac):,} registros")

print("Carregando TemposAtracacao...")
raw_tempos = load_antaq_yearly("TemposAtracacao.txt")
print(f"  {len(raw_tempos):,} registros")

print("Carregando Carga...")
raw_carga = load_antaq_yearly("Carga.txt")
print(f"  {len(raw_carga):,} registros")

# ── 1.2 Preparar campos numéricos ──
raw_tempos["t1_horas"] = pd.to_numeric(
    raw_tempos[TEMPOS_T1_COL].astype(str).str.replace(",", "."), errors="coerce")
raw_tempos["tatracado_horas"] = pd.to_numeric(
    raw_tempos[TEMPOS_TATRACADO_COL].astype(str).str.replace(",", "."), errors="coerce")
raw_carga["tonelagem"] = pd.to_numeric(
    raw_carga[CARGA_TONELAGEM_COL].astype(str).str.replace(",", "."), errors="coerce")

# ── 1.3 Merge ──
raw = raw_atrac[[ANTAQ_ID_COL, ANTAQ_PORT_COL, ANTAQ_DATE_COL,
                  ANTAQ_NAV_COL]].copy()
raw = raw.merge(raw_tempos[[ANTAQ_ID_COL, "t1_horas", "tatracado_horas"]],
                on=ANTAQ_ID_COL, how="left")

carga_agg = raw_carga.groupby([ANTAQ_ID_COL, CARGA_SENTIDO_COL])["tonelagem"].sum().reset_index()
carga_exp = carga_agg[carga_agg[CARGA_SENTIDO_COL].isin(EXPORT_VALUES)]
carga_exp = carga_exp.groupby(ANTAQ_ID_COL)["tonelagem"].sum().reset_index()
carga_exp.columns = [ANTAQ_ID_COL, "tonelagem_exp"]
raw = raw.merge(carga_exp, on=ANTAQ_ID_COL, how="left")
raw["tonelagem_exp"] = raw["tonelagem_exp"].fillna(0)

raw = raw.rename(columns={
    ANTAQ_ID_COL: "id_atracacao",
    ANTAQ_PORT_COL: "porto_complexo",
    ANTAQ_DATE_COL: "data_atracacao",
    ANTAQ_NAV_COL: "tipo_navegacao",
})
raw["data_atracacao"] = pd.to_datetime(raw["data_atracacao"], dayfirst=True, errors="coerce")
raw["porto"] = raw["porto_complexo"].map(PORTO_NAMES)

print(f"\nANTAQ merged: {len(raw):,}")
print(f"Período: {raw['data_atracacao'].min().date()} — {raw['data_atracacao'].max().date()}")

# ── 1.4 Filtro Longo Curso + top portos ──
raw_mapped = raw[raw["porto"].notna()].copy()
n_pre = len(raw_mapped)
raw_mapped = raw_mapped[raw_mapped["tipo_navegacao"] == ANTAQ_NAV_LONGO_CURSO].copy()
print(f"Filtro LC: {n_pre:,} → {len(raw_mapped):,}")

ranking = raw_mapped.groupby("porto")["id_atracacao"].count().sort_values(ascending=False)
print(f"\nRanking (top {TOP_N_PORTOS}):")
print(ranking.head(TOP_N_PORTOS))

portos = ranking.head(TOP_N_PORTOS).index.tolist()
df = raw_mapped[raw_mapped["porto"].isin(portos)].copy()

df.loc[df["t1_horas"] < 0, "t1_horas"] = np.nan
df.loc[df["t1_horas"] > T1_OUTLIER_MAX_HORAS, "t1_horas"] = np.nan
df.loc[df["tatracado_horas"] < 0, "tatracado_horas"] = np.nan
df.loc[df["tatracado_horas"] > T1_OUTLIER_MAX_HORAS, "tatracado_horas"] = np.nan
df.loc[df["tonelagem_exp"] < 0, "tonelagem_exp"] = 0

print(f"Selecionados: {len(portos)} portos, {len(df):,} registros")

# ── 1.5 Série diária ──
print("\nConstruindo séries diárias...")
daily_rows = []
for porto in portos:
    dp = df[df["porto"] == porto].copy()
    dp["date"] = dp["data_atracacao"].dt.normalize()
    g = dp.groupby("date").agg(
        atracacoes=("id_atracacao", "count"),
        tonelagem_exp=("tonelagem_exp", "sum"),
        t1_mediano=("t1_horas", "median"),
        tatracado_mediano=("tatracado_horas", "median"),
    ).reset_index()
    g["porto"] = porto

    full_dates = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
    g = g.set_index("date").reindex(full_dates).rename_axis("date").reset_index()
    g["porto"] = porto
    for dim in DIMS:
        if dim in ["atracacoes", "tonelagem_exp"]:
            g[dim] = g[dim].fillna(0)
    daily_rows.append(g)

daily = pd.concat(daily_rows, ignore_index=True)
daily.to_parquet(DATA_PROC / "series_diario.parquet", index=False)
print(f"Série diária: {len(daily):,} ({daily['date'].min().date()} — {daily['date'].max().date()})")

# ── 1.6 Série semanal ──
print("Construindo séries semanais...")
daily["yr"] = daily["date"].dt.isocalendar().year.astype(int)
daily["wk"] = daily["date"].dt.isocalendar().week.astype(int)

weekly_rows = []
for porto in portos:
    dp = daily[daily["porto"] == porto]
    wk = dp.groupby(["yr", "wk"]).agg(
        atracacoes=("atracacoes", "sum"),
        tonelagem_exp=("tonelagem_exp", "sum"),
        t1_mediano=("t1_mediano", "median"),
        tatracado_mediano=("tatracado_mediano", "median"),
        n_dias=("date", "count"),
    ).reset_index()
    # Só semanas completas (≥5 dias)
    wk = wk[wk["n_dias"] >= 5].copy()
    # Descartar última semana se incompleta (< 7 dias) — dados ANTAQ
    # frequentemente ainda não estão consolidados no final da série
    if len(wk) > 0 and wk.iloc[-1]["n_dias"] < 7:
        wk = wk.iloc[:-1].copy()
    wk["date"] = pd.to_datetime(
        wk["yr"].astype(str) + wk["wk"].astype(str).str.zfill(2) + "1",
        format="%G%V%u")
    wk["porto"] = porto
    weekly_rows.append(wk)

weekly = pd.concat(weekly_rows, ignore_index=True).sort_values(["porto", "date"])

# Filtro MIN_DATA_YEAR
weekly = weekly[weekly["date"].dt.year >= MIN_DATA_YEAR].copy()
weekly.to_parquet(DATA_PROC / "series_semanal.parquet", index=False)
print(f"Série semanal: {len(weekly):,} ({weekly['date'].min().date()} — {weekly['date'].max().date()})")

# ── 1.7 Features temporais ──
print("Gerando features temporais...")
feat_rows = []
for porto in portos:
    wp = weekly[weekly["porto"] == porto].sort_values("date").reset_index(drop=True).copy()

    for dim in DIMS:
        for w in MM_WINDOWS:
            wp[f"{dim}_mm{w}"] = wp[dim].rolling(w, min_periods=max(w // 2, 1)).mean()
        for lag in LAG_WEEKS:
            wp[f"{dim}_lag{lag}"] = wp[dim].shift(lag)
        wp[f"{dim}_pct4"] = wp[dim].pct_change(4)
        wp[f"{dim}_std{ROLLING_STD_WINDOW}"] = wp[dim].rolling(
            ROLLING_STD_WINDOW, min_periods=4).std()

    # Calendário
    wp["month_sin"] = np.sin(2 * np.pi * wp["date"].dt.month / 12)
    wp["month_cos"] = np.cos(2 * np.pi * wp["date"].dt.month / 12)
    wp["week_of_year"] = wp["date"].dt.isocalendar().week.astype(int)

    feat_rows.append(wp)

feat = pd.concat(feat_rows, ignore_index=True)

# Substituir inf por NaN (pct_change pode gerar inf quando denominator=0)
feat = feat.replace([np.inf, -np.inf], np.nan)

# STL para dims de volume
for dim in STL_DIMS:
    print(f"  STL {dim}...")
    stl_rows = []
    for porto in portos:
        mask = feat["porto"] == porto
        s = feat.loc[mask, dim].copy()
        s_interp = s.interpolate().bfill().ffill()
        if s_interp.notna().sum() >= 2 * STL_PERIOD:
            try:
                decomp = STL(s_interp, period=STL_PERIOD, robust=True).fit()
                stl_rows.append(pd.DataFrame({
                    f"{dim}_stl_trend": decomp.trend.values,
                    f"{dim}_stl_seasonal": decomp.seasonal.values,
                    f"{dim}_stl_resid": decomp.resid.values,
                }, index=s.index))
            except Exception:
                stl_rows.append(pd.DataFrame({
                    f"{dim}_stl_trend": np.nan,
                    f"{dim}_stl_seasonal": np.nan,
                    f"{dim}_stl_resid": np.nan,
                }, index=s.index))
        else:
            stl_rows.append(pd.DataFrame({
                f"{dim}_stl_trend": np.nan,
                f"{dim}_stl_seasonal": np.nan,
                f"{dim}_stl_resid": np.nan,
            }, index=s.index))
    stl_all = pd.concat(stl_rows)
    for col in stl_all.columns:
        feat[col] = stl_all[col].values

feat.to_parquet(DATA_PROC / "features_semanal.parquet", index=False)
print(f"Features: {feat.shape}")

# ── 1.8 PortWatch + clustering ──
print("\nCarregando PortWatch...")
pw = pd.read_csv(PORTWATCH_FILE, parse_dates=["date"])
pw["date"] = pd.to_datetime(pw["date"], utc=True).dt.tz_localize(None)
print(f"PortWatch: {pw.shape}")


def compute_port_features(pw):
    feats = []
    for pid, g in pw.groupby("portid"):
        total = g["portcalls"]
        m = total.mean()
        if m < 0.5 or len(g) < 365:
            continue
        total_sum = max(total.sum(), 1)
        feats.append({
            "portid": pid, "portname": g["portname"].iloc[0],
            "country": g["country"].iloc[0],
            "media_portcalls": m,
            "cv": total.std() / m if m > 0 else 0,
            "sazonalidade_semanal": (
                g.assign(dow=g["date"].dt.dayofweek)
                .groupby("dow")["portcalls"].mean()
                .pipe(lambda x: (x.max() - x.min()) / m if m > 0 else 0)),
            "pct_container": g["portcalls_container"].sum() / total_sum,
            "pct_dry_bulk": g["portcalls_dry_bulk"].sum() / total_sum,
            "pct_tanker": g["portcalls_tanker"].sum() / total_sum,
            "n_days": len(g),
        })
    return pd.DataFrame(feats)


pf = compute_port_features(pw)
X_cl = pf[CLUSTERING_FEATURES].fillna(pf[CLUSTERING_FEATURES].median())
X_sc = StandardScaler().fit_transform(X_cl)
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
pf["cluster"] = km.fit_predict(X_sc)
pca = PCA(n_components=2).fit(X_sc)
pf["pca1"], pf["pca2"] = pca.transform(X_sc).T
pf.to_csv(DATA_PROC / "portos_clusters.csv", index=False)
print(f"Clusters: {pf['cluster'].value_counts().to_dict()}")

# ── 1.9 Índice global semanal ──
non_br = pf[pf["country"] != "Brazil"]
top50 = non_br.nlargest(TOP_N_GLOBAL_INDEX, "media_portcalls")["portid"].tolist()
pw_g = pw[pw["portid"].isin(top50)].copy()
pw_g["yr"] = pw_g["date"].dt.isocalendar().year.astype(int)
pw_g["wk"] = pw_g["date"].dt.isocalendar().week.astype(int)

gi = pw_g.groupby(["yr", "wk"]).agg(
    gi_portcalls=("portcalls", "mean"),
    n_dias=("date", "nunique"),
).reset_index()
gi["date"] = pd.to_datetime(
    gi["yr"].astype(str) + gi["wk"].astype(str).str.zfill(2) + "1",
    format="%G%V%u")
gi = gi.sort_values("date")
# Descartar última semana se incompleta (< 5 dias) — PortWatch
if len(gi) > 0 and gi.iloc[-1]["n_dias"] < 5:
    gi = gi.iloc[:-1].copy()
gi["gi_mm12"] = gi["gi_portcalls"].rolling(12, min_periods=6).mean()
gi["gi_std12"] = gi["gi_portcalls"].rolling(12, min_periods=6).std()
gi["gi_z"] = (gi["gi_portcalls"] - gi["gi_mm12"]) / gi["gi_std12"].replace(0, np.nan)
gi.to_parquet(DATA_PROC / "indice_global.parquet", index=False)
print(f"Índice global: {len(gi)} semanas")

# ── 1.10 Mapeamento ANTAQ ↔ PortWatch ──
pw_br = pf[pf["country"] == "Brazil"][["portid", "portname"]].copy()
mapa = []
for porto_short, pw_id in ANTAQ_TO_PORTWATCH.items():
    pw_row = pw_br[pw_br["portid"] == pw_id]
    pw_name = pw_row["portname"].values[0] if len(pw_row) > 0 else "NÃO ENCONTRADO"
    mapa.append({"porto_antaq": porto_short, "portid_portwatch": pw_id,
                 "portname_portwatch": pw_name, "no_ranking": porto_short in portos})
pd.DataFrame(mapa).to_csv(DATA_PROC / "mapeamento_portos.csv", index=False)

t1 = time.time()
print(f"\n✅ PARTE 1 concluída em {t1-t0:.0f}s")
for f in sorted(DATA_PROC.glob("*")):
    if f.is_file():
        print(f"  {f.name} ({f.stat().st_size / 1e6:.1f} MB)")


# ╔══════════════════════════════════════════════════════════════╗
# ║  PARTE 2 — ANÁLISE (NB2)                                   ║
# ╚══════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("PARTE 2 — ANÁLISE")
print("=" * 70)

PORTOS = sorted(feat["porto"].unique())
print(f"Portos: {len(PORTOS)}, Features shape: {feat.shape}")

# ── A1: Feature columns e modelos ──
CALENDAR_COLS = [c for c in feat.columns if c in ("month_sin", "month_cos", "week_of_year")]


def get_feat_cols_for(df, target_dim):
    own = [c for c in df.columns
           if c.startswith(target_dim + "_") and "stl" not in c and "resid" not in c]
    return own + CALENDAR_COLS


MODEL_GRID = {
    "naive52":      {"type": "naive", "params": {"lag": 52}},
    "xgb_shallow":  {"type": "xgboost", "params": {"max_depth": 4, "n_estimators": 200, "learning_rate": 0.05}},
    "xgb_deep":     {"type": "xgboost", "params": {"max_depth": 6, "n_estimators": 300, "learning_rate": 0.03}},
    "lgbm_shallow": {"type": "lightgbm", "params": {"max_depth": 4, "n_estimators": 200, "learning_rate": 0.05}},
}


# ── A2: Funções de treino/predição ──

def train_predict(mname, mconf, y, X, tr, pr, dates=None):
    preds = np.full(len(pr), np.nan)
    y_tr = y[tr]
    mtype = mconf.get("type", mname)
    params = mconf.get("params", {})
    if np.isnan(y_tr).mean() > 0.3:
        return preds
    if mtype == "naive":
        lag = params.get("lag", 52)
        for i, idx in enumerate(pr):
            if idx >= lag:
                preds[i] = y[idx - lag]
    elif mtype in ("xgboost", "lightgbm"):
        X_tr, X_pr = X[tr].copy(), X[pr].copy()
        # Substituir inf por NaN antes de treinar
        X_tr[~np.isfinite(X_tr)] = np.nan
        X_pr[~np.isfinite(X_pr)] = np.nan
        ok_tr = ~(np.any(np.isnan(X_tr), 1) | np.isnan(y_tr))
        ok_pr = ~np.any(np.isnan(X_pr), 1)
        if ok_tr.sum() < 20 or ok_pr.sum() == 0:
            return preds
        Cls = XGBRegressor if mtype == "xgboost" else LGBMRegressor
        extra = {"verbosity": 0} if mtype == "xgboost" else {"verbose": -1}
        m = Cls(**params, random_state=42, **extra)
        m.fit(X_tr[ok_tr], y_tr[ok_tr])
        preds[ok_pr] = m.predict(X_pr[ok_pr])
    return preds


def cv_expanding(y, X, dates, mname, mconf):
    n = len(y)
    maes = []
    te = CV_MIN_TRAIN_WEEKS
    while te + CV_VAL_WEEKS <= n:
        tr = np.arange(0, te)
        va = np.arange(te, te + CV_VAL_WEEKS)
        p = train_predict(mname, mconf, y, X, tr, va, dates)
        valid = ~(np.isnan(p) | np.isnan(y[va]))
        if valid.sum() >= 5:
            maes.append(np.mean(np.abs(y[va][valid] - p[valid])))
        te += CV_STEP_WEEKS
    return maes


def walkforward(y, X, dates, mname, mconf):
    n = len(y)
    preds = np.full(n, np.nan)
    te = CV_MIN_TRAIN_WEEKS
    while te < n:
        pe = min(te + WALKFORWARD_RETRAIN_WEEKS, n)
        tr = np.arange(0, te)
        pr = np.arange(te, pe)
        preds[pr] = train_predict(mname, mconf, y, X, tr, pr, dates)
        te = pe
    # Naive fallback
    for i in range(len(preds)):
        if np.isnan(preds[i]) and i >= 52 and not np.isnan(y[i - 52]):
            preds[i] = y[i - 52]
    return preds


# ── A3: Cross-validation ──
print("\nA3: Cross-validation...")
cv_all = []
for porto in PORTOS:
    df_p = feat[feat["porto"] == porto].sort_values("date").reset_index(drop=True)
    dates_p = df_p["date"].values
    for dim in DIMS:
        cols = get_feat_cols_for(df_p, dim)
        X = df_p[cols].values
        y = df_p[dim].values
        for mname, mconf in MODEL_GRID.items():
            maes = cv_expanding(y, X, dates_p, mname, mconf)
            if maes:
                cv_all.append({"porto": porto, "dim": dim, "modelo": mname,
                               "mae_mean": np.mean(maes), "n_folds": len(maes)})
    print(f"  CV {porto}: OK")

cv_df = pd.DataFrame(cv_all)

best_by_dim = []
for dim in DIMS:
    dim_cv = cv_df[cv_df["dim"] == dim]
    if dim_cv.empty:
        best_by_dim.append({"dim": dim, "modelo": "naive52", "mae_mean": np.nan})
        continue
    best = dim_cv.groupby("modelo")["mae_mean"].mean().reset_index().sort_values("mae_mean")
    winner = best.iloc[0]
    naive_mae = best[best["modelo"] == "naive52"]["mae_mean"].values
    naive_mae = naive_mae[0] if len(naive_mae) > 0 else np.inf
    improvement = 1 - winner["mae_mean"] / naive_mae if naive_mae > 0 else 0
    print(f"  {dim:20s}: {winner['modelo']:15s} MAE={winner['mae_mean']:.2f} (↓{improvement:.0%} vs naive)")
    best_by_dim.append({"dim": dim, "modelo": winner["modelo"], "mae_mean": winner["mae_mean"]})

best_by_dim = pd.DataFrame(best_by_dim)

# ── A4: Walk-forward ──
print("\nA4: Walk-forward...")
all_p1 = []
for porto in PORTOS:
    df_p = feat[feat["porto"] == porto].sort_values("date").reset_index(drop=True)
    dates_p = df_p["date"].values
    for dim in DIMS:
        cols = get_feat_cols_for(df_p, dim)
        X = df_p[cols].values
        row = best_by_dim[best_by_dim["dim"] == dim].iloc[0]
        y = df_p[dim].values
        preds = walkforward(y, X, dates_p, row["modelo"], MODEL_GRID[row["modelo"]])
        all_p1.append(pd.DataFrame({
            "date": dates_p, "porto": porto, "dim": dim,
            "y_true": y, "y_pred": preds, "residual": y - preds}))
    print(f"  P1 {porto}: OK")

resid_p1 = pd.concat(all_p1, ignore_index=True)
print(f"Resíduos P1: {len(resid_p1):,}")

# ── A5: Correção AR(1) ──
print("\nA5: Correção AR(1)...")
ar1_coefs = {}
all_corrected = []

for porto in PORTOS:
    for dim in DIMS:
        mask = (resid_p1["porto"] == porto) & (resid_p1["dim"] == dim)
        df_rd = resid_p1.loc[mask].sort_values("date").copy()
        r = df_rd["residual"].values
        r_clean = r[~np.isnan(r)]
        rho = acf(r_clean, nlags=1, fft=True)[1] if len(r_clean) > 52 else 0.0
        ar1_coefs[(porto, dim)] = rho
        r_shifted = np.roll(r, 1)
        r_shifted[0] = np.nan
        df_rd["innovation"] = r - rho * r_shifted
        df_rd["rho_ar1"] = rho
        all_corrected.append(df_rd)

resid = pd.concat(all_corrected, ignore_index=True)

rho_vals = list(ar1_coefs.values())
print(f"  ρ̂ mediano: {np.median(rho_vals):.3f}, médio: {np.mean(rho_vals):.3f}")

# ACF check
acf_before, acf_after = [], []
for porto in PORTOS:
    for dim in DIMS:
        mask = (resid["porto"] == porto) & (resid["dim"] == dim)
        r_orig = resid.loc[mask, "residual"].dropna().values
        r_innov = resid.loc[mask, "innovation"].dropna().values
        if len(r_orig) > 21:
            acf_before.append(acf(r_orig, nlags=1, fft=True)[1])
        if len(r_innov) > 21:
            acf_after.append(acf(r_innov, nlags=1, fft=True)[1])

print(f"  ACF(1) mediana: {np.median(acf_before):.3f} → {np.median(acf_after):.3f}")

# Sobrescrever residual com inovações
resid["residual"] = resid["innovation"]

# Burn-in filter
resid_full = resid.copy()
resid["_date_rank"] = resid.groupby(["porto", "dim"])["date"].rank(method="dense")
resid["is_burnin"] = resid["_date_rank"] <= BURNIN_WEEKS
n_cut = resid["is_burnin"].sum()
resid = resid[~resid["is_burnin"]].copy()
resid.drop(columns=["_date_rank", "is_burnin"], inplace=True)
print(f"  Burn-in: {n_cut:,} removidos, restam {len(resid):,}")
print(f"  Período: {resid['date'].min().date()} — {resid['date'].max().date()}")


# ── C1: Detecção ensemble ──
print("\nC1: Detecção ensemble...")


def detect_mad(r, k=MAD_K):
    med = np.nanmedian(r)
    mad = np.nanmedian(np.abs(r - med))
    return np.abs(r - med) > k * 1.4826 * mad


def detect_stl_r(r, period=STL_PERIOD, threshold=STL_RESID_ZSCORE):
    s = pd.Series(r).interpolate().bfill().ffill()
    if s.notna().sum() < 2 * period:
        return np.full(len(r), False)
    try:
        sr = STL(s, period=period, robust=True).fit().resid.values
        z = np.abs(sr - np.nanmean(sr)) / max(np.nanstd(sr), 1e-10)
        return z > threshold
    except Exception:
        return np.full(len(r), False)


def detect_if(r, contamination=IFOREST_CONTAMINATION):
    s = pd.Series(r)
    X = pd.DataFrame({
        "r": s, "rd": s.diff(),
        "rd2": s - s.rolling(30, min_periods=10).mean()
    }).fillna(0).values
    return IsolationForest(contamination=contamination, random_state=42,
                            n_estimators=200).fit_predict(X) == -1


# CV por porto×dim para threshold adaptativo
cv_by_port = {}
for porto in PORTOS:
    for dim in DIMS:
        mask = (resid["porto"] == porto) & (resid["dim"] == dim)
        r = resid.loc[mask, "innovation"].dropna().values
        if len(r) > 52:
            cv_by_port[(porto, dim)] = np.std(r) / max(abs(np.mean(r)), 1e-10)

cv_values = list(cv_by_port.values())
cv_mediano = np.median(cv_values) if cv_values else 1.0

resid["a_ens"] = False
anomaly_stats = []

for porto in PORTOS:
    for dim in DIMS:
        mask = (resid["porto"] == porto) & (resid["dim"] == dim)
        r = resid.loc[mask, "innovation"].values
        if len(r) < 104 or np.isnan(r).sum() / len(r) > 0.5:
            continue

        cv_val = cv_by_port.get((porto, dim), cv_mediano)
        ratio = cv_val / cv_mediano if cv_mediano > 0 else 1.0
        k_floor = ADAPTIVE_K_FLOORS.get(dim, 2.0)
        k_porto = np.clip(MAD_K * ratio, k_floor, ADAPTIVE_K_MAX)
        if np.isnan(k_porto):
            k_porto = k_floor

        f1 = detect_mad(r, k=k_porto)
        f2 = detect_stl_r(r, threshold=k_porto)
        f3 = detect_if(r)
        agree = f1.astype(int) + f2.astype(int) + f3.astype(int)
        resid.loc[mask, "a_ens"] = agree >= ENSEMBLE_MIN_AGREEMENT

        n_anom = (agree >= ENSEMBLE_MIN_AGREEMENT).sum()
        anomaly_stats.append({
            "porto": porto, "dim": dim, "k_floor": k_floor,
            "k_adaptado": k_porto, "cv_ratio": ratio,
            "n_anomalias": n_anom, "pct": n_anom / max(len(r), 1)
        })

astat = pd.DataFrame(anomaly_stats)
n_anom_total = resid["a_ens"].sum()
print(f"  Anomalias: {n_anom_total:.0f} / {resid['a_ens'].notna().sum()} "
      f"({n_anom_total / resid['a_ens'].notna().sum():.1%})")

for dim in DIMS:
    n = resid[(resid["dim"] == dim) & (resid["a_ens"] == True)].shape[0]
    print(f"    {dim:20s}: {n:4d}")

# ── C1b: Near-misses ──
print("\nC1b: Near-misses...")
near_miss_rows = []
for porto in PORTOS:
    for dim in DIMS:
        mask = (resid["porto"] == porto) & (resid["dim"] == dim)
        r = resid.loc[mask, "innovation"].values
        dates_nm = resid.loc[mask, "date"].values
        if len(r) < 104:
            continue
        med = np.nanmedian(r)
        mad = np.nanmedian(np.abs(r - med))
        mad_scaled = 1.4826 * mad
        if mad_scaled < 1e-10:
            continue
        mad_dist = np.abs(r - med) / mad_scaled
        row_stat = astat[(astat["porto"] == porto) & (astat["dim"] == dim)]
        if len(row_stat) == 0:
            continue
        k_porto = row_stat["k_adaptado"].values[0]
        is_anomaly = resid.loc[mask, "a_ens"].values
        nm_mask = (mad_dist >= 0.8 * k_porto) & (mad_dist < k_porto) & (~is_anomaly)
        for idx in np.where(nm_mask)[0]:
            near_miss_rows.append({
                "porto": porto, "dim": dim, "date": dates_nm[idx],
                "mad_dist": round(float(mad_dist[idx]), 3),
                "k_adaptado": round(float(k_porto), 3),
                "pct_of_k": round(float(mad_dist[idx] / k_porto), 3),
                "innovation": round(float(r[idx]), 4),
            })

near_misses = pd.DataFrame(near_miss_rows)
print(f"  Near-misses: {len(near_misses)}")

# ── C2: Padrões multidimensionais ──
def classify_pattern(dims_set):
    if dims_set >= set(DIMS):
        return "severo"
    if {"atracacoes", "tonelagem_exp"} <= dims_set:
        return "demanda"
    if {"t1_mediano", "tatracado_mediano"} <= dims_set:
        return "gargalo"
    if "t1_mediano" in dims_set and "atracacoes" not in dims_set:
        return "operacional"
    if dims_set == {"tatracado_mediano"}:
        return "berco"
    if len(dims_set) >= 2:
        return "misto"
    return f"isolado_{list(dims_set)[0]}"


ens = resid[resid["a_ens"] == True].copy()
patterns = (ens.groupby(["porto", "date"])["dim"]
    .agg(lambda x: classify_pattern(set(x)))
    .reset_index(name="padrao"))
print(f"\nPadrões multidimensionais:")
print(patterns["padrao"].value_counts().to_string())


# ── D1: Score A (co-ocorrência) ──
print("\nD1: Score A...")


def compute_score_a(ens, window=COOC_WINDOW_WEEKS, same_dim=COOC_SAME_DIM):
    results = []
    for dim in DIMS:
        ens_d = ens[ens["dim"] == dim] if same_dim else ens
        if len(ens_d) == 0:
            continue
        lookup = ens_d.groupby("date")["porto"].apply(set).to_dict()
        for _, row in ens[ens["dim"] == dim].iterrows():
            portos_w = set()
            for dw in range(-window, window + 1):
                d = row["date"] + pd.Timedelta(weeks=dw)
                if d in lookup:
                    portos_w |= lookup[d]
            results.append({
                "porto": row["porto"], "date": row["date"], "dim": dim,
                "residual": row["residual"],
                "innovation": row.get("innovation", row["residual"]),
                "score_a": len(portos_w),
                "portos_co": sorted(portos_w - {row["porto"]}),
            })
    return pd.DataFrame(results)


scores = compute_score_a(ens)
print(f"  Score A calculado: {len(scores)}")

# ── D2: Score B (índice global) ──
print("D2: Score B...")


def add_score_b(scores_df, gi, window=COOC_GI_WINDOW_WEEKS):
    gi_z = gi.set_index("date")["gi_z"] if "gi_z" in gi.columns else gi.set_index("date").iloc[:, 0]
    score_b = []
    for _, row in scores_df.iterrows():
        vals = []
        for dw in range(-window, window + 1):
            d = row["date"] + pd.Timedelta(weeks=dw)
            if d in gi_z.index:
                vals.append(abs(gi_z[d]))
        score_b.append(max(vals) if vals else 0.0)
    scores_df = scores_df.copy()
    scores_df["score_b"] = score_b
    return scores_df


scores = add_score_b(scores, gi)
scores["score_b"] = scores["score_b"].fillna(0.0)
print(f"  Score B adicionado")

# ── D3: Grid search de limiares ──
print("D3: Grid search...")

TIPO_MAP = {"global": "global", "nacional": "nacional",
            "isolado": "isolado", "sazonal": "isolado",
            "exogena": "global", "endogena": "isolado"}

eval_events = []
for ev in KNOWN_EVENTS:
    if "start" not in ev:
        continue
    tipo = TIPO_MAP.get(ev["expected"], "isolado")
    portos_ev = ev.get("portos_foco", PORTOS)
    for p in portos_ev:
        eval_events.append({
            "nome": ev["name"], "porto": p,
            "inicio": ev["start"], "fim": ev["end"], "tipo": tipo})

print(f"  Eventos para validação: {len(eval_events)}")


def classify_tri(scores_df, ta, tb):
    sa = scores_df["score_a"].values
    sb = scores_df["score_b"].values
    out = np.full(len(scores_df), "isolado", dtype=object)
    out[(sa >= ta) & (sb >= tb)] = "global"
    out[(sa >= ta) & (sb < tb)] = "nacional"
    out[(sa < ta) & (sb >= tb)] = "global"
    return out


grid_ta = [3, 4, 5, 6, 7, 8]
grid_tb = [0.8, 1.0, 1.2, 1.5]

results_grid = []
for ta in grid_ta:
    for tb in grid_tb:
        c = classify_tri(scores, ta, tb)
        n_global = (c == "global").sum()
        n_nacional = (c == "nacional").sum()
        n_isolado = (c == "isolado").sum()
        hits = 0
        for ev in eval_events:
            porto_mask = scores["porto"].values == ev["porto"]
            date_mask = ((scores["date"].values >= np.datetime64(ev["inicio"])) &
                         (scores["date"].values <= np.datetime64(ev["fim"])))
            sub_cls = c[porto_mask & date_mask]
            if len(sub_cls) == 0:
                continue
            mode_cls = Counter(sub_cls).most_common(1)[0][0]
            expected = ev["tipo"]
            if expected == "global" and mode_cls == "global":
                hits += 1
            elif expected == "nacional" and mode_cls in ("nacional", "global"):
                hits += 1
            elif expected == "isolado" and mode_cls == "isolado":
                hits += 1
        results_grid.append({
            "ta": ta, "tb": tb,
            "n_global": n_global, "n_nacional": n_nacional, "n_isolado": n_isolado,
            "pct_global": round(100 * n_global / len(c), 1) if len(c) > 0 else 0,
            "hits": hits, "total_events": len(eval_events),
            "recall": round(hits / len(eval_events), 2) if eval_events else 0,
        })

grid_df = pd.DataFrame(results_grid).sort_values(["recall", "n_global"], ascending=[False, False])
best = grid_df.iloc[0]
BEST_TA = int(best["ta"])
BEST_TB = float(best["tb"])
print(f"  Melhor: ta={BEST_TA}, tb={BEST_TB}, recall={best['recall']}")

# ── D4: Classificação final ──
scores["classificacao"] = classify_tri(scores, BEST_TA, BEST_TB)
print(f"\nDistribuição final:")
print(scores["classificacao"].value_counts().to_string())

# ── D5: Validação ──
print("\nValidação com eventos conhecidos:")
for ev in eval_events:
    mask = ((scores["porto"] == ev["porto"]) &
            (scores["date"] >= pd.Timestamp(ev["inicio"])) &
            (scores["date"] <= pd.Timestamp(ev["fim"])))
    sub = scores[mask]
    if len(sub) == 0:
        status = "SEM ANOMALIAS"
        cls = "-"
    else:
        cls = sub["classificacao"].mode().iloc[0]
        status = "✅" if cls == ev["tipo"] else "❌"
    print(f"  {status} {ev['nome'][:25]:25s} {ev['porto']:20s} esp={ev['tipo']:10s} obt={cls}")

# ── D5b: Métricas ──
hits_list = []
for ev in eval_events:
    mask = ((scores["porto"] == ev["porto"]) &
            (scores["date"] >= pd.Timestamp(ev["inicio"])) &
            (scores["date"] <= pd.Timestamp(ev["fim"])))
    sub = scores[mask]
    if len(sub) > 0:
        cls = sub["classificacao"].mode().iloc[0]
        hits_list.append({"acerto": cls == ev["tipo"], "tipo_esperado": ev["tipo"]})

hits_df = pd.DataFrame(hits_list)
precision = hits_df["acerto"].mean() if len(hits_df) > 0 else 0
recall = len(hits_df) / len(eval_events) if eval_events else 0
total_obs = len(resid[resid["residual"].notna()])
taxa_total = len(scores) / total_obs if total_obs > 0 else 0

print(f"\nRecall: {recall:.2f}, Precision: {precision:.2f}")
print(f"Taxa de alertas: {taxa_total:.1%}")


# ── E1: Fingerprints ──
print("\nE1: Fingerprints...")


def extract_fingerprint(scores_df):
    fp_rows = []
    for (porto, date), grp in scores_df.groupby(["porto", "date"]):
        fp = {}
        for _, r in grp.iterrows():
            fp[r["dim"]] = abs(r["innovation"])
        mx = max(fp.values()) if fp else 1.0
        if mx == 0:
            mx = 1.0
        fp_norm = {k: v / mx for k, v in fp.items()}
        for d in DIMS:
            if d not in fp_norm:
                fp_norm[d] = 0.0
        fp_norm.update({"porto": porto, "date": date,
                        "classificacao": grp["classificacao"].iloc[0],
                        "score_a": grp["score_a"].iloc[0],
                        "score_b": grp["score_b"].iloc[0]})
        fp_rows.append(fp_norm)
    return pd.DataFrame(fp_rows)


fingerprints = extract_fingerprint(scores)
print(f"  Fingerprints: {len(fingerprints)}")

# ── E3: Classificador LOO ──
classes_present = fingerprints["classificacao"].nunique()
if classes_present >= 2:
    X_fp = fingerprints[DIMS].values
    y_fp = fingerprints["classificacao"].values
    class_names = sorted(fingerprints["classificacao"].unique())
    loo = LeaveOneOut()
    y_pred = np.empty_like(y_fp)
    for train_idx, test_idx in loo.split(X_fp):
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(X_fp[train_idx], y_fp[train_idx])
        y_pred[test_idx] = clf.predict(X_fp[test_idx])
    print(f"\nClassificador LOO ({classes_present} classes):")
    print(classification_report(y_fp, y_pred, target_names=class_names))

# ── Ranking de vulnerabilidade ──
vuln = scores.groupby("porto").agg(
    total_anomalias=("date", "size"),
    globais=("classificacao", lambda x: (x == "global").sum()),
    nacionais=("classificacao", lambda x: (x == "nacional").sum()),
    isoladas=("classificacao", lambda x: (x == "isolado").sum()),
    score_a_medio=("score_a", "mean"),
    score_b_medio=("score_b", "mean"),
    dims_afetadas=("dim", "nunique"),
    primeiro=("date", "min"),
    ultimo=("date", "max"),
).sort_values("total_anomalias", ascending=False)

vuln["pct_global"] = (100 * vuln["globais"] / vuln["total_anomalias"]).round(1)
vuln["vulnerabilidade"] = (
    vuln["total_anomalias"] * 0.2 + vuln["globais"] * 0.4 +
    vuln["nacionais"] * 0.2 + vuln["dims_afetadas"] * 0.2
).round(2)
vuln = vuln.sort_values("vulnerabilidade", ascending=False)

print("\nRANKING DE VULNERABILIDADE:")
print(vuln[["total_anomalias", "globais", "nacionais", "isoladas",
            "pct_global", "dims_afetadas", "vulnerabilidade"]].to_string())


# ╔══════════════════════════════════════════════════════════════╗
# ║  SALVAR RESULTADOS                                          ║
# ╚══════════════════════════════════════════════════════════════╝

print("\n" + "=" * 70)
print("SALVANDO RESULTADOS")
print("=" * 70)

scores.to_parquet(OUT / "anomalias_classificadas.parquet", index=False)
scores.to_csv(OUT / "anomalias_classificadas.csv", index=False)
print(f"✅ anomalias_classificadas: {len(scores)}")

fingerprints.to_parquet(OUT / "fingerprints.parquet", index=False)
fingerprints.to_csv(OUT / "fingerprints.csv", index=False)
print(f"✅ fingerprints: {len(fingerprints)}")

resid.to_parquet(OUT / "residuos.parquet", index=False)
print(f"✅ residuos: {len(resid)}")

vuln.to_csv(OUT / "ranking_vulnerabilidade.csv")
print(f"✅ ranking_vulnerabilidade: {len(vuln)} portos")

astat.to_csv(OUT / "threshold_adaptativo.csv", index=False)
print(f"✅ threshold_adaptativo: {len(astat)}")

if len(near_misses) > 0:
    near_misses.to_parquet(OUT / "near_misses.parquet", index=False)
    near_misses.to_csv(OUT / "near_misses.csv", index=False)
    print(f"✅ near_misses: {len(near_misses)}")

grid_df.to_csv(OUT / "grid_search_dual_score.csv", index=False)
print(f"✅ grid_search_dual_score: {len(grid_df)}")

# ╔══════════════════════════════════════════════════════════════╗
# ║  PARTE 3 — COMEXSTAT / VULNERABILIDADE ECONÔMICA (NB3)     ║
# ╚══════════════════════════════════════════════════════════════╝

t3 = time.time()
print(f"\n{'=' * 70}")
print("PARTE 3 — COMEXSTAT / VULNERABILIDADE ECONÔMICA")
print("=" * 70)

import urllib.request

AUX = ROOT / "data" / "aux"
AUX.mkdir(parents=True, exist_ok=True)

# ── 3.1 Carregar anomalias NB2 ──
anomalias_nb2 = scores.copy()
anomalias_nb2["date"] = pd.to_datetime(anomalias_nb2["date"])
anomalias_nb2["ym"] = anomalias_nb2["date"].dt.to_period("M").dt.to_timestamp()

anom_mensal = (anomalias_nb2.groupby(["porto", "ym"])
    .agg(n_anomalias=("date", "size"),
         classificacao=("classificacao", lambda x: x.mode().iloc[0]),
         dims=("dim", lambda x: ",".join(sorted(set(x)))))
    .reset_index())

anom_por_porto = anomalias_nb2.groupby("porto").agg(
    n_anomalias=("date", "size"),
    n_global=("classificacao", lambda x: (x == "global").sum()),
    n_nacional=("classificacao", lambda x: (x == "nacional").sum()),
    n_isolado=("classificacao", lambda x: (x == "isolado").sum()),
).reset_index()

print(f"Anomalias mensais: {len(anom_mensal):,} pares porto×mês")

# ── 3.2 Carregar ComexStat ──
csv_files = sorted(COMEXSTAT_RAW.glob("EXP_*.csv"))
assert len(csv_files) > 0, f"ERRO: nenhum EXP_*.csv em {COMEXSTAT_RAW}"
print(f"Arquivos ComexStat: {len(csv_files)}")

frames_comex = []
for f in csv_files:
    year = int(f.stem.split("_")[1])
    if year < MIN_DATA_YEAR:
        continue
    df = pd.read_csv(f, sep=";",
                     dtype={"CO_NCM": str, "CO_URF": str, "CO_PAIS": str,
                            "SG_UF_NCM": str, "CO_VIA": str,
                            "CO_ANO": int, "CO_MES": int})
    frames_comex.append(df)
    print(f"  {f.name}: {len(df):>10,} linhas")

comex_raw = pd.concat(frames_comex, ignore_index=True)
print(f"ComexStat bruto: {len(comex_raw):,} linhas")

# Filtrar marítimo
comex = comex_raw[comex_raw["CO_VIA"] == "01"].copy()
print(f"Após filtro marítimo: {len(comex):,} linhas ({len(comex)/len(comex_raw):.0%})")

# De-para URF → Porto
urf_porto = pd.read_csv(DATA_RAW / "de_para_urf_porto.csv", dtype={"CO_URF": str})
comex["CO_URF"] = comex["CO_URF"].str.zfill(7)
urf_porto["CO_URF"] = urf_porto["CO_URF"].str.zfill(7)
comex = comex.merge(urf_porto[["CO_URF", "porto"]], on="CO_URF", how="inner")

# Filtrar apenas portos do pipeline NB2
portos_pipeline = set(vuln.index)
comex = comex[comex["porto"].isin(portos_pipeline)]

comex["CO_NCM"] = comex["CO_NCM"].astype(str).str.zfill(8)
comex["date"] = pd.to_datetime(
    comex["CO_ANO"].astype(str) + "-" +
    comex["CO_MES"].astype(str).str.zfill(2) + "-01")
print(f"ComexStat final: {len(comex):,} linhas, {comex['porto'].nunique()} portos, "
      f"{comex['date'].min().date()} — {comex['date'].max().date()}")

# ── 3.3 NCM → CUCI ──
urls_cuci = {
    "NCM.csv": "https://balanca.economia.gov.br/balanca/bd/tabelas/NCM.csv",
    "NCM_CUCI.csv": "https://balanca.economia.gov.br/balanca/bd/tabelas/NCM_CUCI.csv",
}
for name, url in urls_cuci.items():
    path = AUX / name
    if not path.exists():
        print(f"  Downloading {name}...")
        urllib.request.urlretrieve(url, path)

ncm_tab = pd.read_csv(AUX / "NCM.csv", sep=";", dtype=str, encoding="latin-1")
cuci_tab = pd.read_csv(AUX / "NCM_CUCI.csv", sep=";", dtype=str, encoding="latin-1")
for tab in [ncm_tab, cuci_tab]:
    for col in tab.columns:
        tab[col] = tab[col].str.strip('"').str.strip()

ncm_cuci_map = ncm_tab[["CO_NCM", "CO_CUCI_ITEM"]].drop_duplicates()
ncm_cuci_map["CO_NCM"] = ncm_cuci_map["CO_NCM"].str.zfill(8)
cuci_grupo = cuci_tab[["CO_CUCI_ITEM", "CO_CUCI_GRUPO", "NO_CUCI_GRUPO",
                        "CO_CUCI_SEC", "NO_CUCI_SEC"]].drop_duplicates()
ncm_grupo = ncm_cuci_map.merge(cuci_grupo, on="CO_CUCI_ITEM", how="left")

comex = comex.merge(
    ncm_grupo[["CO_NCM", "CO_CUCI_GRUPO", "NO_CUCI_GRUPO",
               "CO_CUCI_SEC", "NO_CUCI_SEC"]].drop_duplicates(subset="CO_NCM"),
    on="CO_NCM", how="left")

cob = comex["CO_CUCI_GRUPO"].notna().mean()
cob_fob = comex.loc[comex["CO_CUCI_GRUPO"].notna(), "VL_FOB"].sum() / comex["VL_FOB"].sum()
print(f"Cobertura CUCI: {cob:.1%} das linhas, {cob_fob:.1%} do FOB")

# ── 3.4 Mapeamento de países → regiões ──
pais_path = AUX / "NCM_PAIS.csv"
if not pais_path.exists():
    urllib.request.urlretrieve("https://balanca.economia.gov.br/balanca/bd/tabelas/PAIS.csv", pais_path)
pais_tab = pd.read_csv(pais_path, sep=";", dtype=str, encoding="latin-1")
for col in pais_tab.columns:
    pais_tab[col] = pais_tab[col].str.strip('"').str.strip()

PAIS_REGIAO = {
    "China": "Asia", "Japão": "Asia", "Japan": "Asia",
    "Coreia do Sul": "Asia", "Coréia do Sul": "Asia",
    "Cingapura": "Asia", "Singapura": "Asia",
    "Índia": "Asia", "India": "Asia",
    "Tailândia": "Asia", "Malásia": "Asia",
    "Indonésia": "Asia", "Vietnã": "Asia",
    "Filipinas": "Asia", "Taiwan": "Asia",
    "Hong Kong": "Asia", "Bangladesh": "Asia",
    "Paquistão": "Asia",
    "Emirados Árabes Unidos": "Asia",
    "Arábia Saudita": "Asia",
    "Irã": "Asia", "Irã (República Islâmica)": "Asia",
    "Turquia": "Asia",
    "Países Baixos": "Europe", "Holanda": "Europe",
    "Bélgica": "Europe", "Alemanha": "Europe",
    "Espanha": "Europe", "França": "Europe",
    "Reino Unido": "Europe", "Itália": "Europe",
    "Portugal": "Europe", "Grécia": "Europe",
    "Polônia": "Europe", "Suécia": "Europe",
    "Dinamarca": "Europe", "Noruega": "Europe",
    "Finlândia": "Europe", "Romênia": "Europe",
    "Irlanda": "Europe", "Rússia": "Europe",
    "Federação da Rússia": "Europe",
    "Estados Unidos": "Americas", "Canadá": "Americas",
    "México": "Americas", "Argentina": "Americas",
    "Chile": "Americas", "Colômbia": "Americas",
    "Peru": "Americas", "Paraguai": "Americas",
    "Uruguai": "Americas", "Venezuela": "Americas",
    "Equador": "Americas", "Cuba": "Americas",
    "República Dominicana": "Americas", "Panamá": "Americas",
    "África do Sul": "Africa", "Egito": "Africa",
    "Marrocos": "Africa", "Nigéria": "Africa",
    "Argélia": "Africa", "Gana": "Africa",
    "Quênia": "Africa", "Angola": "Africa",
    "Moçambique": "Africa", "Senegal": "Africa",
    "Tunísia": "Africa", "Tanzânia": "Africa",
}

def map_regiao(co_pais, pais_nomes):
    nome = pais_nomes.get(co_pais, "")
    if not nome:
        return "Other"
    if nome in PAIS_REGIAO:
        return PAIS_REGIAO[nome]
    for pais_key, regiao in PAIS_REGIAO.items():
        if pais_key.lower() in nome.lower() or nome.lower() in pais_key.lower():
            return regiao
    return "Other"

comex_pais = comex[["CO_PAIS"]].drop_duplicates().merge(
    pais_tab[["CO_PAIS", "NO_PAIS"]].drop_duplicates(), on="CO_PAIS", how="left")
pais_nomes_dict = dict(zip(comex_pais["CO_PAIS"], comex_pais["NO_PAIS"].fillna("")))
comex["regiao_destino"] = comex["CO_PAIS"].apply(lambda x: map_regiao(x, pais_nomes_dict))

# ── 3.5 HHI e perfil exportador ──
def hhi(series):
    shares = series / series.sum()
    return (shares ** 2).sum()

# Porto × CUCI grupo → FOB
porto_cuci = (comex[comex["CO_CUCI_GRUPO"].notna()]
    .groupby(["porto", "CO_CUCI_GRUPO", "NO_CUCI_GRUPO"])
    .agg(fob=("VL_FOB", "sum"), kg=("KG_LIQUIDO", "sum"))
    .reset_index()
    .sort_values(["porto", "fob"], ascending=[True, False]))
total_porto = porto_cuci.groupby("porto")["fob"].transform("sum")
porto_cuci["pct"] = porto_cuci["fob"] / total_porto

# HHI pauta por porto
hhi_porto = (comex[comex["CO_CUCI_GRUPO"].notna()]
    .groupby(["porto", "CO_CUCI_GRUPO"])["VL_FOB"].sum()
    .groupby("porto").apply(hhi)
    .rename("hhi_pauta")
    .sort_values(ascending=False)
    .reset_index())
fob_total = comex.groupby("porto")["VL_FOB"].sum().rename("fob_total")
hhi_porto = hhi_porto.merge(fob_total, on="porto")
top1 = (porto_cuci.groupby("porto").first()
    [["CO_CUCI_GRUPO", "NO_CUCI_GRUPO", "pct"]]
    .rename(columns={"CO_CUCI_GRUPO": "top1_cuci",
                     "NO_CUCI_GRUPO": "top1_nome", "pct": "top1_pct"}))
hhi_porto = hhi_porto.merge(top1, on="porto")

# HHI portuário por cadeia
cadeia_porto = (comex[comex["CO_CUCI_GRUPO"].notna()]
    .groupby(["CO_CUCI_GRUPO", "NO_CUCI_GRUPO", "porto"])
    ["VL_FOB"].sum().reset_index())
total_cadeia = cadeia_porto.groupby("CO_CUCI_GRUPO")["VL_FOB"].transform("sum")
cadeia_porto["pct_porto"] = cadeia_porto["VL_FOB"] / total_cadeia

hhi_cadeia = (cadeia_porto.groupby(["CO_CUCI_GRUPO", "NO_CUCI_GRUPO"])
    .apply(lambda g: hhi(g["VL_FOB"]), include_groups=False)
    .rename("hhi_portuario")
    .reset_index()
    .sort_values("hhi_portuario", ascending=False))
fob_cadeia = comex[comex["CO_CUCI_GRUPO"].notna()].groupby("CO_CUCI_GRUPO")["VL_FOB"].sum()
hhi_cadeia = hhi_cadeia.merge(fob_cadeia.rename("fob_total"), on="CO_CUCI_GRUPO")
porto_principal = (cadeia_porto.sort_values("VL_FOB", ascending=False)
    .groupby("CO_CUCI_GRUPO").first()
    [["porto", "pct_porto"]]
    .rename(columns={"porto": "porto_principal", "pct_porto": "pct_principal"}))
hhi_cadeia = hhi_cadeia.merge(porto_principal, on="CO_CUCI_GRUPO")

print(f"Portos HHI: {len(hhi_porto)}, Cadeias HHI: {len(hhi_cadeia)}")

# ── 3.6 Exposição de cadeias a anomalias ──
cuci_portos = (comex[comex["CO_CUCI_GRUPO"].notna()]
    .groupby(["CO_CUCI_GRUPO", "porto"])["VL_FOB"].sum()
    .reset_index().rename(columns={"VL_FOB": "fob_par"}))
n_meses_total = comex["date"].nunique()
anom_set = set(zip(anom_mensal["porto"], anom_mensal["ym"]))

exposicao_cuci = []
for cuci_g in cuci_portos["CO_CUCI_GRUPO"].unique():
    portos_deste = cuci_portos[cuci_portos["CO_CUCI_GRUPO"] == cuci_g]
    fob_total_g = portos_deste["fob_par"].sum()
    meses_afetados = set()
    fob_ponderado = 0
    for _, row in portos_deste.iterrows():
        porto_meses = {ym for (p, ym) in anom_set if p == row["porto"]}
        meses_afetados |= porto_meses
        pct_porto = row["fob_par"] / fob_total_g if fob_total_g > 0 else 0
        fob_ponderado += pct_porto * len(porto_meses)
    exposicao_cuci.append({
        "CO_CUCI_GRUPO": cuci_g,
        "n_portos": len(portos_deste["porto"].unique()),
        "meses_com_anomalia": len(meses_afetados),
        "pct_meses_afetados": len(meses_afetados) / n_meses_total if n_meses_total > 0 else 0,
        "exposicao_ponderada": fob_ponderado / n_meses_total if n_meses_total > 0 else 0,
        "fob_total": fob_total_g,
    })
df_exposicao = pd.DataFrame(exposicao_cuci)
nomes_cuci = (porto_cuci[["CO_CUCI_GRUPO", "NO_CUCI_GRUPO"]]
    .drop_duplicates().set_index("CO_CUCI_GRUPO")["NO_CUCI_GRUPO"])
df_exposicao = df_exposicao.merge(
    nomes_cuci.rename("NO_CUCI_GRUPO"), left_on="CO_CUCI_GRUPO", right_index=True, how="left")
print(f"Exposição CUCI: {len(df_exposicao)} cadeias")

# ── 3.7 Rankings (Seção E do NB3) ──
def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-10)

ranking_porto_comex = hhi_porto[["porto", "hhi_pauta", "fob_total"]].copy()
ranking_porto_comex = ranking_porto_comex.merge(
    anom_por_porto[["porto", "n_anomalias", "n_global", "n_nacional"]],
    on="porto", how="left")
ranking_porto_comex["n_anomalias"] = ranking_porto_comex["n_anomalias"].fillna(0)
ranking_porto_comex["norm_anomalias"] = minmax(ranking_porto_comex["n_anomalias"])
ranking_porto_comex["norm_hhi"] = minmax(ranking_porto_comex["hhi_pauta"])
ranking_porto_comex["norm_fob"] = minmax(ranking_porto_comex["fob_total"])
ranking_porto_comex["score_vuln"] = (
    0.40 * ranking_porto_comex["norm_anomalias"] +
    0.30 * ranking_porto_comex["norm_hhi"] +
    0.30 * ranking_porto_comex["norm_fob"])
ranking_porto_comex = ranking_porto_comex.sort_values("score_vuln", ascending=False)

ranking_cadeia = hhi_cadeia[["CO_CUCI_GRUPO", "NO_CUCI_GRUPO",
                             "hhi_portuario", "fob_total",
                             "porto_principal", "pct_principal"]].copy()
ranking_cadeia = ranking_cadeia.merge(
    df_exposicao[["CO_CUCI_GRUPO", "exposicao_ponderada", "n_portos"]],
    on="CO_CUCI_GRUPO", how="left")
ranking_cadeia = ranking_cadeia[ranking_cadeia["fob_total"] > 5e8]
ranking_cadeia["norm_hhi"] = minmax(ranking_cadeia["hhi_portuario"])
ranking_cadeia["norm_exp"] = minmax(ranking_cadeia["exposicao_ponderada"])
ranking_cadeia["norm_fob"] = minmax(ranking_cadeia["fob_total"])
ranking_cadeia["score_risco"] = (
    0.35 * ranking_cadeia["norm_hhi"] +
    0.35 * ranking_cadeia["norm_exp"] +
    0.30 * ranking_cadeia["norm_fob"])
ranking_cadeia = ranking_cadeia.sort_values("score_risco", ascending=False)

print(f"Porto mais vulnerável: {ranking_porto_comex.iloc[0]['porto']} "
      f"(score={ranking_porto_comex.iloc[0]['score_vuln']:.3f})")
print(f"Cadeia mais em risco: {ranking_cadeia.iloc[0]['CO_CUCI_GRUPO']} "
      f"{str(ranking_cadeia.iloc[0].get('NO_CUCI_GRUPO',''))[:35]} "
      f"(score={ranking_cadeia.iloc[0]['score_risco']:.3f})")

# ── 3.8 FOB Exposto (Seção F do NB3) ──
comex["ym"] = comex["date"]
fob_anomalia = comex.merge(
    anom_mensal[["porto", "ym", "n_anomalias", "classificacao"]],
    on=["porto", "ym"], how="inner")

fob_exp = (fob_anomalia
    .groupby(["porto", "CO_CUCI_GRUPO", "NO_CUCI_GRUPO", "classificacao"])
    .agg(fob_exposto=("VL_FOB", "sum"), meses_expostos=("ym", "nunique"))
    .reset_index())
fob_total_par = (comex.groupby(["porto", "CO_CUCI_GRUPO"])["VL_FOB"]
    .sum().reset_index(name="fob_total"))
fob_exp = fob_exp.merge(fob_total_par, on=["porto", "CO_CUCI_GRUPO"])
fob_exp["pct_exposto"] = fob_exp["fob_exposto"] / fob_exp["fob_total"]

fob_total_exposto = fob_exp["fob_exposto"].sum()
fob_total_geral = comex["VL_FOB"].sum()
print(f"FOB total exposto: US$ {fob_total_exposto/1e9:.1f} bi "
      f"({fob_total_exposto/fob_total_geral:.1%} do FOB total)")

# Rankings FOB
ranking_porto_fob = (fob_exp.groupby("porto")
    .agg(fob_exposto=("fob_exposto", "sum"), n_cadeias=("CO_CUCI_GRUPO", "nunique"))
    .reset_index().sort_values("fob_exposto", ascending=False))
fob_total_porto = comex.groupby("porto")["VL_FOB"].sum().reset_index(name="fob_total")
ranking_porto_fob = ranking_porto_fob.merge(fob_total_porto, on="porto")
ranking_porto_fob["pct_exposto"] = ranking_porto_fob["fob_exposto"] / ranking_porto_fob["fob_total"]
ranking_porto_fob = ranking_porto_fob.merge(hhi_porto[["porto", "hhi_pauta"]], on="porto", how="left")
split_cls = (fob_exp.groupby(["porto", "classificacao"])["fob_exposto"]
    .sum().unstack(fill_value=0).rename(columns=lambda c: f"fob_{c}"))
ranking_porto_fob = ranking_porto_fob.merge(split_cls, on="porto", how="left")

ranking_cadeia_fob = (fob_exp
    .groupby(["CO_CUCI_GRUPO", "NO_CUCI_GRUPO"])
    .agg(fob_exposto=("fob_exposto", "sum"), n_portos=("porto", "nunique"))
    .reset_index().sort_values("fob_exposto", ascending=False))
ranking_cadeia_fob = ranking_cadeia_fob.merge(
    hhi_cadeia[["CO_CUCI_GRUPO", "hhi_portuario", "porto_principal", "pct_principal"]],
    on="CO_CUCI_GRUPO", how="left")

# FOB exposto por UF
fob_uf_anom = (fob_anomalia
    .groupby(["SG_UF_NCM", "classificacao"]).agg(fob_exposto=("VL_FOB", "sum"))
    .reset_index())
fob_uf_total = comex.groupby("SG_UF_NCM")["VL_FOB"].sum().reset_index(name="fob_total")
fob_uf_ranking = (fob_uf_anom.groupby("SG_UF_NCM")["fob_exposto"]
    .sum().reset_index()
    .merge(fob_uf_total, on="SG_UF_NCM")
    .sort_values("fob_exposto", ascending=False))
fob_uf_ranking["pct_exposto"] = fob_uf_ranking["fob_exposto"] / fob_uf_ranking["fob_total"]
hhi_uf_calc = (comex.groupby(["SG_UF_NCM", "porto"])["VL_FOB"]
    .sum().reset_index()
    .groupby("SG_UF_NCM")
    .apply(lambda g: ((g["VL_FOB"] / g["VL_FOB"].sum()) ** 2).sum(), include_groups=False)
    .reset_index(name="hhi_portuario_uf"))
fob_uf_ranking = fob_uf_ranking.merge(hhi_uf_calc, on="SG_UF_NCM", how="left")

# ── 3.9 FOB por cadeia × porto × região ──
FOB_ANO_REF = comex["CO_ANO"].max()
comex_ref = comex[(comex["CO_ANO"] == FOB_ANO_REF) & (comex["regiao_destino"] != "Other")]
print(f"Ano de referência para FOB regional: {FOB_ANO_REF}")

fob_destino = (comex_ref
    .groupby(["porto", "CO_CUCI_GRUPO", "NO_CUCI_GRUPO", "regiao_destino"])
    .agg(fob_regional=("VL_FOB", "sum")).reset_index())
fob_destino = fob_destino.merge(
    hhi_cadeia[["CO_CUCI_GRUPO", "hhi_portuario", "porto_principal", "pct_principal"]],
    on="CO_CUCI_GRUPO", how="left")

fob_cadeia_regiao = (fob_destino
    .groupby(["CO_CUCI_GRUPO", "NO_CUCI_GRUPO", "regiao_destino"])
    .agg(fob_regional=("fob_regional", "sum"), n_portos=("porto", "nunique"))
    .reset_index()
    .merge(hhi_cadeia[["CO_CUCI_GRUPO", "hhi_portuario",
                        "porto_principal", "pct_principal"]],
           on="CO_CUCI_GRUPO", how="left"))

fob_porto_regiao = (comex_ref
    .groupby(["porto", "regiao_destino"]).agg(fob_regional=("VL_FOB", "sum"))
    .reset_index())
total_por_porto_reg = fob_porto_regiao.groupby("porto")["fob_regional"].transform("sum")
fob_porto_regiao["pct_regiao"] = fob_porto_regiao["fob_regional"] / total_por_porto_reg

# Vulnerability matrix (scatter data)
top20_cuci = fob_cadeia.nlargest(20).index.tolist()
vuln_matrix = (cadeia_porto[cadeia_porto["CO_CUCI_GRUPO"].isin(top20_cuci)]
    .merge(hhi_porto[["porto", "hhi_pauta"]], on="porto")
    .merge(hhi_cadeia[["CO_CUCI_GRUPO", "hhi_portuario"]], on="CO_CUCI_GRUPO"))
fob_min = comex["VL_FOB"].sum() * 0.001
vuln_matrix = vuln_matrix[vuln_matrix["VL_FOB"] > fob_min]

# ── 3.10 Salvar outputs NB3 ──
print("\nSalvando outputs NB3...")
porto_cuci.to_csv(OUT / "comex_v2_perfil_cuci_porto.csv", index=False)
hhi_porto.to_csv(OUT / "comex_v2_hhi_porto.csv", index=False)
hhi_cadeia.to_csv(OUT / "comex_v2_hhi_cadeia.csv", index=False)
df_exposicao.to_csv(OUT / "comex_v2_exposicao_cuci.csv", index=False)
ranking_porto_comex.to_csv(OUT / "comex_v2_ranking_portos.csv", index=False)
ranking_cadeia.to_csv(OUT / "comex_v2_ranking_cadeias.csv", index=False)
vuln_matrix.to_csv(OUT / "comex_v2_matriz_vulnerabilidade.csv", index=False)
fob_exp.to_csv(OUT / "comex_v2_fob_exposto.csv", index=False)
ranking_porto_fob.to_csv(OUT / "comex_v2_ranking_portos_fob.csv", index=False)
ranking_cadeia_fob.to_csv(OUT / "comex_v2_ranking_cadeias_fob.csv", index=False)
fob_uf_ranking.to_csv(OUT / "comex_v2_fob_exposto_uf.csv", index=False)
fob_destino.to_csv(OUT / "comex_v2_fob_cadeia_porto_regiao.csv", index=False)
fob_cadeia_regiao.to_csv(OUT / "comex_v2_fob_cadeia_regiao.csv", index=False)
fob_porto_regiao.to_csv(OUT / "comex_v2_fob_porto_regiao.csv", index=False)

t3_end = time.time()
print(f"\n✅ PARTE 3 concluída em {t3_end-t3:.0f}s")
for f in sorted(OUT.glob("comex_v2_*")):
    print(f"  {f.name}: {f.stat().st_size/1024:.0f} KB")

# ── Resumo final ──
t_end = time.time()
print(f"\n{'=' * 70}")
print(f"RESUMO FINAL")
print(f"{'=' * 70}")
print(f"Dimensões: {len(DIMS)} ({DIMS})")
print(f"Total de anomalias: {len(scores)}")
n_gl = (scores['classificacao'] == 'global').sum()
n_na = (scores['classificacao'] == 'nacional').sum()
n_is = (scores['classificacao'] == 'isolado').sum()
print(f"  Global: {n_gl}  |  Nacional: {n_na}  |  Isolado: {n_is}")
print(f"Portos: {scores['porto'].nunique()}")
print(f"Período: {scores['date'].min().date()} → {scores['date'].max().date()}")
print(f"Limiar: ta={BEST_TA}, tb={BEST_TB}")
print(f"Recall: {recall:.2f}, Precision: {precision:.2f}")
print(f"Taxa de alertas: {taxa_total:.1%}")
print(f"\nFOB exposto: US$ {fob_total_exposto/1e9:.1f} bi ({fob_total_exposto/fob_total_geral:.1%})")
print(f"ComexStat: {len(comex):,} linhas, {comex['date'].min().date()} — {comex['date'].max().date()}")
print(f"PortWatch: até {pw['date'].max().date()}")
print(f"\nTempo total: {t_end - t0:.0f}s ({(t_end-t0)/60:.1f} min)")
print(f"\nOutputs em: {OUT}")
print(f"Processed em: {DATA_PROC}")
