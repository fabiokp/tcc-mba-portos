"""
Prepara dados leves (JSON) para o dashboard Quarto/OJS v2.
Lê parquets do pipeline artigo + CSVs NB3 (ComexStat) e gera JSONs.

Uso:
    cd artigo/dashboard
    python data/prepare_data.py
"""
import pandas as pd
import json
from pathlib import Path
import sys, os

# ── Paths ──────────────────────────────────────────────
DASH = Path(__file__).parent                               # dashboard/data/
ROOT = DASH.parent.parent / "data"                         # artigo/data/
OUT  = ROOT / "output"                                     # artigo/data/output/
COMEXSTAT_RAW = DASH.parent.parent / "data" / "raw" / "comexstat"

print(f"ROOT = {ROOT.resolve()}")
print(f"OUT  = {OUT.resolve()}")
print(f"DASH = {DASH.resolve()}")

# ── 1. Séries semanais (18 portos × 4 dims × ~620 semanas) ──
feat = pd.read_parquet(ROOT / "processed" / "features_semanal.parquet")
series = feat[["date", "porto", "atracacoes", "tonelagem_exp",
               "t1_mediano", "tatracado_mediano"]].copy()
series["date"] = series["date"].dt.strftime("%Y-%m-%d")
series.to_json(DASH / "series.json", orient="records")
print(f"  series.json          → {len(series):,} linhas")

# ── 2. Anomalias classificadas ──
scores = pd.read_parquet(OUT / "anomalias_classificadas.parquet")
scores["date"] = scores["date"].dt.strftime("%Y-%m-%d")
if "portos_co" in scores.columns:
    scores["portos_co"] = scores["portos_co"].astype(str)
scores.to_json(DASH / "anomalias.json", orient="records")
print(f"  anomalias.json       → {len(scores):,} linhas")

# ── 3. Resíduos (obs vs pred) ──
resid = pd.read_parquet(OUT / "residuos.parquet")
resid["date"] = resid["date"].dt.strftime("%Y-%m-%d")
resid.to_json(DASH / "residuos.json", orient="records")
print(f"  residuos.json        → {len(resid):,} linhas")

# ── 4. Fingerprints ──
fp = pd.read_parquet(OUT / "fingerprints.parquet")
fp["date"] = fp["date"].dt.strftime("%Y-%m-%d")
fp.to_json(DASH / "fingerprints.json", orient="records")
print(f"  fingerprints.json    → {len(fp):,} linhas")

# ── 5. Índice global ──
gi = pd.read_parquet(ROOT / "processed" / "indice_global.parquet")
gi["date"] = gi["date"].dt.strftime("%Y-%m-%d")
gi.to_json(DASH / "indice_global.json", orient="records")
print(f"  indice_global.json   → {len(gi):,} linhas")

# ── 6. Ranking de vulnerabilidade ──
vuln = pd.read_csv(OUT / "ranking_vulnerabilidade.csv")
vuln.to_json(DASH / "vulnerabilidade.json", orient="records")
print(f"  vulnerabilidade.json → {len(vuln):,} linhas")

# ── 7. Thresholds adaptativos ──
astat = pd.read_csv(OUT / "threshold_adaptativo.csv")
astat.to_json(DASH / "thresholds.json", orient="records")
print(f"  thresholds.json      → {len(astat):,} linhas")

# ── 8. Metadados dos portos (lat, lon, volume médio) ──
port_meta = feat.groupby("porto").agg(
    atracacoes_media=("atracacoes", "mean"),
    tonelagem_media=("tonelagem_exp", "mean"),
).reset_index()

coords = {
    "Santos":                  (-23.95, -46.30),
    "Paranaguá":               (-25.50, -48.51),
    "Rio Grande":              (-32.05, -52.10),
    "Itaguaí":                 (-22.90, -43.80),
    "São Luís":                (-2.50,  -44.28),
    "Suape":                   (-8.39,  -34.95),
    "Vila do Conde":           (-1.55,  -48.75),
    "Itajaí":                  (-26.91, -48.67),
    "Rio de Janeiro":          (-22.89, -43.17),
    "Salvador":                (-12.97, -38.52),
    "Manaus":                  (-3.15,  -59.98),
    "Pecém":                   (-3.53,  -38.81),
    "Imbituba":                (-28.23, -48.66),
    "São Francisco do Sul":    (-26.24, -48.64),
    "Vitória":                 (-20.32, -40.29),
    "São Sebastião":           (-23.81, -45.41),
    "Angra dos Reis":          (-23.01, -44.32),
    "São João da Barra":       (-21.64, -41.05),
    "Porto Alegre":            (-30.03, -51.23),
    "Barra do Riacho":         (-19.83, -40.07),
}
port_meta["lat"] = port_meta["porto"].map(lambda p: coords.get(p, (0, 0))[0])
port_meta["lon"] = port_meta["porto"].map(lambda p: coords.get(p, (0, 0))[1])
port_meta.to_json(DASH / "portos_meta.json", orient="records")
print(f"  portos_meta.json     → {len(port_meta):,} linhas")

# ══════════════════════════════════════════════════════
# ── ComexStat JSONs (Telas 7, 8) ──
# ══════════════════════════════════════════════════════

def fix_encoding(s):
    """Fix double-encoded UTF-8-as-latin-1 strings from MDIC tables."""
    if pd.isna(s) or not isinstance(s, str):
        return s
    try:
        return s.encode('latin-1').decode('utf-8')
    except (UnicodeDecodeError, UnicodeEncodeError):
        return s

# ── 9. Perfil de produto por porto (Grupo CUCI) ──
try:
    cprod = pd.read_csv(OUT / "comex_v2_perfil_cuci_porto.csv")
    cprod.rename(columns={
        "CO_CUCI_GRUPO": "cuci_grupo",
        "NO_CUCI_GRUPO": "cuci_nome",
        "fob": "fob_total",
        "kg":  "kg_total",
    }, inplace=True)
    cprod["cuci_nome"] = cprod["cuci_nome"].apply(fix_encoding)
    # Top 10 por porto
    cprod_top = (cprod.sort_values("fob_total", ascending=False)
        .groupby("porto").head(10).reset_index(drop=True))
    # Calcular % do total do porto
    porto_totals = cprod.groupby("porto")["fob_total"].sum().rename("porto_total")
    cprod_top = cprod_top.merge(porto_totals, on="porto")
    cprod_top["pct"] = cprod_top["fob_total"] / cprod_top["porto_total"]
    cprod_top.to_json(DASH / "comex_perfil_produto.json", orient="records")
    print(f"  comex_perfil_produto.json → {len(cprod_top):,} linhas (CUCI)")
except Exception as e:
    print(f"  ⚠️ comex_perfil_produto.json FALHOU: {e}")

# ── 10. HHI por UF (rich — com porto_principal, fob_total, n_portos) ──
try:
    hhi = pd.read_csv(OUT / "comex_v2_hhi_uf_rich.csv")
    hhi["porto_principal"] = hhi["porto_principal"].apply(fix_encoding)
    hhi.to_json(DASH / "comex_hhi.json", orient="records")
    print(f"  comex_hhi.json       → {len(hhi):,} linhas (rich)")
except Exception as e:
    print(f"  ⚠️ comex_hhi.json FALHOU: {e}")

# ── 11. Vulnerabilidade economica (UF × Porto) ──
try:
    vuln_ec = pd.read_csv(OUT / "comex_vulnerabilidade_economica.csv")
    vuln_ec.to_json(DASH / "comex_vulnerabilidade.json", orient="records")
    print(f"  comex_vulnerabilidade.json → {len(vuln_ec):,} linhas")
except Exception as e:
    print(f"  ⚠️ comex_vulnerabilidade.json FALHOU: {e}")

# ── 12. Ranking UF exposição ──
try:
    rank_uf = pd.read_csv(OUT / "comex_ranking_uf_exposicao.csv")
    if rank_uf.columns[0] not in ["SG_UF_NCM", "UF"]:
        rank_uf = rank_uf.rename(columns={rank_uf.columns[0]: "SG_UF_NCM"})
    rank_uf.to_json(DASH / "comex_ranking_uf.json", orient="records")
    print(f"  comex_ranking_uf.json → {len(rank_uf):,} linhas")
except Exception as e:
    print(f"  ⚠️ comex_ranking_uf.json FALHOU: {e}")

# ── 13. Perfil UF por porto (para tela 7 lado direito) ──
try:
    cpuf = pd.read_csv(OUT / "comex_perfil_uf.csv")
    cpuf_top = (cpuf.sort_values("fob_total", ascending=False)
        .groupby("porto").head(10).reset_index(drop=True))
    porto_totals_uf = cpuf.groupby("porto")["fob_total"].sum().rename("porto_total")
    cpuf_top = cpuf_top.merge(porto_totals_uf, on="porto")
    cpuf_top["pct"] = cpuf_top["fob_total"] / cpuf_top["porto_total"]
    cpuf_top.to_json(DASH / "comex_perfil_uf.json", orient="records")
    print(f"  comex_perfil_uf.json → {len(cpuf_top):,} linhas")
except Exception as e:
    print(f"  ⚠️ comex_perfil_uf.json FALHOU: {e}")

# ── 14. Matriz UF × Grupo CUCI × Porto (para tela detalhada) ──
try:
    mup = pd.read_csv(OUT / "comex_v2_uf_cuci_porto.csv")
    mup.rename(columns={
        "CO_CUCI_GRUPO": "cuci_grupo",
        "NO_CUCI_GRUPO": "cuci_nome",
    }, inplace=True)
    mup["cuci_nome"] = mup["cuci_nome"].apply(fix_encoding)
    # Top 5 por UF para manter JSON leve
    mup_top = (mup.sort_values("fob", ascending=False)
        .groupby("SG_UF_NCM").head(5).reset_index(drop=True))
    mup_top.to_json(DASH / "comex_uf_produto_porto.json", orient="records")
    print(f"  comex_uf_produto_porto.json → {len(mup_top):,} linhas (CUCI)")
except Exception as e:
    print(f"  ⚠️ comex_uf_produto_porto.json FALHOU: {e}")


# ══════════════════════════════════════════════════════
# ── NB3 v2 — CUCI Vulnerability Analysis JSONs ──
# ══════════════════════════════════════════════════════

# ── 15. CUCI HHI por porto ──
try:
    cuci_hhi_p = pd.read_csv(OUT / "comex_v2_hhi_porto.csv")
    cuci_hhi_p["top1_nome"] = cuci_hhi_p["top1_nome"].apply(fix_encoding)
    cuci_hhi_p.to_json(DASH / "cuci_hhi_porto.json", orient="records")
    print(f"  cuci_hhi_porto.json  → {len(cuci_hhi_p):,} linhas")
except Exception as e:
    print(f"  ⚠️ cuci_hhi_porto.json FALHOU: {e}")

# ── 16. CUCI scatter (cross-vulnerability) ──
try:
    cuci_sc = pd.read_csv(OUT / "comex_v2_matriz_vulnerabilidade.csv")
    cuci_sc["NO_CUCI_GRUPO"] = cuci_sc["NO_CUCI_GRUPO"].apply(fix_encoding)
    cuci_sc.to_json(DASH / "cuci_scatter.json", orient="records")
    print(f"  cuci_scatter.json    → {len(cuci_sc):,} linhas")
except Exception as e:
    print(f"  ⚠️ cuci_scatter.json FALHOU: {e}")

# ── 17. CUCI ranking de cadeias em risco ──
try:
    cuci_rk = pd.read_csv(OUT / "comex_v2_ranking_cadeias.csv")
    cuci_rk["NO_CUCI_GRUPO"] = cuci_rk["NO_CUCI_GRUPO"].apply(fix_encoding)
    cuci_rk = cuci_rk.head(25)
    cuci_rk.to_json(DASH / "cuci_ranking_cadeias.json", orient="records")
    print(f"  cuci_ranking_cadeias → {len(cuci_rk):,} linhas")
except Exception as e:
    print(f"  ⚠️ cuci_ranking_cadeias.json FALHOU: {e}")

# ── 18. CUCI ranking de portos ──
try:
    cuci_rp = pd.read_csv(OUT / "comex_v2_ranking_portos.csv")
    cuci_rp.to_json(DASH / "cuci_ranking_portos.json", orient="records")
    print(f"  cuci_ranking_portos  → {len(cuci_rp):,} linhas")
except Exception as e:
    print(f"  ⚠️ cuci_ranking_portos.json FALHOU: {e}")

# ── 19. CUCI HHI por cadeia (top 30) ──
try:
    cuci_hhi_c = pd.read_csv(OUT / "comex_v2_hhi_cadeia.csv")
    cuci_hhi_c["NO_CUCI_GRUPO"] = cuci_hhi_c["NO_CUCI_GRUPO"].apply(fix_encoding)
    cuci_hhi_c = cuci_hhi_c.nlargest(30, "fob_total")
    cuci_hhi_c.to_json(DASH / "cuci_hhi_cadeia.json", orient="records")
    print(f"  cuci_hhi_cadeia.json → {len(cuci_hhi_c):,} linhas")
except Exception as e:
    print(f"  ⚠️ cuci_hhi_cadeia.json FALHOU: {e}")


# ══════════════════════════════════════════════════════
# ── Seção F — FOB Exposto (cruzamento anomalias × ComexStat) ──
# ══════════════════════════════════════════════════════

# ── 20. CUCI ranking cadeias FOB exposto ──
try:
    rkf = pd.read_csv(OUT / "comex_v2_ranking_cadeias_fob.csv")
    rkf["NO_CUCI_GRUPO"] = rkf["NO_CUCI_GRUPO"].apply(fix_encoding)
    if "porto_principal" in rkf.columns:
        rkf["porto_principal"] = rkf["porto_principal"].apply(fix_encoding)
    rkf = rkf.head(30)
    rkf.to_json(DASH / "cuci_ranking_cadeias_fob.json", orient="records")
    print(f"  cuci_ranking_cadeias_fob → {len(rkf):,} linhas")
except Exception as e:
    print(f"  ⚠️ cuci_ranking_cadeias_fob.json FALHOU: {e}")

# ── 21. CUCI ranking portos FOB exposto ──
try:
    rpf = pd.read_csv(OUT / "comex_v2_ranking_portos_fob.csv")
    rpf.to_json(DASH / "cuci_ranking_portos_fob.json", orient="records")
    print(f"  cuci_ranking_portos_fob  → {len(rpf):,} linhas")
except Exception as e:
    print(f"  ⚠️ cuci_ranking_portos_fob.json FALHOU: {e}")

# ── 22. FOB exposto por UF ──
try:
    fuf = pd.read_csv(OUT / "comex_v2_fob_exposto_uf.csv")
    fuf.to_json(DASH / "fob_exposto_uf.json", orient="records")
    print(f"  fob_exposto_uf.json      → {len(fuf):,} linhas")
except Exception as e:
    print(f"  ⚠️ fob_exposto_uf.json FALHOU: {e}")

# ── 23. FOB exposto por porto × classificação ──
try:
    fpc = pd.read_csv(OUT / "comex_v2_fob_exposto.csv")
    fpc["NO_CUCI_GRUPO"] = fpc["NO_CUCI_GRUPO"].apply(fix_encoding)
    # Agregar por porto + classificação para JSON leve
    fpc_agg = (fpc.groupby(["porto", "classificacao"])
        .agg(fob_exposto=("fob_exposto", "sum"),
             meses_expostos=("meses_expostos", "sum"))
        .reset_index())
    fpc_agg.to_json(DASH / "fob_exposto_porto_cls.json", orient="records")
    print(f"  fob_exposto_porto_cls    → {len(fpc_agg):,} linhas")
except Exception as e:
    print(f"  ⚠️ fob_exposto_porto_cls.json FALHOU: {e}")

# ── 24. FOB por porto × região destino ──
try:
    fpr = pd.read_csv(OUT / "comex_v2_fob_porto_regiao.csv")
    fpr.to_json(DASH / "fob_porto_regiao.json", orient="records")
    print(f"  fob_porto_regiao.json    → {len(fpr):,} linhas")
except Exception as e:
    print(f"  ⚠️ fob_porto_regiao.json FALHOU: {e}")

# ── 25. FOB por cadeia × região destino ──
try:
    fcr = pd.read_csv(OUT / "comex_v2_fob_cadeia_regiao.csv")
    fcr["NO_CUCI_GRUPO"] = fcr["NO_CUCI_GRUPO"].apply(fix_encoding)
    if "porto_principal" in fcr.columns:
        fcr["porto_principal"] = fcr["porto_principal"].apply(fix_encoding)
    fcr.to_json(DASH / "fob_cadeia_regiao.json", orient="records")
    print(f"  fob_cadeia_regiao.json   → {len(fcr):,} linhas")
except Exception as e:
    print(f"  ⚠️ fob_cadeia_regiao.json FALHOU: {e}")


# ══════════════════════════════════════════════════════
# ── 26. Export via ano (participação marítima por ano) ──
# ══════════════════════════════════════════════════════

try:
    print("\n  Gerando export_via_ano.json a partir do ComexStat...")
    comex_parts = []
    for f in sorted(COMEXSTAT_RAW.glob("EXP_*.csv")):
        df = pd.read_csv(f, sep=";", usecols=["CO_ANO", "CO_VIA", "VL_FOB"],
                         dtype={"CO_ANO": int, "CO_VIA": int, "VL_FOB": float})
        comex_parts.append(df)
    comex_all = pd.concat(comex_parts, ignore_index=True)

    # FOB total por ano (todas as vias)
    total = comex_all.groupby("CO_ANO")["VL_FOB"].sum().rename("fob_total")
    # FOB marítima (CO_VIA == 1)
    marit = (comex_all[comex_all["CO_VIA"] == 1]
             .groupby("CO_ANO")["VL_FOB"].sum().rename("fob_maritima"))

    eva = pd.DataFrame({"fob_maritima": marit, "fob_total": total}).dropna()
    eva = eva.reset_index().rename(columns={"CO_ANO": "ano"})
    eva["pct_maritima"] = (eva["fob_maritima"] / eva["fob_total"] * 100).round(2)
    eva = eva.sort_values("ano")
    eva.to_json(DASH / "export_via_ano.json", orient="records")
    print(f"  export_via_ano.json      → {len(eva):,} anos ({eva['ano'].min()}-{eva['ano'].max()})")
except Exception as e:
    print(f"  ⚠️ export_via_ano.json FALHOU: {e}")


print(f"\n✓ Todos os dados gerados em {DASH.resolve()}")
