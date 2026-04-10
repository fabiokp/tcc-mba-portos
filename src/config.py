"""Configuração global — v3.1 pós-tracers."""
from pathlib import Path
import numpy as np

# ── PATHS ──
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGS = OUTPUTS / "figuras"

ANTAQ_RAW = DATA_RAW / "antaq"
PORTWATCH_RAW = DATA_RAW / "portwatch"
COMEXSTAT_RAW = DATA_RAW / "comexstat"

for d in [DATA_PROC, OUTPUTS, FIGS]:
    d.mkdir(parents=True, exist_ok=True)

# ── DADOS BRUTOS — caminhos reais ──
_WORKSPACE = ROOT.parent                       # TCC MBA ENAP
ANTAQ_RAW_REAL = _WORKSPACE / "dados" / "estatistico"   # fallback: txts extraídos
ANTAQ_RAW_ZIPS = DATA_RAW / "antaq"                     # zips por ano (fonte primária)
PORTWATCH_FILE = PORTWATCH_RAW / "Daily_Ports_Data.csv"  # v2 PortWatch (até fev/2026)

# ── ANTAQ — metadados dos arquivos ──
ANTAQ_ENCODING = "utf-8-sig"
ANTAQ_SEP = ";"

ANTAQ_ID_COL = "IDAtracacao"
ANTAQ_PORT_COL = "Complexo Portuário"
ANTAQ_DATE_COL = "Data Atracação"
TEMPOS_T1_COL = "TEsperaAtracacao"
TEMPOS_TATRACADO_COL = "TAtracado"
ANTAQ_NAV_COL = "Tipo de Navegação da Atracação"
ANTAQ_NAV_LONGO_CURSO = "Longo Curso"
CARGA_SENTIDO_COL = "Sentido"
CARGA_TONELAGEM_COL = "VLPesoCargaBruta"
EXPORT_VALUES = ["Embarcados"]

# ── PORTOS — mapeamento Complexo Portuário → nome curto ──
PORTO_NAMES = {
    # ── Top 15 originais ──
    "Santos": "Santos",
    "Paranaguá - Antonina": "Paranaguá",
    "Rio Grande": "Rio Grande",
    "Itaguaí": "Itaguaí",
    "Itaqui": "São Luís",
    "Rio de Janeiro -  Niterói": "Rio de Janeiro",
    "Suape - Recife": "Suape",
    "Vitória": "Vitória",
    "Aratu - Salvador": "Salvador",
    "Manaus": "Manaus",
    "Imbituba": "Imbituba",
    "Itajaí": "Itajaí",
    "São Francisco do Sul": "São Francisco do Sul",
    "Pecém - Fortaleza": "Pecém",
    "Vila do Conde - Belém": "Vila do Conde",
    # ── Expansão +3 (LC ≥ 4/sem) ──
    # Removidos São Sebastião (3.2/sem) e Angra dos Reis (3.0/sem) — terminais
    # petroleiros com LC insuficiente para medianas semanais estáveis.
    "São João da Barra": "São João da Barra",  # #11 — Porto do Açu (minério)
    "Porto Alegre": "Porto Alegre",            # #17 — regional Sul (enchentes RS)
    "Barra do Riacho": "Barra do Riacho",      # #20 — celulose (Aracruz)
}

ANTAQ_TO_PORTWATCH = {
    # ── Top 15 originais ──
    "Santos": "port1160",
    "Paranaguá": "port885",
    "Rio Grande": "port1104",
    "Itaguaí": "port2045",
    "São Luís": "port506",
    "Rio de Janeiro": "port1103",
    "Suape": "port1002",
    "Vitória": "port1368",
    "Salvador": "port62",
    "Manaus": "port693",
    "Imbituba": "port492",
    "Itajaí": "port505",
    "São Francisco do Sul": "port1161",
    "Pecém": "port1010",
    "Vila do Conde": "port2046",
    # ── Expansão +3 (LC ≥ 4/sem) ──
    "São João da Barra": "port166",     # Porto do Acu
    "Porto Alegre": "port997",          # Porto Alegre
    "Barra do Riacho": "port1012",      # Portocel
}

# ── DIMENSÕES ──
DIMS = ["atracacoes", "tonelagem_exp", "t1_mediano", "tatracado_mediano"]
DIM_LABELS = {
    "atracacoes": "Atracações/sem",
    "tonelagem_exp": "Ton. export/sem",
    "t1_mediano": "T1 mediano (h)",
    "tatracado_mediano": "TAtracado med. (h)",
}

# ── NB1: PREPARAÇÃO ──
MIN_DATA_YEAR = 2014          # Fix 2: cortar séries longas demais
TOP_N_PORTOS = 18              # v8: era 20, cortados S.Sebastião e Angra (LC < 4/sem)
T1_OUTLIER_MAX_HORAS = 720
N_CLUSTERS = 5
TOP_N_GLOBAL_INDEX = 50
CLUSTERING_FEATURES = [
    "media_portcalls", "cv", "sazonalidade_semanal",
    "pct_container", "pct_dry_bulk", "pct_tanker",
]

MM_WINDOWS = [4, 8, 12]
LAG_WEEKS = [1, 4, 13, 52]
ROLLING_STD_WINDOW = 8
STL_PERIOD = 52
STL_DIMS = ["atracacoes", "tonelagem_exp"]

# ── NB2: MODELAGEM ──
CV_MIN_TRAIN_WEEKS = 78            # 🔧 ERA 156. 78 sem ≈ 1.5 anos.
CV_VAL_WEEKS = 13                  # Janela de validação
CV_STEP_WEEKS = 13                 # Passo entre folds
WALKFORWARD_RETRAIN_WEEKS = 26

# ── NB2: DETECÇÃO ──
MAD_K = 3.0                        # Threshold base MAD (será adaptado por porto)
STL_RESID_ZSCORE = 3.0             # Threshold base STL (será adaptado por porto)
IFOREST_CONTAMINATION = 0.03
ENSEMBLE_MIN_AGREEMENT = 2

# ── NB2: THRESHOLD ADAPTATIVO ──
ADAPTIVE_THRESHOLD = True

# Floor e ceiling POR DIMENSÃO
# Todas as 4 dims são contínuas → floor padrão uniforme
ADAPTIVE_K_FLOORS = {
    "atracacoes":       2.0,
    "tonelagem_exp":    2.0,
    "t1_mediano":       2.0,
    "tatracado_mediano": 2.0,
}
ADAPTIVE_K_MAX = 4.0           # Ceiling global (portos muito voláteis)

# ── NB2: BURN-IN ──
BURNIN_WEEKS = 78               # Semanas iniciais excluídas (fallback naive)
                                # = CV_MIN_TRAIN_WEEKS (lag-52 + treino mínimo)

# ── NB2: CLASSIFICAÇÃO (Dual Score Paralelo) ──
COOC_WINDOW_WEEKS = 4
COOC_GI_WINDOW_WEEKS = 2
COOC_SAME_DIM = True
SCORE_A_THRESHOLD = 4         # Fix 3: subir para 15 portos
SCORE_B_THRESHOLD = 1.0

# ── NB2: FINGERPRINT ──
FINGERPRINT_WINDOW_WEEKS = 4

# ── EVENTOS ──
KNOWN_EVENTS = [
    {"name": "Greve Caminhoneiros",
     "start": "2018-05-21", "end": "2018-06-10",
     "expected": "nacional",
     "portos_foco": ["Santos", "Paranaguá", "Rio Grande", "Itaguaí"],
     "note": "Sem PortWatch (pré-2019). TRACER: ✅ classificado nacional em v3/v3.1"},

    {"name": "Greve Receita Federal",
     "start": "2024-12-01", "end": "2025-01-15",
     "expected": "nacional",
     "portos_foco": ["Santos", "Paranaguá", "Itaguaí"],
     "note": "VERIFICAR datas exatas"},

    {"name": "COVID",
     "start": "2020-03-01", "end": "2020-06-30",
     "expected": "global",
     "note": "TRACER: Score B captura (7/23 global) mas mode=isolado com 5 portos."},

    {"name": "COVID recuperação",
     "start": "2020-09-01", "end": "2021-06-30",
     "expected": "global",
     "note": "Boom de frete, congestão global"},

    {"name": "Bloqueio Canal de Suez",
     "start": "2021-03-20", "end": "2021-04-15",
     "expected": "global",
     "note": "TRACER: ✅ classificado global em v3.1 (Score B)"},

    {"name": "Lockdowns China",
     "start": "2022-03-01", "end": "2022-06-30",
     "expected": "global",
     "note": "Impacto em cadeias de suprimento, container"},

    {"name": "Guerra Ucrânia",
     "start": "2022-02-24", "end": "2022-12-31",
     "expected": "global",
     "note": "TRACER: ✅ classificado global em v3.1 (Score B)."},

    {"name": "Ataques Houthi",
     "start": "2023-11-15", "end": "2024-06-30",
     "expected": "global",
     "note": "Desvio rotas via Cabo"},

    {"name": "Tarifação EUA",
     "start": "2025-02-01", "end": "2025-06-30",
     "expected": "global",
     "note": "VERIFICAR datas e escopo"},

    {"name": "Enchentes RS",
     "start": "2024-04-25", "end": "2024-06-30",
     "expected": "isolado",
     "portos_foco": ["Rio Grande", "Porto Alegre"],
     "note": "Porto Alegre adicionado na expansão 15→20. Evento de validação."},

    {"name": "Safra soja",
     "start_month": 2, "end_month": 5,
     "expected": "sazonal",
     "portos_foco": ["Santos", "Paranaguá", "Rio Grande", "São Luís"],
     "note": "Pico anual. Se detectado como anomalia → modelo subestima sazonalidade."},
]
