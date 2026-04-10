# Monitoramento de Disrupcões em Portos Brasileiros de Exportação

**Trabalho de Conclusão de Curso — MBA em Gestão Pública (ENAP)**

Autor: Fábio Paim  
Orientador: *a definir*  
Dashboard interativo: [tccmbafkp.netlify.app](https://tccmbafkp.netlify.app)

---

## Resumo

Este repositório contém o código-fonte completo para reprodutibilidade das análises
do artigo *"Monitoramento de eventos disruptivos em portos brasileiros de exportação:
uma abordagem baseada em anomalias multidimensionais"*.

O trabalho propõe um sistema de detecção de anomalias portuárias usando séries temporais
semanais de **18 portos brasileiros** (responsáveis por >95% das atracações de longo curso)
em **4 dimensões operacionais** (atracações, tonelagem exportada, tempo de espera e
tempo atracado), combinando modelagem preditiva (XGBoost + correção AR(1)),
detecção por ensemble (MAD adaptativo + STL + Isolation Forest) e classificação
hierárquica (dual score: co-ocorrência local + índice global).

A análise de exposição econômica cruza as anomalias detectadas com dados de exportação
(ComexStat/MDIC), calculando o FOB exposto por cadeia produtiva, UF e porto, com
métricas de concentração (HHI) para identificar vulnerabilidades na logística de exportação.

## Estrutura do Repositório

```
├── README.md
├── requirements.txt              ← Dependências Python
├── run_pipeline.py               ← Pipeline completo headless (NB1 + NB2 + NB3)
│
├── notebooks/                    ← Jupyter Notebooks (sem outputs)
│   ├── 01_preparacao.ipynb           Ingestão ANTAQ, limpeza, séries semanais
│   ├── 02_analise.ipynb              XGBoost + AR(1), ensemble, detecção, classificação
│   ├── 03_comexstat.ipynb            Cruzamento ComexStat: HHI, FOB exposto
│   ├── 04_analises_adicionais.ipynb  Análises exploratórias, validações
│   ├── 05_validacao_artigo.ipynb     Validação final dos números do artigo
│   └── figuras_artigo.ipynb          Figuras estáticas (16 artigo + 11 suplementares)
│
├── src/
│   └── config.py                 ← Parâmetros, caminhos, constantes do modelo
│
├── scripts/
│   └── prepare_radar_v1.py       ← Painel de alerta: z-scores PortWatch (origem + destino)
│
├── dashboard/                    ← Dashboard interativo (Quarto + OJS/D3)
│   ├── index.qmd                     Código-fonte (≈150 KB, 12 abas, 40+ gráficos)
│   ├── _quarto.yml                   Configuração Quarto
│   ├── _publish.yml                  Configuração Netlify
│   ├── assets/                       CSS, SCSS, D3.js, TopoJSON
│   └── data/                         JSONs do dashboard + scripts de preparação
│       ├── prepare_data.py               Converte parquets → JSONs
│       ├── gen_flows.py                  Gera fluxos porto↔país (mapa)
│       ├── download_states.py            Baixa geojson dos estados BR
│       ├── *.json                        30 JSONs consumidos pelo dashboard
│       └── *.geojson / *.topojson        Mapas base (BR estados, mundo)
│
└── data/
    └── tabelas_auxiliares/       ← Tabelas auxiliares de mapeamento
        ├── NCM.csv                   Nomenclatura Comum do Mercosul
        ├── NCM_CUCI.csv              NCM → CUCI (classificação por cadeia)
        └── NCM_PAIS.csv              Código de países
```

## Pipeline de Execução

Os notebooks devem ser executados em ordem sequencial:

| Etapa | Arquivo | Descrição | Entrada | Saída |
|-------|---------|-----------|---------|-------|
| 1 | `01_preparacao.ipynb` | Ingestão, limpeza, agregação semanal, features, clustering | ANTAQ zips, PortWatch CSV | `data/processed/*.parquet` |
| 2 | `02_analise.ipynb` | Walk-forward CV, XGBoost+AR(1), ensemble, dual score, fingerprints | Parquets NB1 | `data/output/anomalias_classificadas.parquet`, `residuos.parquet`, `fingerprints.parquet` |
| 3 | `03_comexstat.ipynb` | NCM→CUCI, HHI, FOB exposto por cadeia/UF/porto | ComexStat CSVs, anomalias NB2 | `data/output/comex_v2_*.csv` |
| 4 | `04_analises_adicionais.ipynb` | EDA complementar, diagnósticos | Todos anteriores | Visualizações |
| 5 | `figuras_artigo.ipynb` | Figuras para o artigo (PNG 300 DPI + PDF vetorial) | Todos anteriores | `figuras/*.png`, `figuras/*.pdf` |
| — | `run_pipeline.py` | Execução headless completa (NB1+NB2+NB3) | Dados brutos | Todos os outputs |

### Dashboard

```bash
cd dashboard
python data/prepare_data.py      # Gera JSONs a partir dos parquets
quarto render                    # Renderiza HTML
quarto publish netlify           # Publica no Netlify
```

## Fontes de Dados

Os dados brutos **não estão incluídos** neste repositório. Para reproduzir as análises,
é necessário obter os dados das fontes originais:

| Fonte | Descrição | Acesso |
|-------|-----------|--------|
| **ANTAQ** | Anuário Estatístico Aquaviário — atracações, carga, tempos (2014–2026) | [web.antaq.gov.br/Anuario](https://web.antaq.gov.br/Anuario/) |
| **ComexStat/MDIC** | Exportações brasileiras por porto, UF, NCM/SH (2014–2026) | [comexstat.mdic.gov.br](http://comexstat.mdic.gov.br/) |
| **PortWatch/FMI** | Daily Port Activity Data — atividade portuária global (2019–2026) | [portwatch.imf.org](https://portwatch.imf.org/) |

### Estrutura esperada dos dados brutos

```
data/
├── raw/
│   ├── antaq/           ← Zips anuais: 2014.zip … 2026.zip
│   ├── comexstat/        ← CSVs de exportação: EXP_2014.csv … EXP_2026.csv
│   └── portwatch/        ← Daily_Ports_Data.csv
├── processed/            ← Gerado pelo NB1 (parquets intermediários)
└── output/               ← Gerado pelo NB2/NB3 (anomalias, rankings, comex)
```

## Ambiente

- **Python 3.12** (recomendado via conda)
- Dependências: ver `requirements.txt`

```bash
conda create -n port_analysis python=3.12
conda activate port_analysis
pip install -r requirements.txt
```

## Principais Parâmetros do Modelo

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `TOP_N_PORTOS` | 18 | Portos com ≥4 atracações LC/semana |
| `MIN_DATA_YEAR` | 2014 | Início das séries |
| `CV_MIN_TRAIN_WEEKS` | 78 | Mínimo treino walk-forward (1,5 anos) |
| `MAD_K` | 3.0 | Threshold base MAD (adaptado por porto) |
| `ADAPTIVE_K_FLOORS` | 2.0 | Floor do threshold adaptativo |
| `ENSEMBLE_MIN_AGREEMENT` | 2 | Mínimo de detectores concordando (de 3) |
| `COOC_WINDOW_WEEKS` | 4 | Janela co-ocorrência (Score A) |
| `SCORE_A_THRESHOLD` | 4 | Limiar Score A (≥4 portos = nacional/global) |
| `N_CLUSTERS` | 5 | Perfis de disrupção (K-Means + PCA) |

Todos os parâmetros estão documentados em `src/config.py`.

## Licença

Este repositório é disponibilizado para fins acadêmicos e de reprodutibilidade.
