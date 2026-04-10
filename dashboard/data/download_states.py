"""Download Brazil states GeoJSON from IBGE API and enrich with UF siglas."""
import requests, json

# IBGE code → UF sigla mapping
IBGE_TO_UF = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA",
    "16": "AP", "17": "TO", "21": "MA", "22": "PI", "23": "CE",
    "24": "RN", "25": "PB", "26": "PE", "27": "AL", "28": "SE",
    "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT",
    "52": "GO", "53": "DF",
}

url = (
    "https://servicodados.ibge.gov.br/api/v3/malhas/paises/BR"
    "?formato=application/vnd.geo+json&qualidade=minima&intrarregiao=UF"
)

print("Downloading from IBGE...")
r = requests.get(url, timeout=30)
print(f"Status: {r.status_code}, size: {len(r.content)} bytes")

geo = r.json()
print(f"Features: {len(geo['features'])}")

# Enrich each feature with UF sigla
for feat in geo["features"]:
    code = str(feat["properties"].get("codarea", ""))
    feat["properties"]["uf"] = IBGE_TO_UF.get(code, code)

print("Sample:", geo["features"][0]["properties"])

with open("brazil_states.geojson", "w", encoding="utf-8") as f:
    json.dump(geo, f)

print("Saved brazil_states.geojson")
