"""Generate comex_flows.json — aggregated UF→Porto flows for the arc map."""
import json
from collections import defaultdict
from pathlib import Path

DASH = Path(__file__).parent
VALID_UFS = {"AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG","MS","MT",
             "PA","PB","PE","PI","PR","RJ","RN","RO","RR","RS","SC","SE","SP","TO"}

# UF centroid coordinates (approximate)
UF_COORDS = {
    "AC": [-8.77, -70.55], "AL": [-9.57, -36.78], "AM": [-3.47, -65.10],
    "AP": [1.41, -51.77], "BA": [-12.97, -41.68], "CE": [-5.20, -39.53],
    "DF": [-15.83, -47.86], "ES": [-19.19, -40.34], "GO": [-15.98, -49.86],
    "MA": [-5.42, -45.44], "MG": [-18.10, -44.38], "MS": [-20.51, -54.54],
    "MT": [-12.64, -55.42], "PA": [-3.79, -52.48], "PB": [-7.28, -36.72],
    "PE": [-8.38, -37.86], "PI": [-7.72, -42.73], "PR": [-24.89, -51.55],
    "RJ": [-22.25, -42.66], "RN": [-5.81, -36.59], "RO": [-10.83, -63.34],
    "RR": [1.99, -61.33], "RS": [-30.17, -53.50], "SC": [-27.45, -50.95],
    "SE": [-10.57, -37.45], "SP": [-22.19, -48.79], "TO": [-10.25, -48.25],
}

# Load raw UF-product-porto data
ufp = json.load(open(DASH / "comex_uf_produto_porto.json"))

# Aggregate: UF→Porto total FOB
flows_raw = defaultdict(float)
for r in ufp:
    uf = r["SG_UF_NCM"]
    if uf not in VALID_UFS:
        continue
    flows_raw[(uf, r["porto"])] += r["fob"]

# Build flow list
flows = []
for (uf, porto), fob in flows_raw.items():
    if uf in UF_COORDS:
        flows.append({
            "uf": uf,
            "porto": porto,
            "fob": round(fob),
            "lat_uf": UF_COORDS[uf][0],
            "lon_uf": UF_COORDS[uf][1],
        })

# Sort by FOB descending
flows.sort(key=lambda x: -x["fob"])

with open(DASH / "comex_flows.json", "w") as f:
    json.dump(flows, f)

print(f"comex_flows.json -> {len(flows)} fluxos UF->Porto")
for f in flows[:5]:
    print(f"  {f['uf']} -> {f['porto']}: {f['fob']/1e9:.1f} bi")
