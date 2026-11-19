import yaml

input_file = "simulation_map.yaml"
output_file = "simulation_map_filtered.yaml"
soglia_alpha = 0.0202

# Carica il file YAML
with open(input_file, "r", encoding="utf-8") as f:
  dati = yaml.safe_load(f)

dati_filtrati = {}

for chiave, lista_file in dati.items():
  # Estrae z, alpha, lambda dalla stringa chiave (es. "15.0,0.0190,1050")
  parti = str(chiave).split(",")
  alpha = float(parti[1])

  # Mantiene solo i blocchi con alpha > 0.0202
  if alpha > soglia_alpha:
    dati_filtrati[chiave] = lista_file

# Salva il risultato nel nuovo file YAML
with open(output_file, "w", encoding="utf-8") as f:
  yaml.dump(dati_filtrati, f, sort_keys=False, default_flow_style=False)

print(f"Fatto! Blocchi originali: {len(dati)}")
print(f"Blocchi rimasti (alpha > {soglia_alpha}): {len(dati_filtrati)}")
