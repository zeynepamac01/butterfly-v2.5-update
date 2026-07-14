"""
prepare_release.py

This script:
1) ask for the HRA version
2) download the JSON
3) save it
4) print the notebook paths
5) check whether key files exist

"""

import os
import json
import requests
import subprocess

# Ask user for HRA version
version = input("Enter HRA version (e.g., 2.5): ").strip()
hra_version = f"v{version}"

# Create output directory
output_folder = f"./data/{hra_version}"
os.makedirs(output_folder, exist_ok=True)

# HRA JSON URL
asctb_url = (
    "https://cdn.humanatlas.io/hra-asctb-json-releases/"
    f"hra-asctb-all.{hra_version}.json"
)

print(f"\nDownloading HRA {hra_version}...")

response = requests.get(asctb_url)

if response.status_code == 200:
    data = response.json()

    json_path = f"{output_folder}/hra-asctb-all.{hra_version}.json"

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print("✓ Successfully downloaded HRA data.")
    print(f"✓ Saved JSON to: {json_path}")

else:
    print(f"✗ Failed to download HRA data (status code {response.status_code})")
    raise SystemExit(1)

print("\nRunning build-network.py...")
result = subprocess.run(["python3", "build-network.py"], capture_output=True, text=True)

if result.returncode == 0:
    print("✓ build-network.py completed successfully")
    print(result.stdout)
else:
    print("✗ build-network.py failed")
    print(result.stderr)
    raise SystemExit(1)

# Notebook paths
print("\nNotebook paths:")
print(f'nodes_url = "data/{hra_version}/asct-nodes.csv"')
print(f'edges_url = "data/{hra_version}/asct-edges.csv"')
print('output_path = "."')

# Check expected files
nodes_path = f"./data/{hra_version}/asct-nodes.csv"
edges_path = f"./data/{hra_version}/asct-edges.csv"

print("\nChecking generated files:")

if os.path.exists(nodes_path):
    print("✓ Found asct-nodes.csv")
else:
    print("✗ Missing asct-nodes.csv")

if os.path.exists(edges_path):
    print("✓ Found asct-edges.csv")
else:
    print("✗ Missing asct-edges.csv")

table_s1_path = "./data/Table_S1.csv"
vega_config_path = "./vega_config/vega_config.json"

print("\nChecking additional required files:")

if os.path.exists(table_s1_path):
    print("✓ Found Table_S1.csv")
else:
    print("✗ Missing Table_S1.csv")

if os.path.exists(vega_config_path):
    print("✓ Found vega_config.json")
else:
    print("✗ Missing vega_config.json")

