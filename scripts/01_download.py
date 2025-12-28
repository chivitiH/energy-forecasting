import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import *
import subprocess

print("\n" + "="*70)
print("📥 DOWNLOAD")
print("="*70 + "\n")

kaggle_file = Path.home() / ".kaggle" / "kaggle.json"
if not kaggle_file.exists():
    print("❌ kaggle.json manquant !")
    sys.exit(1)

print("✅ kaggle.json trouvé\n")

DATASET_SLUG = "robikscube/hourly-energy-consumption"
print(f"📦 Téléchargement : {DATASET_SLUG}\n")

try:
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", DATASET_SLUG,
        "-p", str(RAW_DIR),
        "--unzip"
    ], check=True)
    
    print("\n✅ Téléchargé et dézippé !\n")
    
    csv_files = list(RAW_DIR.glob("*.csv"))
    print(f"📄 {len(csv_files)} fichiers CSV :")
    for f in sorted(csv_files):
        size_mb = f.stat().st_size / 1024 / 1024
        print(f"   {f.name} ({size_mb:.1f} MB)")
    
    print("\n💡 Utilise PJME_hourly.csv\n")
    
except subprocess.CalledProcessError as e:
    print(f"\n❌ Erreur : {e}\n")
    sys.exit(1)

print("="*70)
