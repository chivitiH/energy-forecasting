"""Comparaison rapide via JSON"""
import json
from pathlib import Path

print("\n" + "="*70)
print("📊 COMPARAISON V1 vs IMPROVED (via métriques)")
print("="*70 + "\n")

# Charger métriques V1
with open('results/test_metrics.json', 'r') as f:
    v1 = json.load(f)

# Charger métriques improved (du training)
with open('results/metrics_improved.json', 'r') as f:
    improved = json.load(f)

print("MODEL V1 (2 LSTM):")
print(f"  RMSE: {v1['rmse']:.2f} MW")
print(f"  MAE:  {v1['mae']:.2f} MW")
print(f"  MAPE: {v1['mape']:.2f}%")
print(f"  Val Loss: N/A\n")

print("MODEL IMPROVED (3 LSTM + Attention):")
print(f"  Best Val Loss: {improved['best_val_loss']:.4f}")
print(f"  Best Val MAE:  {improved['best_val_mae']:.4f}")
print(f"  Epochs: {improved['epochs_completed']}")
print(f"  Params: {improved['params']:,}\n")

print("="*70)
print("⚠️  Note: Comparaison indirecte via validation loss")
print("="*70 + "\n")

# Comparer val loss si disponible
print("📊 ANALYSE:")
print("  V1 val_loss final: ~0.0825 (de l'historique)")
print(f"  Improved val_loss: {improved['best_val_loss']:.4f}\n")

if improved['best_val_loss'] > 0.15:
    print("❌ IMPROVED : Val loss PLUS ÉLEVÉE → Moins bon")
    print("   Possible overfitting avec 3 LSTM + Attention\n")
    print("🏆 VERDICT: V1 reste le meilleur modèle !")
else:
    print("✅ IMPROVED : Val loss acceptable")
    print("   Besoin d'évaluer sur test set pour confirmer\n")

print("="*70)
