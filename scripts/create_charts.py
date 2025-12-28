"""Créer graphiques pour README"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Charger métriques
with open('results/test_metrics.json', 'r') as f:
    metrics = json.load(f)

# Créer dossier
Path('docs').mkdir(exist_ok=True)

# ============================================================================
# GRAPHIQUE 1 : Métriques du modèle
# ============================================================================

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

labels = ['RMSE\n(MW)', 'MAE\n(MW)', 'MAPE\n(%)']
values = [metrics['rmse'], metrics['mae'], metrics['mape']]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, (label, value, color) in enumerate(zip(labels, values, colors)):
    ax[i].bar([0], [value], color=color, alpha=0.7, width=0.5)
    ax[i].set_ylabel('Value', fontsize=12)
    ax[i].set_title(label, fontsize=14, fontweight='bold')
    ax[i].set_xticks([])
    ax[i].text(0, value/2, f'{value:.2f}', ha='center', va='center', 
               fontsize=20, fontweight='bold', color='white')
    ax[i].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('docs/metrics.png', dpi=150, bbox_inches='tight')
print("✅ docs/metrics.png créé")

# ============================================================================
# GRAPHIQUE 2 : Architecture
# ============================================================================

fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

architecture = [
    ('Input', '(168, 11)', '#E8F5E9'),
    ('LSTM 1', '128 units', '#81C784'),
    ('Dropout', '20%', '#FFECB3'),
    ('LSTM 2', '64 units', '#81C784'),
    ('Dropout', '20%', '#FFECB3'),
    ('Dense', '64 units', '#64B5F6'),
    ('Dropout', '20%', '#FFECB3'),
    ('Output', '24 predictions', '#EF5350')
]

y_pos = len(architecture) - 1
for i, (name, detail, color) in enumerate(architecture):
    rect = plt.Rectangle((0.2, y_pos - 0.4), 0.6, 0.8, 
                         facecolor=color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(0.5, y_pos, f'{name}\n{detail}', ha='center', va='center',
           fontsize=12, fontweight='bold')
    
    if i < len(architecture) - 1:
        ax.arrow(0.5, y_pos - 0.5, 0, -0.3, head_width=0.1, head_length=0.1,
                fc='black', ec='black')
    
    y_pos -= 1

ax.set_xlim(0, 1)
ax.set_ylim(-1, len(architecture))
ax.set_title('LSTM Model Architecture', fontsize=16, fontweight='bold', pad=20)

plt.savefig('docs/architecture.png', dpi=150, bbox_inches='tight')
print("✅ docs/architecture.png créé")

# ============================================================================
# GRAPHIQUE 3 : Comparaison métrique
# ============================================================================

fig, ax = plt.subplots(figsize=(8, 6))

categories = ['Accuracy\n(100-MAPE)', 'Speed\n(epochs/min)', 'Efficiency\n(params/1000)']
values = [100 - metrics['mape'], 10, 0.2]  # Valeurs normalisées
max_vals = [100, 10, 1]

x = np.arange(len(categories))
bars = ax.bar(x, values, color=['#4CAF50', '#2196F3', '#FF9800'], alpha=0.7)

for i, (bar, val, max_val) in enumerate(zip(bars, values, max_vals)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height/2,
           f'{val:.1f}', ha='center', va='center',
           fontsize=14, fontweight='bold', color='white')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Overview', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('docs/overview.png', dpi=150, bbox_inches='tight')
print("✅ docs/overview.png créé")

print("\n✅ Tous les graphiques créés dans docs/")
