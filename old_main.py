import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1) Carregar dados
file_path = "banana.csv"
data = pd.read_csv(file_path)
X, y = data[['x', 'y']].values, data['class'].values

# 2) Dividir Treino/Teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# ----------------------------------------------------
# 3) Função de Redução por Vizinhos Opostos
# ----------------------------------------------------
def select_opposite_neighbors(X, y):
    classes = np.unique(y)
    # Separa os dados por classe
    X_c1 = X[y == classes[0]]
    X_c2 = X[y == classes[1]]
    
    # Índices originais para podermos recuperar depois
    idx_c1 = np.where(y == classes[0])[0]
    idx_c2 = np.where(y == classes[1])[0]
    
    to_keep_indices = set()
    
    # A) Quem de C1 é o "vizinho mais próximo" de alguém de C2?
    # Treina em C1 para buscar vizinhos nele
    knn_c1 = KNeighborsClassifier(n_neighbors=6)
    knn_c1.fit(X_c1, np.zeros(len(X_c1))) 
    
    # Para cada ponto de C2, quem é o vizinho em C1?
    dist, ind = knn_c1.kneighbors(X_c2)
    # Recupera os índices reais
    real_indices_c1 = idx_c1[ind.flatten()]
    to_keep_indices.update(real_indices_c1)
    
    # B) O inverso: Quem de C2 é vizinho de alguém de C1?
    knn_c2 = KNeighborsClassifier(n_neighbors=6)
    knn_c2.fit(X_c2, np.zeros(len(X_c2)))
    
    dist, ind = knn_c2.kneighbors(X_c1)
    real_indices_c2 = idx_c2[ind.flatten()]
    to_keep_indices.update(real_indices_c2)
    
    # Retorna o subconjunto filtrado
    keep_list = sorted(list(to_keep_indices))
    return X[keep_list], y[keep_list]

# Aplica a redução
X_reduced, y_reduced = select_opposite_neighbors(X_train, y_train)

# ----------------------------------------------------
# 4) Comparação e Avaliação
# ----------------------------------------------------
# kNN Original
knn_full = KNeighborsClassifier(n_neighbors=5)
knn_full.fit(X_train, y_train)
acc_full = knn_full.score(X_test, y_test)

# kNN SRM (Reduzido)
# Como sobram poucos pontos, k deve ser pequeno (ex: 1 ou 3)
knn_srm = KNeighborsClassifier(n_neighbors=3) 
knn_srm.fit(X_reduced, y_reduced)
acc_srm = knn_srm.score(X_test, y_test)

reduction = 100 * (1 - len(X_reduced) / len(X_train))

print(f"--- RESULTADOS ---")
print(f"Tamanho Original: {len(X_train)}")
print(f"Tamanho Reduzido: {len(X_reduced)} (Redução de {reduction:.2f}%)")
print(f"Acurácia Original: {acc_full:.4f}")
print(f"Acurácia SRM:      {acc_srm:.4f}")

# Plot para visualizar a "limpeza"
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=10, alpha=0.3)
plt.title(f"kNN original ({len(X_train)} pontos)")

plt.subplot(1, 2, 2)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y_reduced, cmap='coolwarm', s=20, alpha=1.0)
plt.title(f"kNNSRM ({len(X_reduced)} pontos)")

plt.tight_layout()
plt.show()