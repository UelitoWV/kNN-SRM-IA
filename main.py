import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Carregar dados
file_path = "banana.csv"
data = pd.read_csv(file_path)
X, y = data[['x', 'y']].values, data['class'].values

# Dividir Treino/Teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)


# Função que atende o SRM (Structural Risk Minimization)
def srm_nn_reduce(X, y):
    n = len(X)
    J = set()

    # Calcula todas as distâncias entre os pontos opostos
    opposite_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if y[i] != y[j]:
                d = np.linalg.norm(X[i] - X[j])
                opposite_pairs.append((d, i, j))

    # Ordenar pela menor distância
    opposite_pairs.sort(key=lambda x: x[0])

    # Testa a partir do cálculo das distâncias, se o classificador classifica corretamente todos os pontos
    def classifies_all(J):
        J_list = list(J)
        if len(J_list) == 0:
            return False

        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(X[J_list], y[J_list])

        preds = knn.predict(X)
        return np.all(preds == y)

    # Inclusão dos pontos não adicionados em outra lista limpa de maneira iterativa
    for _, i, j in opposite_pairs:

        if i not in J:
            J.add(i)
        if j not in J:
            J.add(j)

        if classifies_all(J):
            break

    # Conjunto reduzido
    J = sorted(list(J))
    return X[J], y[J]


# Aplica a redução
X_reduced, y_reduced = srm_nn_reduce(X_train, y_train)


# kNN Original
knn_full = KNeighborsClassifier(n_neighbors=10)
knn_full.fit(X_train, y_train)
acc_full = knn_full.score(X_test, y_test)

# kNN SRM-NN (Reduzido)
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