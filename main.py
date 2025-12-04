import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from utils.functions import srm_nn_reduce, srm_nn_reduce_fast
from utils.plot import plot_informations

# Carregar dados
file_path = "database/banana.csv"
data = pd.read_csv(file_path)
X, y = data[['x', 'y']].values, data['class'].values

# Dividir Treino/Teste (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Aplica a redução
X_reduced, y_reduced = srm_nn_reduce_fast(X_train, y_train)



k_full = 3
k_srm = 1

# kNN Original
knn_full = KNeighborsClassifier(n_neighbors=k_full)
knn_full.fit(X_train, y_train)
acc_full = knn_full.score(X_test, y_test)

# kNN SRM-NN (Reduzido)
knn_srm = KNeighborsClassifier(n_neighbors=k_srm)
knn_srm.fit(X_reduced, y_reduced)
acc_srm = knn_srm.score(X_test, y_test)

reduction = 100 * (1 - len(X_reduced) / len(X_train))

# Resultados e métricas da redução
plot_informations(X_train, y_train, X_reduced, y_reduced, reduction, acc_full, acc_srm, k_full, k_srm)