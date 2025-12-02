from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Função antiga
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


# Função que atende o SRM (código feito de acordo com o artigo)
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