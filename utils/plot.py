import matplotlib.pyplot as plt

def plot_informations(X_train, y_train, X_reduced, y_reduced, reduction, acc_full, acc_srm):
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