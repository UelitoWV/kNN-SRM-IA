import pandas as pd
from scipy.io import arff

# Carrega os dados
data, meta = arff.loadarff('database/disk-4000n.arff')
df = pd.DataFrame(data)

# Realiza a substituição: b'0' vira 1, b'1' vira 2
df['class'] = df['class'].replace({b'0': 1, b'1': 2})

# (Opcional) Garante que a coluna seja do tipo Inteiro para não ficar 1.0 e 2.0 no CSV
df['class'] = df['class'].astype(int)

# Salva em CSV
df.to_csv('database/disk.csv', index=False)

# Verifica o resultado
print(df.head())