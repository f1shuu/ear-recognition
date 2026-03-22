import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

# Wczytanie danych
df = pd.read_csv("data/ear_data.csv")

# Wybranie dowolnych użytkowników
selected_users = [22, 37, 44, 47, 48]
df_filtered = df[df[df.columns[-1]].isin(selected_users)].copy()

# Podzielenie danych i etykiet
X = df_filtered.iloc[:, :-1].values
y = df_filtered.iloc[:, -1].values

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# t-SNE
tsne = TSNE(n_components=2, random_state=0, perplexity=10, max_iter=1000)
X_tsne = tsne.fit_transform(X)


# Funkcja rysująca
def plot_embedding(X_embedded, y, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y, palette="Set1", s=70)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(title="User ID")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Rysowanie wykresów
plot_embedding(X_pca, y, "PCA (5 users)")
plot_embedding(X_tsne, y, "t-SNE (5 users)")
