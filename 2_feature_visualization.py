import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

#############
# Parametry #
#############

INPUT_DIR = "data/photos_ear_data.csv" # Domyślna ścieżka do pliku z danymi cech
SELECTED_USERS = [5, 7, 12, 19, 20] # Lista wybranych użytkowników do wizualizacji

#####################
# Główna część kodu #
#####################

# Wybór pliku CSV z danymi cech
user_input = ""
while not os.path.isfile("data/" + user_input):
    print("Wybierz plik CSV:")
    if os.path.isdir("data"):
        print("Pliki dostępne w /data:")
        for file in os.listdir("data"):
            print(f"  - {file}")
    else:
        print("Brak katalogu 'data' lub brak plików w tym katalogu.")
        print("Najprawdopodobniej pominięto krok ekstrakcji cech.")
        print("Proszę najpierw uruchomić opcję [2] - Ekstrakcja cech.")
        sys.exit(0)
    
    user_input = input("\n> ").strip()
    if not user_input.endswith(".csv"):
        user_input += ".csv"
    if user_input not in os.listdir("data"):
        print(f"Błąd: Plik '{user_input}' nie istnieje. Spróbuj ponownie.")
    else:
        INPUT_DIR = "data/" + user_input

# Wczytanie danych
df = pd.read_csv(INPUT_DIR)

# Wybranie dowolnych użytkowników
df_filtered = df[df[df.columns[-1]].isin(SELECTED_USERS)].copy()

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
