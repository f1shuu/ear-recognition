### Opis

Niniejsze repozytorium stanowi część badawczą pracy magisterskiej pod tytułem **"Analiza skuteczności identyfikacji osób na podstawie danych biometrycznych ucha z wykorzystaniem metod uczenia maszynowego"**. Kod składa się z kilku plików realizujących kolejne etapy tego procesu.

**0_data_augmentation.py** przeprowadza proces augmentacji danych, czyli zwiększenia liczby próbek poprzez rotacje i nakładanie zakłóceń.

**1_feature_extraction.py** #TODO

**2_feature_visualization.py** #TODO

**3_classification.py** #TODO

### Uruchomienie

Aby uruchomić projekt lokalnie, należy wykonać poniższe kroki.

1. Stworzenie środowiska wirtualnego

    ```bash
   python -m venv .venv
    ```

2. Uruchomienie środowiska wirtualnego

   ```bash
   source .venv/bin/activate
   ```

3. Zainstalowanie wymaganych bibliotek, których lista znajduje się w pliku `requirements.txt`

    ```bash
   pip install -r requirements.txt
    ```

4. Uruchomienie wybranego skryptu

    ```bash
    python <nazwa_pliku>.py
    ```

**Można też uruchomić wszystkie skrypty po kolei jednym poleceniem:**

    python main.py