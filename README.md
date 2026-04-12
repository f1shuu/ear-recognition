### Opis

Niniejsze repozytorium stanowi część badawczą pracy magisterskiej pod tytułem **"Analiza skuteczności identyfikacji osób na podstawie danych biometrycznych ucha z wykorzystaniem metod uczenia maszynowego"**. Kod składa się z kilku plików realizujących kolejne etapy tego procesu.

**0_data_augmentation.py** przeprowadza proces augmentacji danych, czyli zwiększenia liczby próbek poprzez rotacje i nakładanie zakłóceń.

**1_feature_extraction.py** ekstrahuje z przygotowanych obrazów 19 różnych parametrów, a następnie umieszcza je w pliku CSV wraz z parametrem decyzyjnym - ID użytkownika (właściciela danego ucha).

**2_feature_visualization.py** #TODO

**3_classification.py** #TODO

**main.py** stanowi prosty program konsolowy, upraszczający proces uruchamiania dostępnych skryptów jeden po drugim.

### Uruchomienie

Aby uruchomić projekt lokalnie, należy wykonać poniższe kroki.

1. Stworzenie środowiska wirtualnego

    ```bash
   python -m venv .venv
    ```

2. Uruchomienie środowiska wirtualnego

   ```bash
   .venv/Scripts/activate
   ```

3. Zainstalowanie wymaganych bibliotek, których lista znajduje się w pliku `requirements.txt`

    ```bash
   pip install -r requirements.txt
    ```

4. Uruchomienie wybranego skryptu

    ```bash
    python <nazwa_pliku>.py
    ```
