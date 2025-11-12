1. **Stwórz środowisko wirtualne**

    ```bash
   python -m venv .venv
    ```

2. **Uruchom środowisko wirtualne**

    ```bash
   .venv\Scripts\activate
    ```

3. **Zainstaluj wymagane biblioteki**

    ```bash
   pip install -r requirements.txt
    ```

4. **Stwórz wymagany folder "Data"**

    ```bash
    New-Item -Name "Data" -ItemType Directory
    ```

5. **Uruchom skrypty jeden po drugim**

    ```bash
    python <nazwa_pliku>.py
    ```