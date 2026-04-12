import subprocess
import sys

def display_menu():
    print("\n" + "="*30)
    print("Wybierz opcję do uruchomienia:")
    print("="*30)
    print("[1] - Augmentacja danych")
    print("[2] - Ekstrakcja cech")
    print("[3] - Wizualizacja cech")
    print("[4] - Klasyfikacja*")
    print("\n[0] - Wyjście z programu")
    print("="*30)

def run_program(option):
    if option == "1":
        subprocess.run([sys.executable, "0_data_augmentation.py"])
    elif option == "2":
        subprocess.run([sys.executable, "1_feature_extraction.py"])
    elif option == "3":
        subprocess.run([sys.executable, "2_feature_visualization.py"])
    elif option == "4":
        print("\nTa opcja nie została jeszcze w pełni zaimplementowana.")
        # subprocess.run([sys.executable, "3_classification.py"])
    elif option == "0":
        sys.exit(0)
    else:
        print("\nNieprawidłowa opcja. Proszę wybrać 0, 1, 2, 3, lub 4.")

def main():
    while True:
        display_menu()
        user_input = input("\n> ").strip()
        run_program(user_input)

if __name__ == "__main__":
    main()
