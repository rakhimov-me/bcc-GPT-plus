# tools/remove_new-mock.py
import os
from pathlib import Path
import pandas as pd

CSV_DIR = Path("csv")
CLIENTS_PATH = CSV_DIR / "clients.csv"

def remove_client_files():
    removed = []
    if not CSV_DIR.exists():
        return removed
    for f in CSV_DIR.glob("client_*_*.csv"):
        try:
            # имя вида client_61_transactions_3m.csv → берём id
            parts = f.stem.split("_")
            if len(parts) >= 2 and parts[1].isdigit():
                cid = int(parts[1])
                if cid > 60:
                    os.remove(f)
                    removed.append(f.name)
        except Exception as e:
            print(f"Не удалось удалить {f}: {e}")
    return removed

def clean_clients_csv():
    if not CLIENTS_PATH.exists():
        print("Файл clients.csv не найден, пропускаем очистку.")
        return None
    try:
        df = pd.read_csv(CLIENTS_PATH)
        before = len(df)
        df = df[df["client_code"] <= 60]  # оставляем только <=60
        after = len(df)
        df.to_csv(CLIENTS_PATH, index=False, encoding="utf-8-sig")
        return before - after
    except Exception as e:
        print(f"Ошибка при очистке clients.csv: {e}")
        return None

def main():
    print("Удаляем mock-данные для всех клиентов с id > 60...")
    removed_files = remove_client_files()
    cleaned = clean_clients_csv()
    print(f"Удалено файлов: {len(removed_files)}")
    for f in removed_files:
        print(f"  - {f}")
    if cleaned is not None:
        print(f"Из clients.csv удалено записей: {cleaned}")

if __name__ == "__main__":
    main()
