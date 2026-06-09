import pandas as pd

# ==========================
# НАСТРОЙКИ
# ==========================
INPUT_FILE = "yandex_practicum_courses.csv"
OUTPUT_FILE = "yandex_practicum_courses_no_empty_duration.csv"
ENCODING = "utf-8-sig"

# ==========================
# ЗАГРУЗКА CSV
# ==========================
df = pd.read_csv(INPUT_FILE, encoding=ENCODING)

before_count = len(df)

# ==========================
# УДАЛЯЕМ ПУСТОЙ duration
# ==========================
mask_not_empty = (
    df["full_price"].notna() &
    (df["full_price"].astype(str).str.strip() != "")
)

df_clean = df[mask_not_empty].copy()

after_count = len(df_clean)
deleted_count = before_count - after_count

# ==========================
# СОХРАНЕНИЕ
# ==========================
df_clean.to_csv(OUTPUT_FILE, index=False, encoding=ENCODING)

print(f"Исходных строк: {before_count}")
print(f"Удалено строк с пустым full_price: {deleted_count}")
print(f"Осталось строк: {after_count}")
print(f"Файл сохранён: {OUTPUT_FILE}")