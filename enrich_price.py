import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

# ==========================
# НАСТРОЙКИ
# ==========================
INPUT_CSV = "yandex_practicum_courses.csv"
PAGES_DIR = "practicum_cache/pages"   # папка с html-кэшем страниц
OUTPUT_CSV = "yandex_practicum_courses_filled_from_cache.csv"
ENCODING = "utf-8-sig"

SKIP_SLUGS = {
    "promo",
    "sale",
    "discount",
    "callback",
}

# ==========================
# УТИЛИТЫ
# ==========================
def is_empty(value) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def extract_slug_from_url(url: str):
    if not url or pd.isna(url):
        return None
    path = urlparse(str(url)).path.strip("/")
    if not path:
        return None
    return path.split("/")[0].lower()


def parse_duration_months(value):
    if value is None or is_empty(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).lower().strip().replace(",", ".")

    m = re.search(r"(\d+(?:\.\d+)?)\s*(месяц|месяца|месяцев|мес\.?)", text)
    if m:
        return float(m.group(1))

    y = re.search(r"(\d+(?:\.\d+)?)\s*(год|года|лет)", text)
    if y:
        return float(y.group(1)) * 12

    return None


def parse_int_or_none(value: str):
    if value is None or value == "null":
        return None
    return int(value)


def find_html_file_for_slug(slug: str, pages_dir: Path):
    """
    Ищет html по имени файла:
    1) slug.html
    2) любой *slug*.html
    3) первый html, внутри которого встречается '"slug":"<slug>"'
    """
    direct = pages_dir / f"{slug}.html"
    if direct.exists():
        return direct

    matches = list(pages_dir.glob(f"*{slug}*.html"))
    if matches:
        return matches[0]

    # медленный fallback: поиск по содержимому
    marker = f'"slug":"{slug}"'
    for file in pages_dir.glob("*.html"):
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
            if marker in text:
                return file
        except Exception:
            pass

    return None


def extract_profession_window(text: str, marker: str, window_size: int = 15000):
    pos = text.find(marker)
    if pos == -1:
        return None
    return text[pos:pos + window_size]


def extract_fields_from_window(window_text: str):
    """
    Ищет поля внутри ограниченного куска текста.
    Нам не нужен полный JSON parse.
    """
    if not window_text:
        return None

    slug_match = re.search(r'"slug":"([^"]+)"', window_text)
    price_match = re.search(r'"price":(\d+|null)', window_text)
    partial_match = re.search(r'"partial_price":(\d+|null)', window_text)
    bank_match = re.search(r'"bank_installment_price":(\d+|null)', window_text)
    duration_match = re.search(r'"duration":(\d+(?:\.\d+)?)', window_text)
    name_match = re.search(r'"name":"([^"]+)"', window_text)

    if not slug_match:
        return None

    return {
        "slug": slug_match.group(1),
        "name": name_match.group(1) if name_match else None,
        "price": parse_int_or_none(price_match.group(1)) if price_match else None,
        "partial_price": parse_int_or_none(partial_match.group(1)) if partial_match else None,
        "bank_installment_price": parse_int_or_none(bank_match.group(1)) if bank_match else None,
        "duration": float(duration_match.group(1)) if duration_match else None,
    }


def extract_prices_from_preloaded_data(html_text: str, expected_slug: str = None):
    """
    Сначала пробуем:
      window.__preloadedData__ -> apiData -> getV2ProfessionBySlug

    Если не нашли — fallback:
      professionsV2Reducer -> profession

    Никакого json.loads, только regex по локальному окну текста.
    """
    markers = [
        '"getV2ProfessionBySlug":{',
        '"profession":{"id":',
    ]

    candidates = []

    for marker in markers:
        window_text = extract_profession_window(html_text, marker)
        data = extract_fields_from_window(window_text) if window_text else None
        if data:
            candidates.append(data)

    if not candidates:
        return None

    # Если знаем ожидаемый slug — выберем точное совпадение
    if expected_slug:
        expected_slug = expected_slug.lower()
        for item in candidates:
            if item.get("slug", "").lower() == expected_slug:
                return item

    # Иначе просто первый нормальный
    return candidates[0]


# ==========================
# ОСНОВНАЯ ЛОГИКА
# ==========================
df = pd.read_csv(INPUT_CSV, encoding=ENCODING)
pages_dir = Path(PAGES_DIR)

if not pages_dir.exists():
    raise FileNotFoundError(f"Папка с html не найдена: {pages_dir}")

if "monthly_price" not in df.columns:
    df["monthly_price"] = None

if "full_price" not in df.columns:
    df["full_price"] = None

updated_rows = 0
not_found_files = 0
not_found_data = 0
skipped_rows = 0

for idx, row in df.iterrows():
    monthly_empty = is_empty(row.get("monthly_price"))
    full_empty = is_empty(row.get("full_price"))

    # Фильтр строк оставляем таким же
    if not (monthly_empty or full_empty):
        continue

    slug = extract_slug_from_url(row.get("url"))
    if not slug or slug in SKIP_SLUGS:
        skipped_rows += 1
        print(f"[SKIP] slug={slug} | url={row.get('url')}")
        continue

    html_file = find_html_file_for_slug(slug, pages_dir)
    if html_file is None:
        not_found_files += 1
        print(f"[NO HTML] slug={slug}")
        continue

    html_text = html_file.read_text(encoding="utf-8", errors="ignore")
    extracted = extract_prices_from_preloaded_data(html_text, expected_slug=slug)

    if not extracted:
        not_found_data += 1
        print(f"[NO PRELOADED DATA] slug={slug} | file={html_file.name}")
        continue

    full_price = extracted.get("price")
    monthly_price = extracted.get("partial_price")

    if monthly_price is None:
        monthly_price = extracted.get("bank_installment_price")

    duration_months = parse_duration_months(row.get("duration"))

    if duration_months is None:
        duration_months = extracted.get("duration")

    if monthly_price is None and full_price is not None and duration_months:
        monthly_price = round(full_price / duration_months)

    if full_price is None and monthly_price is not None and duration_months:
        full_price = round(monthly_price * duration_months)

    row_updated = False

    if monthly_empty and monthly_price is not None:
        df.at[idx, "monthly_price"] = int(monthly_price)
        row_updated = True

    if full_empty and full_price is not None:
        df.at[idx, "full_price"] = int(full_price)
        row_updated = True

    if row_updated:
        updated_rows += 1
        print(
            f"[UPDATED] slug={slug} | "
            f"file={html_file.name} | "
            f"monthly_price={df.at[idx, 'monthly_price']} | "
            f"full_price={df.at[idx, 'full_price']}"
        )
    else:
        print(
            f"[NO UPDATE] slug={slug} | "
            f"file={html_file.name} | "
            f"found_monthly={monthly_price} | "
            f"found_full={full_price}"
        )

df.to_csv(OUTPUT_CSV, index=False, encoding=ENCODING)

print("\n=== ГОТОВО ===")
print(f"Обновлено строк: {updated_rows}")
print(f"Пропущено служебных slug: {skipped_rows}")
print(f"Не найден html: {not_found_files}")
print(f"Не найдено данных в preloadedData: {not_found_data}")
print(f"Сохранено в: {OUTPUT_CSV}")