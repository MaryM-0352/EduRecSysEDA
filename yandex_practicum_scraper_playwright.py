#!/usr/bin/env python3
"""Скрапер каталога Яндекс Практикума через Playwright с ручным прохождением капчи.

Идея:
- скрипт открывает обычный браузер (headless=False по умолчанию);
- если появляется SmartCaptcha, пользователь проходит её вручную;
- после этого скрипт продолжает работу в той же browser session / cookies;
- HTML каталога и страниц курсов сохраняются в локальный кэш;
- результат пишется в CSV.

Что собирается:
- name
- description
- duration
- full_price
- monthly_price
- audience_tags
- is_free
- author
- url

Примеры:
    python yandex_practicum_scraper_playwright.py
    python yandex_practicum_scraper_playwright.py --output courses.csv
    python yandex_practicum_scraper_playwright.py --cache-dir ./cache --storage-state state.json
    python yandex_practicum_scraper_playwright.py --catalog-html ./cache/catalog.html --pages-dir ./cache/pages

Перед первым запуском:
    pip install -r requirements_playwright.txt
    playwright install chromium
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

from bs4 import BeautifulSoup
from playwright.sync_api import (
    BrowserContext,
    Error as PlaywrightError,
    Page,
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)

CATALOG_URL = "https://practicum.yandex.ru/catalog/"
AUTHOR = "Yandex.Practicum"
DEFAULT_OUTPUT = "yandex_practicum_courses.csv"
DEFAULT_CACHE_DIR = "practicum_cache"
DEFAULT_PAGES_DIR_NAME = "pages"


@dataclass
class CatalogCard:
    url: str
    category: str = ""
    title: str = ""
    footer: str = ""
    raw_text: str = ""
    is_free_hint: bool = False


@dataclass
class CourseRow:
    name: str
    description: str
    duration: str
    full_price: Optional[int]
    monthly_price: Optional[int]
    audience_tags: str
    is_free: bool
    author: str
    url: str


class CaptchaStillPresentError(RuntimeError):
    pass


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalize_url(url: str, base_url: str = CATALOG_URL) -> str:
    url = normalize_whitespace(url)
    if not url:
        return ""

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) and not url.startswith("/"):
        url = "/" + url.lstrip("./")

    url = urljoin(base_url, url)

    parsed = urlparse(url)
    clean = parsed._replace(params="", query="", fragment="")
    normalized = urlunparse(clean)
    if normalized.endswith("/") and parsed.path not in ("", "/"):
        normalized = normalized[:-1]
    return normalized


def slug_from_url(url: str) -> str:
    path = urlparse(url).path.rstrip("/")
    return path.split("/")[-1] or "index"


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "page"


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def is_probably_course_url(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if not path or path == "/catalog":
        return False
    if any(part in path.lower() for part in ["showcaptcha", "/legal", "/contacts", "/events"]):
        return False
    if parsed.netloc and "practicum.yandex" not in parsed.netloc and "start.practicum.yandex" not in parsed.netloc:
        return False
    return True


def detect_captcha_text(html: str) -> bool:
    lowered = html.lower()
    return any(token in lowered for token in ["showcaptcha", "smartcaptcha", "not a robot", "я не робот"])


def lines_from_html(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    raw_lines = soup.get_text("\n", strip=True).splitlines()
    lines = [normalize_whitespace(line) for line in raw_lines]
    return [line for line in lines if line]


def extract_meta_description(soup: BeautifulSoup) -> str:
    for attrs in (
        {"name": "description"},
        {"property": "og:description"},
        {"name": "twitter:description"},
    ):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            return normalize_whitespace(tag["content"])
    return ""


def extract_title(soup: BeautifulSoup, fallback: str = "") -> str:
    candidates: List[str] = []

    h1 = soup.find("h1")
    if h1:
        h1_text = normalize_whitespace(h1.get_text(" ", strip=True))
        generic_h1 = {"Программа курса", "Program", "Курс"}
        if h1_text and h1_text not in generic_h1:
            candidates.append(h1_text)

    if fallback:
        candidates.append(fallback)

    title_tag = soup.title
    if title_tag:
        title_text = normalize_whitespace(title_tag.get_text(" ", strip=True))
        title_text = re.sub(r"\s*[—-]\s*Яндекс Практикум.*$", "", title_text, flags=re.I)
        title_text = re.sub(r"^Онлайн-курс\s*[«\"]?", "", title_text, flags=re.I)
        title_text = re.sub(r"^Бесплатный курс\s*[«\"]?", "", title_text, flags=re.I)
        title_text = title_text.strip("«»\" ")
        candidates.append(title_text)

    for candidate in candidates:
        if candidate:
            return candidate
    return ""


def parse_int(value: str) -> Optional[int]:
    digits = re.sub(r"\D", "", value or "")
    return int(digits) if digits else None


def extract_tariffs_from_lines(lines: Sequence[str]) -> List[Dict[str, Optional[int]]]:
    start_idx = 0
    for i, line in enumerate(lines):
        if "тарифы" in line.lower():
            start_idx = i
            break

    tariffs: List[Dict[str, Optional[int]]] = []
    idx = start_idx
    while idx < len(lines):
        line = lines[idx]
        if re.search(r"\d[\d\s]*\s*₽/мес", line):
            monthly = parse_int(line)
            full_price = None
            duration = None
            tariff_name = ""

            for back in range(1, 5):
                j = idx - back
                if j < 0:
                    break
                prev_line = lines[j]
                if prev_line and not re.search(r"₽|%|Скачать|Попробовать", prev_line, flags=re.I):
                    tariff_name = prev_line
                    break

            for look_ahead in range(1, 16):
                j = idx + look_ahead
                if j >= len(lines):
                    break
                probe = lines[j]
                if full_price is None and ("или" in probe.lower() or "целиком" in probe.lower()) and "₽" in probe:
                    full_price = parse_int(probe)
                if duration is None and re.search(r"\d+\s+месяц(?:а|ев)?\s+обучения", probe):
                    m = re.search(r"(\d+)\s+месяц(?:а|ев)?", probe)
                    if m:
                        duration = int(m.group(1))
                    else:
                        m = re.search(r"(\d+)\s+час(?:а|ов)?", probe)
                        if m:
                            duration = int(m.group(1))
                if full_price is not None and duration is not None:
                    break

            tariffs.append(
                {
                    "name": tariff_name,
                    "monthly_price": monthly,
                    "full_price": full_price,
                    "duration_months": duration,
                }
            )
            idx += 8
            continue
        idx += 1

    deduped: List[Dict[str, Optional[int]]] = []
    seen = set()
    for tariff in tariffs:
        key = (tariff.get("name"), tariff.get("monthly_price"), tariff.get("full_price"), tariff.get("duration_months"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tariff)
    return deduped


def extract_duration(description: str, lines: Sequence[str], fallback_footer: str = "") -> str:
    candidates: List[str] = []
    for text in (description, " ".join(lines[:120]), fallback_footer):
        if not text:
            continue
        m = re.search(r"(\d+)\s+месяц(?:а|ев)?", text, flags=re.I)
        if m:
            candidates.append(f"{m.group(1)} месяцев")
        else:
            m = re.search(r"(\d+)\s+час(?:а|ов)?", text, flags=re.I)
            if m:
                candidates.append(f"{m.group(1)} месяцев")
    return candidates[0] if candidates else ""


def clean_description(text: str, title: str = "") -> str:
    text = normalize_whitespace(text)
    if title:
        text = text.replace(title, "").strip(" —:-")
        text = normalize_whitespace(text)
    return text


KEYWORD_TAGS: List[Tuple[str, Sequence[str]]] = [
    ("AUD_ANALYST", ["аналит", "data", "sql", "bi", "статист", "дашборд", "ab-тест", "a/b", "datalens"]),
    ("AUD_DEVELOPER", ["разработ", "developer", "backend", "frontend", "fullstack", "devops", "python", "java", "go", "node", "react", "1с", "qa ", "тестирован"]),
    ("AUD_DESIGN", ["дизайн", "designer", "ux", "ui", "графическ", "моушн", "3d", "арт-дир"]),
    ("AUD_MARKETING", ["маркет", "smm", "контент", "бренд", "трафик", "реклам", "marketplace"]),
    ("AUD_MANAGEMENT", ["менедж", "management", "руковод", "cto", "cpo", "продакт", "hr", "рекрутер", "sales", "lead"]),
    ("AUD_FINANCE", ["финанс", "юнит-эконом", "экономик", "бюджет", "p&l"]),
    ("AUD_EDUCATION", ["образован", "наставник", "обучени", "педагог", "преподав"]),
    ("AUD_ENTREPRENEUR", ["предприним", "бизнес", "ип", "свой бизнес"]),
    ("AUD_CAREER", ["карьер", "резюме", "собеседован", "трудоустрой", "поиск работы", "оффер"]),
    ("AUD_AI", ["нейросет", "ии", " ai ", "machine learning", "ml", "nlp", "cv"]),
    ("AUD_ENGLISH", ["английск", "intermediate", "beginner", "elementary"]),
    ("AUD_SECURITY", ["безопасн", "soc", "secops"]),
]


def infer_audience_tags(name: str, description: str, category: str, card_text: str) -> List[str]:
    bag = f" {name} {description} {category} {card_text} ".lower()
    tags: List[str] = []

    if any(token in bag for token in ["с нуля", "для начинающих", "без опыта", "основы", "выбрать профессию", "начинающ"]):
        tags.append("AUD_BEGINNER")

    if any(token in bag for token in [" pro ", "middle", "мидл", "lead", "архитектор", "с опытом", "для действующих"]):
        tags.append("AUD_ADVANCED")

    for tag, keywords in KEYWORD_TAGS:
        if any(keyword in bag for keyword in keywords):
            tags.append(tag)

    if not tags:
        tags.append("AUD_GENERAL")
    elif len(tags) == 1 and "AUD_GENERAL" not in tags:
        tags.append("AUD_GENERAL")

    deduped: List[str] = []
    seen = set()
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    return deduped


def choose_best_price(tariffs: Sequence[Dict[str, Optional[int]]]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    clean = [t for t in tariffs if t.get("full_price") is not None or t.get("monthly_price") is not None]
    if not clean:
        return None, None, None

    best = sorted(
        clean,
        key=lambda item: (
            item.get("full_price") if item.get("full_price") is not None else 10**12,
            item.get("monthly_price") if item.get("monthly_price") is not None else 10**12,
        ),
    )[0]
    return best.get("full_price"), best.get("monthly_price"), best.get("duration_months")


def parse_course_page(html: str, url: str, card: CatalogCard) -> CourseRow:
    soup = BeautifulSoup(html, "html.parser")
    lines = lines_from_html(html)

    description = clean_description(extract_meta_description(soup), title=card.title)
    title = extract_title(soup, fallback=card.title)
    if description:
        description = clean_description(description, title=title)

    tariffs = extract_tariffs_from_lines(lines)
    full_price, monthly_price, duration_from_tariff = choose_best_price(tariffs)

    is_free = card.is_free_hint or "бесплат" in f"{title} {description}".lower()
    if is_free and full_price is None:
        full_price = 0
    if is_free and monthly_price is None:
        monthly_price = 0

    if duration_from_tariff:
        duration = f"{duration_from_tariff} месяцев"
    else:
        duration = extract_duration(description, lines, fallback_footer=card.footer)

    audience_tags = infer_audience_tags(title, description, card.category, card.raw_text)

    return CourseRow(
        name=title or card.title,
        description=description,
        duration=duration,
        full_price=full_price,
        monthly_price=monthly_price,
        audience_tags=";".join(audience_tags),
        is_free=bool(is_free),
        author=AUTHOR,
        url=url,
    )


def extract_catalog_cards(html: str) -> List[CatalogCard]:
    soup = BeautifulSoup(html, "html.parser")
    cards: List[CatalogCard] = []
    seen: set[str] = set()

    candidate_anchors = soup.select("a[href*='?from=catalog']")
    for a in candidate_anchors:
        href = a.get("href")
        if not href:
            continue
        href = normalize_url(href, base_url=CATALOG_URL)
        if not is_probably_course_url(href) or href in seen:
            continue

        text = normalize_whitespace(a.get_text(" ", strip=True))
        if not text:
            continue

        category = normalize_whitespace(" ".join(el.get_text(" ", strip=True) for el in a.select(".prof-window-v2__card-direction")))
        title = normalize_whitespace(" ".join(el.get_text(" ", strip=True) for el in a.select(".prof-window-v2__card-title")))
        footer = normalize_whitespace(" ".join(el.get_text(" ", strip=True) for el in a.select(".prof-window-v2__card-footer")))

        cards.append(
            CatalogCard(
                url=href,
                category=category,
                title=title,
                footer=footer,
                raw_text=text,
                is_free_hint="бесплат" in text.lower(),
            )
        )
        seen.add(href)

    if not cards:
        for a in soup.find_all("a", href=True):
            href = normalize_url(a["href"], base_url=CATALOG_URL)
            if not is_probably_course_url(href) or href in seen:
                continue
            text = normalize_whitespace(a.get_text(" ", strip=True))
            if not text or len(text) < 6:
                continue
            cards.append(CatalogCard(url=href, raw_text=text, is_free_hint="бесплат" in text.lower()))
            seen.add(href)

    return cards


def save_csv(rows: Sequence[CourseRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "description",
        "duration",
        "full_price",
        "monthly_price",
        "audience_tags",
        "is_free",
        "author",
        "url",
    ]

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def auto_scroll(page: Page, rounds: int = 20, pause_sec: float = 0.8) -> None:
    previous_height = -1
    stable_rounds = 0
    for _ in range(rounds):
        try:
            height = page.evaluate("document.body.scrollHeight")
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            page.wait_for_timeout(int(pause_sec * 1000))
        except PlaywrightError:
            break

        if height == previous_height:
            stable_rounds += 1
        else:
            stable_rounds = 0
        previous_height = height
        if stable_rounds >= 2:
            break

    try:
        page.evaluate("window.scrollTo(0, 0)")
        page.wait_for_timeout(300)
    except PlaywrightError:
        pass


def prompt_to_continue(message: str) -> None:
    print(message)
    try:
        input("Нажми Enter после этого... ")
    except EOFError:
        time.sleep(5)


def ensure_page_without_captcha(page: Page, url: str, save_to: Optional[Path] = None) -> str:
    attempts = 0
    while attempts < 10:
        attempts += 1
        try:
            page.wait_for_load_state("domcontentloaded", timeout=30000)
        except PlaywrightTimeoutError:
            pass

        page.wait_for_timeout(1200)
        html = page.content()
        if not detect_captcha_text(html):
            if save_to is not None:
                write_text_file(save_to, html)
            return html

        prompt_to_continue(
            f"[ACTION] На странице обнаружена капча: {url}\n"
            "Реши её вручную в окне браузера. Когда откроется нормальная страница, возвращайся сюда."
        )
        try:
            page.wait_for_load_state("networkidle", timeout=10000)
        except PlaywrightTimeoutError:
            pass

    raise CaptchaStillPresentError(f"Капча всё ещё показывается для {url}")


def goto_with_retries(page: Page, url: str, attempts: int = 3) -> None:
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=45000)
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[WARN] Ошибка открытия {url} (попытка {attempt}/{attempts}): {exc}", file=sys.stderr)
            page.wait_for_timeout(1500 * attempt)
    assert last_error is not None
    raise last_error


def save_storage_state(context: BrowserContext, storage_state_path: Optional[Path]) -> None:
    if not storage_state_path:
        return
    storage_state_path.parent.mkdir(parents=True, exist_ok=True)
    context.storage_state(path=str(storage_state_path))


def load_catalog_html(args: argparse.Namespace, catalog_html_path: Path) -> Optional[str]:
    if args.catalog_html:
        return read_text_file(Path(args.catalog_html))
    if args.use_cache and catalog_html_path.exists():
        return read_text_file(catalog_html_path)
    return None


def load_cached_course_html(pages_dir: Path, url: str) -> Optional[str]:
    slug = safe_filename(slug_from_url(url))
    for candidate in [pages_dir / f"{slug}.html", pages_dir / f"{slug}.htm"]:
        if candidate.exists():
            return read_text_file(candidate)
    return None


def open_catalog_in_browser(page: Page, context: BrowserContext, args: argparse.Namespace, catalog_html_path: Path) -> str:
    goto_with_retries(page, args.catalog_url)
    ensure_page_without_captcha(page, args.catalog_url)

    prompt_to_continue(
        "[ACTION] Если каталог ещё не догрузился полностью, прокрути страницу вниз до конца или просто дождись, пока карточки появятся."
    )
    auto_scroll(page)
    html = ensure_page_without_captcha(page, args.catalog_url, save_to=catalog_html_path)
    save_storage_state(context, Path(args.storage_state) if args.storage_state else None)
    return html


def scrape_live(args: argparse.Namespace, cache_dir: Path, pages_dir: Path) -> Tuple[List[CourseRow], List[str], str]:
    catalog_html_path = cache_dir / "catalog.html"
    cached_catalog_html = load_catalog_html(args, catalog_html_path)

    rows: List[CourseRow] = []
    errors: List[str] = []

    with sync_playwright() as p:
        browser_kwargs = {"headless": args.headless}
        browser = p.chromium.launch(**browser_kwargs)

        context_kwargs = {}
        storage_state_path = Path(args.storage_state) if args.storage_state else None
        if storage_state_path and storage_state_path.exists():
            context_kwargs["storage_state"] = str(storage_state_path)

        context = browser.new_context(**context_kwargs)
        context.set_default_timeout(45000)
        page = context.new_page()

        if cached_catalog_html is not None:
            catalog_html = cached_catalog_html
            print(f"[INFO] Каталог взят из кэша: {catalog_html_path if catalog_html_path.exists() else args.catalog_html}")
        else:
            print("[INFO] Открываю каталог в браузере...")
            catalog_html = open_catalog_in_browser(page, context, args, catalog_html_path)
            print(f"[INFO] Каталог сохранён в: {catalog_html_path}")

        cards = extract_catalog_cards(catalog_html)
        if args.limit:
            cards = cards[: args.limit]
        if not cards:
            browser.close()
            raise RuntimeError("В каталоге не найдено ни одной карточки курса.")

        print(f"[INFO] Найдено карточек: {len(cards)}")

        course_page = context.new_page()
        course_page.set_default_timeout(45000)

        for index, card in enumerate(cards, start=1):
            slug = safe_filename(slug_from_url(card.url))
            html_path = pages_dir / f"{slug}.html"
            print(f"[INFO] [{index}/{len(cards)}] {card.url}")

            try:
                if args.use_cache and html_path.exists():
                    html = read_text_file(html_path)
                    print(f"[CACHE] {html_path}")
                else:
                    goto_with_retries(course_page, card.url)
                    html = ensure_page_without_captcha(course_page, card.url, save_to=html_path)
                    save_storage_state(context, storage_state_path)
                    course_page.wait_for_timeout(int(args.delay * 1000))

                row = parse_course_page(html, card.url, card)
                rows.append(row)
                print(f"[OK] {row.name}")
            except Exception as exc:  # noqa: BLE001
                error = f"{card.url}: {exc}"
                errors.append(error)
                print(f"[WARN] {error}", file=sys.stderr)

        browser.close()
        return rows, errors, catalog_html


def parse_offline(args: argparse.Namespace, cache_dir: Path, pages_dir: Path) -> Tuple[List[CourseRow], List[str]]:
    catalog_html_path = Path(args.catalog_html) if args.catalog_html else cache_dir / "catalog.html"
    if not catalog_html_path.exists():
        raise FileNotFoundError(f"Не найден HTML каталога: {catalog_html_path}")

    catalog_html = read_text_file(catalog_html_path)
    cards = extract_catalog_cards(catalog_html)
    if args.limit:
        cards = cards[: args.limit]

    rows: List[CourseRow] = []
    errors: List[str] = []
    for index, card in enumerate(cards, start=1):
        print(f"[INFO] [{index}/{len(cards)}] offline {card.url}")
        try:
            html = load_cached_course_html(pages_dir, card.url)
            if not html:
                raise FileNotFoundError(f"Нет HTML в кэше для {card.url}")
            rows.append(parse_course_page(html, card.url, card))
        except Exception as exc:  # noqa: BLE001
            error = f"{card.url}: {exc}"
            errors.append(error)
            print(f"[WARN] {error}", file=sys.stderr)
    return rows, errors


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Скрапер курсов Яндекс Практикума через Playwright")
    parser.add_argument("--catalog-url", default=CATALOG_URL, help="URL каталога")
    parser.add_argument("--catalog-html", help="Локальный HTML каталога")
    parser.add_argument("--pages-dir", help="Папка с HTML страниц курсов")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR, help="Папка кэша")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="CSV-файл результата")
    parser.add_argument("--delay", type=float, default=0.8, help="Пауза между курсами в секундах")
    parser.add_argument("--limit", type=int, default=0, help="Ограничение числа курсов для теста")
    parser.add_argument("--storage-state", help="JSON-файл Playwright storage state для повторных запусков")
    parser.add_argument("--use-cache", action="store_true", help="Использовать ранее сохранённый HTML кэш")
    parser.add_argument("--offline", action="store_true", help="Только распарсить уже сохранённые HTML без браузера")
    parser.add_argument("--headless", action="store_true", help="Запускать браузер без окна")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    cache_dir = Path(args.cache_dir)
    pages_dir = Path(args.pages_dir) if args.pages_dir else cache_dir / DEFAULT_PAGES_DIR_NAME
    output_path = Path(args.output)
    pages_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.offline:
            rows, errors = parse_offline(args, cache_dir, pages_dir)
        else:
            rows, errors, _ = scrape_live(args, cache_dir, pages_dir)
    except KeyboardInterrupt:
        print("\n[ERROR] Остановлено пользователем.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    rows.sort(key=lambda item: item.name.lower())
    save_csv(rows, output_path)
    print(f"[INFO] Сохранено строк: {len(rows)} -> {output_path}")

    if errors:
        err_path = output_path.with_suffix(".errors.txt")
        err_path.write_text("\n".join(errors), encoding="utf-8")
        print(f"[INFO] Ошибок: {len(errors)} -> {err_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
