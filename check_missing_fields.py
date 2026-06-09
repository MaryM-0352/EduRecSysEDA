#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List

import pandas as pd


def find_missing_fields(row: pd.Series) -> List[str]:
    missing = []
    for col, value in row.items():
        if pd.isna(value):
            missing.append(col)
        elif isinstance(value, str) and value.strip() == "":
            missing.append(col)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Проверка пропущенных полей в yandex_practicum_courses.csv"
    )
    parser.add_argument(
        "--input",
        default="yandex_practicum_courses.csv",
        help="Путь к CSV-файлу",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="Кодировка CSV (по умолчанию utf-8-sig)",
    )
    parser.add_argument(
        "--sep",
        default=",",
        help="Разделитель CSV (по умолчанию ,)",
    )
    parser.add_argument(
        "--save-row-report",
        default="",
        help="Если указан путь, сохранить построчный отчёт туда",
    )
    parser.add_argument(
        "--save-summary-report",
        default="",
        help="Если указан путь, сохранить сводку туда",
    )
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.exists():
        raise FileNotFoundError(f"Файл не найден: {csv_path}")

    df = pd.read_csv(csv_path, encoding=args.encoding, sep=args.sep)

    print(f"Файл: {csv_path}")
    print(f"Строк: {len(df)}")
    print(f"Колонок: {len(df.columns)}")
    print()

    row_report = []
    for idx, row in df.iterrows():
        missing_fields = find_missing_fields(row)
        row_info = {
            "row_number": idx + 1,
            "missing_count": len(missing_fields),
            "missing_fields": ", ".join(missing_fields),
        }
        if "url" in df.columns:
            row_info["url"] = row.get("url")
        if "name" in df.columns:
            row_info["name"] = row.get("name")
        row_report.append(row_info)

    row_report_df = pd.DataFrame(row_report)

    print("=== Пропуски по строкам ===")
    rows_with_missing = row_report_df[row_report_df["missing_count"] > 0]

    if rows_with_missing.empty:
        print("Пропусков по строкам не найдено.")
    else:
        for _, r in rows_with_missing.iterrows():
            parts = [f"Строка {r['row_number']}"]
            if "name" in r and pd.notna(r["name"]):
                parts.append(f"name={r['name']}")
            if "url" in r and pd.notna(r["url"]):
                parts.append(f"url={r['url']}")
            parts.append(f"пропущены: {r['missing_fields']}")
            print(" | ".join(parts))

    print()
    print("=== Сводка по полям ===")

    summary_rows = []
    for col in df.columns:
        series = df[col]
        nan_count = int(series.isna().sum())
        empty_string_count = int(series.apply(lambda x: isinstance(x, str) and x.strip() == "").sum())
        total_missing = nan_count + empty_string_count

        summary_rows.append(
            {
                "field": col,
                "missing_rows": total_missing,
                "missing_percent": round(total_missing / len(df) * 100, 2) if len(df) else 0.0,
                "nan_count": nan_count,
                "empty_string_count": empty_string_count,
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["missing_rows", "field"], ascending=[False, True]
    ).reset_index(drop=True)

    for _, r in summary_df.iterrows():
        print(
            f"{r['field']}: пропущено в {r['missing_rows']} строках "
            f"({r['missing_percent']}%), "
            f"NaN={r['nan_count']}, пустых строк={r['empty_string_count']}"
        )

    if args.save_row_report:
        row_report_df.to_csv(args.save_row_report, index=False, encoding="utf-8-sig")
        print()
        print(f"Построчный отчёт сохранён в: {args.save_row_report}")

    if args.save_summary_report:
        summary_df.to_csv(args.save_summary_report, index=False, encoding="utf-8-sig")
        print(f"Сводка сохранена в: {args.save_summary_report}")


if __name__ == "__main__":
    main()
