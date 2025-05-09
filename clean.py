import os
import csv

YEARS = list(range(2018, 2025))
INPUT_DIR = "data/clean/complex"
TOP20_DIR = "data/final"
OUTPUT_DIR = "data/clean/complex/final"


def load_top20_data(filepath):
    top20 = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            player = row["Player"].strip()
            top20[player] = {
                "MVPs": row.get("MVPs", ""),
                "EVPs": row.get("EVPs", ""),
                "Rank": row.get("Rank", ""),
            }
    return top20


def merge_data(year):
    input_path = os.path.join(INPUT_DIR, f"player_data_{year}_full.csv")
    top20_path = os.path.join(TOP20_DIR, f"player_data_{year}_final.csv")
    output_path = os.path.join(OUTPUT_DIR, f"player_data_{year}_full_final.csv")

    if not os.path.exists(input_path):
        print(f"❌ Missing stats file for {year}")
        return

    if not os.path.exists(top20_path):
        print(f"⚠️ No top 20 file for {year}, skipping merge")
        return

    top20 = load_top20_data(top20_path)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(input_path, newline="", encoding="utf-8") as f_in, open(
        output_path, "w", newline="", encoding="utf-8"
    ) as f_out:

        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames + ["MVPs", "EVPs", "Rank"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            player = row["Player"].strip()
            if player in top20:
                row.update(top20[player])
            else:
                row.update({"MVPs": "", "EVPs": "", "Rank": ""})
            writer.writerow(row)

    print(f"✅ Merged {year} → {output_path}")


def main():
    for year in YEARS:
        merge_data(year)


if __name__ == "__main__":
    main()
