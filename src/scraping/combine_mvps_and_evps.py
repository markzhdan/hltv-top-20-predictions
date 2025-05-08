import csv
import os

# File paths
MVP_CSV = "data/clean/mvp_counts_by_year.csv"
EVP_CSV = "data/clean/evp_counts_by_year.csv"
INPUT_DIR = "data/clean"
OUTPUT_DIR = "data/final"
YEARS = list(range(2018, 2025))


# Load data from MVP/EVP files into {player: {year: count}} format
def load_award_data(filepath):
    table = {}
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            player = row["Player"].strip()
            table[player] = {str(year): int(row.get(str(year), 0)) for year in YEARS}
    return table


# Get count from a table using case-insensitive matching
def get_award_count(table, player_name, year):
    if player_name in table:
        return table[player_name].get(str(year), 0), True
    for key in table:
        if key.lower() == player_name.lower():
            return table[key].get(str(year), 0), True
    return 0, False


def process_year(year, mvp_table, evp_table):
    input_path = os.path.join(INPUT_DIR, f"player_data_{year}.csv")
    output_path = os.path.join(OUTPUT_DIR, f"player_data_{year}_final.csv")

    if not os.path.exists(input_path):
        print(f"❌ Missing file: {input_path}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    matched, unmatched = set(), set()

    with open(input_path, newline="", encoding="utf-8") as infile, open(
        output_path, "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["MVPs", "EVPs"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            player = row["Player"].strip()

            mvp_count, mvp_found = get_award_count(mvp_table, player, year)
            evp_count, evp_found = get_award_count(evp_table, player, year)

            row["MVPs"] = mvp_count
            row["EVPs"] = evp_count
            writer.writerow(row)

            if mvp_found or evp_found:
                matched.add(player)
            else:
                unmatched.add(player)

    print(f"\n📅 {year}")
    print(f"✅ Matched ({len(matched)}): {sorted(matched)}")
    print(f"⚠️ Unmatched ({len(unmatched)}): {sorted(unmatched)}")


def main():
    mvp_table = load_award_data(MVP_CSV)
    evp_table = load_award_data(EVP_CSV)

    for year in YEARS:
        process_year(year, mvp_table, evp_table)


if __name__ == "__main__":
    main()
