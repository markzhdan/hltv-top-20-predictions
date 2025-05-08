import csv
import os


def merge_csvs(stats_path, ranking_path, output_path):
    # Load rankings.csv into a dict {nickname: {id, rank}}
    ranking_info = {}
    with open(ranking_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nickname = row["Nickname"].strip()
            ranking_info[nickname] = {
                "HLTV_ID": row["HLTV_ID"].strip(),
                "Rank": row["Rank"].strip(),
            }

    # Merge into output file
    with open(stats_path, "r") as f_in, open(output_path, "w", newline="") as f_out:
        reader = csv.DictReader(f_in)
        fieldnames = [
            "Player",
            "HLTV_ID",
            "Rating",
            "DPR",
            "KAST",
            "Impact",
            "ADR",
            "KPR",
            "Rank",
        ]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            player = row["Player"].strip()
            if player in ranking_info:
                merged_row = {
                    "Player": player,
                    "HLTV_ID": ranking_info[player]["HLTV_ID"],
                    "Rating": row["Rating"],
                    "DPR": row["DPR"],
                    "KAST": row["KAST"],
                    "Impact": row["Impact"],
                    "ADR": row["ADR"],
                    "KPR": row["KPR"],
                    "Rank": ranking_info[player]["Rank"],
                }
                writer.writerow(merged_row)
            else:
                print(f"⚠️ Skipping {player} (not found in ranking CSV)")


def main():
    os.makedirs("data/clean", exist_ok=True)

    for year in range(2018, 2025):
        stats_csv = f"data/raw/player_data_{year}_base.csv"
        ranking_csv = f"rankings/ranking_{year}.csv"
        output_csv = f"data/clean/player_data_{year}.csv"

        if not os.path.exists(stats_csv):
            print(f"❌ Missing stats file for {year}: {stats_csv}")
            continue
        if not os.path.exists(ranking_csv):
            print(f"❌ Missing ranking file for {year}: {ranking_csv}")
            continue

        print(f"\n📅 Merging stats with rankings for {year}...")
        merge_csvs(stats_csv, ranking_csv, output_csv)
        print(f"✅ Done: {output_csv}")


if __name__ == "__main__":
    main()
