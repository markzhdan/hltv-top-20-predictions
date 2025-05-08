import csv
import glob
import re

# Step 1: Parse player_links.txt (quoted CSV) into nickname → HLTV ID mapping
nickname_to_id = {}

with open("data/player_links.csv", "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        if not row:
            continue
        url = row[0].strip().strip('"')
        match = re.search(r"/players/(\d+)/([\w\-]+)", url)
        if match:
            hltv_id, nickname = match.groups()
            nickname_to_id[nickname.lower()] = hltv_id

# Prepare unmatched log
unmatched_log = []

# Step 2: Process all ranking_*.csv files
ranking_files = sorted(glob.glob("rankings/ranking_*.csv"))

for filename in ranking_files:
    output_file = filename.replace(".csv", "_with_ids.csv")

    with open(filename, "r", encoding="utf-8") as infile, open(
        output_file, "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["HLTV_ID"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            nickname_raw = row["Nickname"].strip()
            nickname_key = nickname_raw.lower()
            hltv_id = nickname_to_id.get(nickname_key, "")

            if not hltv_id:
                unmatched_log.append(f"{filename}: '{nickname_raw}' not matched")

            row["HLTV_ID"] = hltv_id
            writer.writerow(row)

# Step 3: Save unmatched nicknames to log
if unmatched_log:
    with open("unmatched_players.log", "w", encoding="utf-8") as log_file:
        log_file.write("\n".join(unmatched_log))

print("✅ HLTV ID matching complete. New CSVs created and unmatched nicknames logged.")
