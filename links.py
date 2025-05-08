import csv
import glob
import os
import re
import unicodedata

# ========== CONFIG ==========
LINKS_FILE = "data/player_links.csv"
OUTPUT_FOLDER = "rankings_ids"
LOG_FILE = "unmatched_players.log"
# =============================


def clean_nickname(nick):
    """Normalize nickname for comparison: lowercase, unicode-safe, strip spaces."""
    if not nick:
        return ""
    return (
        unicodedata.normalize("NFKD", nick)
        .encode("ascii", "ignore")
        .decode()
        .lower()
        .strip()
    )


# Step 1: Load player_links.csv into nickname → HLTV ID mapping
nickname_to_id = {}

with open(LINKS_FILE, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    for row in reader:
        if not row:
            continue
        url = row[0].strip().strip('"')
        match = re.search(r"/players/(\d+)/([\w\-]+)", url)
        if match:
            hltv_id, nickname = match.groups()
            nickname_to_id[clean_nickname(nickname)] = hltv_id

# Step 2: Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Step 3: Process ranking_*.csv files
ranking_files = sorted(glob.glob("rankings/ranking_*.csv"))
unmatched_log = []

for filename in ranking_files:
    base_name = os.path.basename(filename)
    output_path = os.path.join(OUTPUT_FOLDER, base_name)

    with open(filename, "r", encoding="utf-8") as infile, open(
        output_path, "w", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ["HLTV_ID"]
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            nickname_raw = row["Nickname"]
            nickname_key = clean_nickname(nickname_raw)
            hltv_id = nickname_to_id.get(nickname_key, "")

            if not hltv_id:
                unmatched_log.append(
                    f"{base_name}: '{nickname_raw}' not matched → cleaned: '{nickname_key}'"
                )

            row["HLTV_ID"] = hltv_id
            writer.writerow(row)

# Step 4: Save unmatched log
if unmatched_log:
    with open(LOG_FILE, "w", encoding="utf-8") as log_file:
        log_file.write("\n".join(unmatched_log))

print(f"✅ All files processed. Results saved in '{OUTPUT_FOLDER}'.")
print(f"🔎 Unmatched nicknames logged in '{LOG_FILE}'.")
