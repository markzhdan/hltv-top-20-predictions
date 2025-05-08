import re
import csv
from collections import defaultdict

# File paths
input_path = "data/evps_html.txt"
output_csv = "data/clean/evp_counts_by_year.csv"

# Load text
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# Initialize structure: evp_counts[year][player] = count
evp_counts = defaultdict(lambda: defaultdict(int))

# Split into yearly blocks (e.g., 2024\nEvent\nDate...)
year_sections = re.split(r"\n(?=\d{4}\nEvent)", text)
for section in year_sections:
    year_match = re.match(r"(\d{4})\nEvent", section)
    if not year_match:
        continue
    year = year_match.group(1)

    # Match quoted player names (e.g., "ZywOo") after teams
    matches = re.findall(r"(?:\"|“)(.+?)(?:\"|”)", section)
    for player in matches:
        player = player.strip()
        evp_counts[year][player] += 1

# Collect all years and all unique players
all_years = sorted(evp_counts.keys())
all_players = sorted(set(player for year in evp_counts for player in evp_counts[year]))

# Write to CSV
with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Player"] + all_years + ["Total"])

    for player in all_players:
        row = [player]
        total = 0
        for year in all_years:
            count = evp_counts[year].get(player, 0)
            row.append(count)
            total += count
        row.append(total)
        writer.writerow(row)

print(f"✅ EVP counts by year saved to: {output_csv}")
