import re
import csv
from collections import defaultdict

# File paths
input_path = "data/mvps_html.txt"
output_csv = "data/clean/mvp_counts_by_year.csv"

# Load file
with open(input_path, "r", encoding="utf-8") as f:
    text = f.read()

# Initialize structure: mvp_counts[year][player] = count
mvp_counts = defaultdict(lambda: defaultdict(int))

# Split data into year chunks
year_sections = re.split(r"\n(?=\d{4}\nEvent)", text)
for section in year_sections:
    year_match = re.match(r"(\d{4})\nEvent", section)
    if not year_match:
        continue
    year = year_match.group(1)

    # Match players from that section
    matches = re.findall(r"\d{4}-\d{2}-\d{2}\s+.*?\s+(?:\"|“)(.+?)(?:\"|”)", section)
    for player in matches:
        player = player.strip()
        mvp_counts[year][player] += 1

# Collect all years and all unique players
all_years = sorted(mvp_counts.keys())
all_players = sorted(set(player for year in mvp_counts for player in mvp_counts[year]))

# Write to CSV
with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Player"] + all_years + ["Total"])

    for player in all_players:
        row = [player]
        total = 0
        for year in all_years:
            count = mvp_counts[year].get(player, 0)
            row.append(count)
            total += count
        row.append(total)
        writer.writerow(row)

print(f"✅ MVP counts by year saved to: {output_csv}")
