import re
import csv

# File to read
input_file = "rankings.txt"

# Regular expression to extract rank, country, full name, nickname, and last name
line_regex = re.compile(r'^(\d+)\.\s+([\w\s&-]+)\s+([^\"]+?)\s+"([^"]+)"\s+(.+)$')

# Initialize
rankings = {}
current_year = 2018
current_list = []

# Read file
with open(input_file, "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        # Handle year section break
        if line == "-":
            if current_list:
                rankings[current_year] = current_list
                current_list = []
                current_year += 1
            continue

        match = line_regex.match(line)
        if match:
            rank = int(match.group(1))
            country = match.group(2).strip()
            first_name = match.group(3).strip()
            nickname = match.group(4).strip()
            last_name = match.group(5).strip()
            full_name = f"{first_name} {last_name}"
            current_list.append((rank, country, full_name, nickname))

    # Save last section
    if current_list:
        rankings[current_year] = current_list

# Write to CSV
for year, players in rankings.items():
    filename = f"rankings/ranking_{year}.csv"
    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Rank", "Country", "Full Name", "Nickname"])
        writer.writerows(players)

print("All yearly rankings saved to CSV files.")
