# Predicting HLTV's Top 20 Players of 2024.
# Purpose: To predict HLTV's top 20 player rankings based on previous year stats.
# File: scrape_data.py - used to parse hltv stats pages and store raw data.
# Mark Zhdan | 05/07/25
import os
import time
import csv
import pickle
import requests
from bs4 import BeautifulSoup as BS

# URL Modifiers
urlInfo = {
    "2018": "?startDate=2018-01-01&endDate=2018-12-31",
    "2019": "?startDate=2019-01-01&endDate=2019-12-31",
    "2020": "?startDate=2020-01-01&endDate=2020-12-31",
    "2021": "?startDate=2021-01-01&endDate=2021-12-31",
    "2022": "?startDate=2022-01-01&endDate=2022-12-31",
    "2023": "?startDate=2023-01-01&endDate=2023-12-31",
    "2024": "?startDate=2024-01-01&endDate=2024-12-31",
}


class Stats:
    def __init__(self, ign, rating, dpr, kast, impact, adr, kpr):
        self.ign = ign
        self.rating = rating
        self.dpr = dpr
        self.kast = kast
        self.impact = impact
        self.adr = adr
        self.kpr = kpr

    def to_list(self):
        return [self.rating, self.dpr, self.kast, self.impact, self.adr, self.kpr]


class HLTVScraper:
    def __init__(self, bypass_url="http://localhost:8000/html"):
        self.bypass_url = bypass_url

    def get_stats(self, url):
        try:
            response = requests.get(self.bypass_url, params={"url": url}, timeout=20)

            if response.status_code != 200:
                print(f"❌ Failed to fetch {url} — HTTP {response.status_code}")
                return None

            html = response.text
            soup = BS(html, "html.parser")
            stats = soup.find_all(class_="summaryStatBreakdownDataValue")

            completeStats = []
            for stat in stats[:6]:  # Rating, DPR, KAST, IMPACT, ADR, KPR
                try:
                    value = stat.get_text().strip().rstrip("%")
                    completeStats.append(float(value))
                except:
                    completeStats.append(0.0)

            print(f"✅ Parsed Stats: {completeStats}")
            return completeStats if len(completeStats) == 6 else None

        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")
            return None


def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def save_csv(data: dict, filename: str):
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Player", "Rating", "DPR", "KAST", "Impact", "ADR", "KPR"])

        for player, stats in data.items():
            if stats is not None:
                writer.writerow([player] + stats)


def scrape_all_years(start_year=2018, end_year=2024):
    scraper = HLTVScraper()

    for year in range(start_year, end_year + 1):
        year_str = str(year)
        print(f"\n📅 Scraping for year {year_str}...\n")

        input_path = f"rankings/ranking_{year_str}.csv"
        pickle_path = f"data/raw/player_data_{year_str}_base.pkl"
        csv_path = f"data/raw/player_data_{year_str}_base.csv"
        data = {}

        if not os.path.exists(input_path):
            print(f"⚠️ File not found: {input_path}")
            continue

        with open(input_path, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                link = row["Link"].strip()
                player_name = row["Nickname"].strip()
                full_url = link + urlInfo[year_str]

                print(f"🔍 Scraping {player_name} ({link})")
                stats = scraper.get_stats(full_url)
                data[player_name] = stats
                time.sleep(1.5)

        save_pickle(data, pickle_path)
        save_csv(data, csv_path)
        print(f"✅ Saved base stats for {year_str} to {pickle_path} and {csv_path}")


if __name__ == "__main__":
    scrape_all_years()
