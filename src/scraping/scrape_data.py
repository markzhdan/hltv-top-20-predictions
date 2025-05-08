# Predicting HLTV's Top 20 Players of 2024.
# Purpose: To predict HLTV's top 20 player rankings based on previous year stats.
# File: scrape_data.py - used to parse hltv stats pages and store raw data.
# Mark Zhdan | 05/07/25

import time
import csv
import pickle
import requests
from bs4 import BeautifulSoup as BS

# URL Modifiers
urlInfo = {
    "baseURL": "https://www.hltv.org/stats/players/",
    "lan": "&matchType=Lan",
    "bigEvents": "&matchType=BigEvents",
    "majors": "&matchType=Majors",
    "2022": "?startDate=2022-01-01&endDate=2022-12-31",
    "2021": "?startDate=2021-01-01&endDate=2021-12-31",
    "2020": "?startDate=2020-01-01&endDate=2020-12-31",
    "2019": "?startDate=2019-01-01&endDate=2019-12-31",
    "2018": "?startDate=2018-01-01&endDate=2018-12-31",
    "2017": "?startDate=2017-01-01&endDate=2017-12-31",
    "top5": "&rankingFilter=Top5",
    "top10": "&rankingFilter=Top10",
    "top20": "&rankingFilter=Top20",
    "top30": "&rankingFilter=Top30",
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

            print(f"\n📥 Response for URL:\n{url}")
            print(f"🔢 Status Code: {response.status_code}")
            print(f"🧾 Content-Type: {response.headers.get('content-type')}")
            print(f"🧪 Raw Text Preview:\n{response.text[:500]}\n{'-'*50}")

            if response.status_code != 200:
                print("❌ Non-200 response — skipping.")
                return None

            # Since server returns HTML directly, not JSON
            html = response.text

            soup = BS(html, "html.parser")
            stats = soup.find_all(class_="summaryStatBreakdownDataValue")

            completeStats = []
            for stat in stats[:6]:
                try:
                    value = stat.get_text().strip().rstrip("%")
                    completeStats.append(float(value))
                except:
                    completeStats.append(0.0)

            print(f"🔍 Parsed Stats: {completeStats}")

            return completeStats if len(completeStats) == 6 else None

        except Exception as e:
            print(f"❌ Error fetching HTML for {url}: {e}")
            return None


def save_pickle(obj, filename):
    with open(filename, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def main(year):
    match_types = ["baseURL", "lan", "bigEvents", "majors"]
    ranking_filters = ["top30", "top20", "top10", "top5"]
    data = {}

    scraper = HLTVScraper()

    with open("data/player_links.csv", "r") as file:
        reader = csv.reader(file)
        for row in reader:
            if not row or not row[0].strip():
                continue  # Skip empty lines
            try:
                base_url = row[0].strip().strip('"')
                player_name = base_url.split("/")[-1]
                data[player_name] = {}

                for mt in match_types:
                    mt_key = "normal" if mt == "baseURL" else mt
                    data[player_name][mt_key] = {}

                    for rf in ranking_filters:
                        url = base_url + urlInfo[year]
                        if mt != "baseURL":
                            url += urlInfo[mt]
                        url += urlInfo[rf]

                        print(f"🔍 Scraping {player_name} - {mt_key} - {rf}...")
                        stats = scraper.get_stats(url)
                        data[player_name][mt_key][rf] = stats
                        time.sleep(1.5)

            except Exception as e:
                print(f"⚠️ Error for {row}: {e}")
                continue

    save_pickle(data, f"player_data_{year}_expanded.pkl")
    print(f"\n✅ All combinations scraped and saved for {year}.")


if __name__ == "__main__":
    main("2022")
