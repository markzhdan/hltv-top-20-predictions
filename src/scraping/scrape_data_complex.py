# Mark Zhdan | Final version: 05/08/25

import os
import time
import csv
import pickle
import requests
import unicodedata
from bs4 import BeautifulSoup as BS

urlInfo = {str(y): f"?startDate={y}-01-01&endDate={y}-12-31" for y in range(2018, 2025)}

match_filters = {
    "": "",  # Regular
    "b_": "&matchType=BigEvents",
    "m_": "&matchType=Majors",
}


def try_float(text):
    text = text.strip().replace("%", "")
    if text == "-" or text == "":
        return None
    try:
        return float(text)
    except:
        return None


def clean_name(name):
    return "".join(
        c for c in name if not unicodedata.category(c).startswith("C")
    ).strip()


class HLTVScraper:
    def __init__(self, bypass_url="http://localhost:8000/html"):
        self.bypass_url = bypass_url

    def get_stats(self, url):
        try:
            response = requests.get(self.bypass_url, params={"url": url}, timeout=20)
            if response.status_code != 200:
                print(f"❌ Failed to fetch {url} — HTTP {response.status_code}")
                return None

            soup = BS(response.text, "html.parser")
            stats = {}

            # Summary stats block (Rating, DPR, KAST, etc.)
            summary_values = soup.find_all(
                "div", class_="summaryStatBreakdownDataValue"
            )
            summary_keys = ["rating", "dpr", "kast", "impact", "adr", "kpr"]
            summary_valid = True
            summary_parsed = []

            if len(summary_values) >= 6:
                for stat in summary_values[:6]:
                    val = try_float(stat.text)
                    summary_parsed.append(val)
                    if val is None:
                        summary_valid = False

                if summary_valid:
                    for k, v in zip(summary_keys, summary_parsed):
                        stats[k] = v
                else:
                    for k in summary_keys:
                        stats[k] = -1

            # Featured ratings (vs top 5–50)
            featured = soup.find("div", class_="featured-ratings-container")
            if featured:
                for box in featured.find_all("div", class_="rating-breakdown"):
                    desc = box.find("div", class_="rating-description").text.lower()
                    val = try_float(box.find("div", class_="rating-value").text)
                    if "top 5" in desc:
                        stats["vs_top5"] = val if val is not None else -1
                    elif "top 10" in desc:
                        stats["vs_top10"] = val if val is not None else -1
                    elif "top 20" in desc:
                        stats["vs_top20"] = val if val is not None else -1
                    elif "top 30" in desc:
                        stats["vs_top30"] = val if val is not None else -1
                    elif "top 50" in desc:
                        stats["vs_top50"] = val if val is not None else -1

            # Full statistics block
            for row in soup.select(".stats-row"):
                spans = row.find_all("span")
                if len(spans) == 2:
                    key = (
                        spans[0]
                        .text.strip()
                        .lower()
                        .replace(" ", "_")
                        .replace("/", "_per_")
                    )
                    val = try_float(spans[1].text)
                    stats[key] = val if val is not None else -1

            return stats

        except Exception as e:
            print(f"❌ Error fetching {url}: {e}")
            return None


def save_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def save_csv(data, path):
    keys = sorted({k for stats in data.values() if stats for k in stats})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Player"] + keys)
        for player, stats in data.items():
            row = [player] + [stats.get(k, "") for k in keys]
            writer.writerow(row)


def scrape_all_years(start_year=2024, end_year=2024):
    scraper = HLTVScraper()

    for year in range(start_year, end_year + 1):
        year_str = str(year)
        print(f"\n📅 Scraping for {year_str}...\n")

        input_path = f"rankings/ranking_{year_str}.csv"
        base = f"player_data_{year_str}_full"
        pickle_path = f"data/raw/complex/{base}.pkl"
        csv_path = f"data/raw/complex/{base}.csv"
        data = {}

        if not os.path.exists(input_path):
            print(f"⚠️ Missing ranking file: {input_path}")
            continue

        with open(input_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_name = row["Nickname"]
                player = clean_name(raw_name)
                link = row["Link"].strip()
                data[player] = {}

                for prefix, mod in match_filters.items():
                    full_url = link + urlInfo[year_str] + mod
                    print(f"🔍 {year_str} | {player} | {prefix or 'base'}")
                    stats = scraper.get_stats(full_url)
                    if stats:
                        for k, v in stats.items():
                            data[player][prefix + k] = v
                    time.sleep(1.5)

                # Save after each player
                save_csv(data, csv_path)

        print(f"✅ Done scraping {year_str} — saved to {csv_path}")


if __name__ == "__main__":
    scrape_all_years()
