import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_nba_box_scores_for_month(season=2025, month='february'):
    """
    Scrape all box scores for a given month and season from Basketball Reference,
    returning a list of dictionaries containing player stats (player, points, date, team).
    """
    base_url = "https://www.basketball-reference.com"
    schedule_url = f"{base_url}/leagues/NBA_{season}_games-{month}.html"

    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"  
    }

    time.sleep(15)

    response = requests.get(schedule_url, headers=headers)

    if not response.ok:
        print(f"Failed to retrieve {schedule_url}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    # The schedule is typically in a table with id="schedule"
    schedule_table = soup.find("table", {"id": "schedule"})
    if not schedule_table:
        print(f"No schedule table found for {month} {season}")
        return []

    player_game_data = []
    
    # Each row of the table usually has data about one game
    for row in schedule_table.tbody.find_all("tr"):
        # Some rows may be header rows for weeks; skip those
        if "thead" in row.get("class", []):
            continue
        
        date_cell = row.find("th", {"data-stat": "date_game"})
        if not date_cell:
            continue
        
        date_text = date_cell.get_text(strip=True)
        
        # Box score link is in a td with data-stat="box_score_text"
        box_score_cell = row.find("td", {"data-stat": "box_score_text"})
        if not box_score_cell:
            # Some games might not have a box score link yet
            continue
        
        link_tag = box_score_cell.find("a")
        if not link_tag:
            continue
        
        box_score_url = base_url + link_tag["href"]  # Full URL to the box score
        
        # Scrape the box score page
        player_game_data.extend(scrape_box_score(box_score_url, date_text))

        # Be polite and sleep to avoid overwhelming the site
        time.sleep(15)

    return player_game_data

def scrape_box_score(url, date_text):
    """
    Given a box score URL, scrape each player's points and return a list of dicts
    with { 'player': ..., 'points': ..., 'team': ..., 'date': ... }.
    """
    resp = requests.get(url)
    if not resp.ok:
        print(f"Failed to retrieve {url}")
        return []
    
    soup = BeautifulSoup(resp.text, "html.parser")
    
    data = []
    # Box scores often have tables with IDs like: box-<TEAM>-game-basic
    for table in soup.find_all("table"):
        if "box-" in table.get("id", "") and table.get("id", "").endswith("-game-basic"):
            # This is one team's box score table
            team_id = table.get("id").replace("box-", "").replace("-game-basic", "").upper()
            
            tbody = table.find("tbody")
            if not tbody:
                continue
            
            for player_row in tbody.find_all("tr"):
                # Skip rows that don't contain actual player data
                if not player_row.find("th", {"data-stat": "player"}):
                    continue
                
                player_cell = player_row.find("th", {"data-stat": "player"})
                player_name = player_cell.get_text(strip=True) if player_cell else None

                pts_cell = player_row.find("td", {"data-stat": "pts"})
                if not pts_cell:
                    # Could be a "Did Not Play" or "Totals" row
                    continue
                
                try:
                    points = int(pts_cell.get_text(strip=True))
                except ValueError:
                    points = 0  # or None, depending on how you want to handle missing data
                
                data.append({
                    "date": date_text,
                    "team": team_id,
                    "player": player_name,
                    "points": points
                })
    
    return data

if __name__ == "__main__":
    # Only scrape October through February
    months = ["october", "november", "december", "january", "february"]
    season = 2025

    all_data = []
    for month in months:
        print(f"Scraping {month.capitalize()} {season}...")
        monthly_data = scrape_nba_box_scores_for_month(season, month)
        all_data.extend(monthly_data)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=["date", "team", "player", "points"])
    
    # Preview
    print(df.head())
    print(f"Total rows scraped: {len(df)}")

    import os

    output_filename = "nba_box_scores_oct_to_feb_2025.csv"
    df.to_csv(output_filename, index=False)



