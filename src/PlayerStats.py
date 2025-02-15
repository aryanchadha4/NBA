import os
import yaml

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import Select
from tqdm import tqdm
import pandas as pd

def select_all_from_dropdown(driver):
    try:
        dropdown = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.XPATH, "//select[contains(@class, 'DropDown_select')]"))
        )
        
        dropdown.click()
        all_option = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, "//option[text()='All']"))
        )
        all_option.click()
        print("Selected all")

        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.XPATH, ".//tbody/tr"))
        )
        print("Table updated")
    except Exception as e:
        print(f"Failed: {e}")


def check_all_cells_have_text(table):
    rows = table.find_elements(By.XPATH, ".//tbody/tr") 
    # i made sure to find "rows" from table specifically
    # by looking at other attributes of the "row" and "cell"
    # data that was being collected.
    for row in rows:
        cells = row.find_elements(By.XPATH, ".//td")
        if not cells or not all(cell.text.strip() for cell in cells):
            return False
    return True


def wait_for_populated_rows(driver, table, timeout=1000):
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: check_all_cells_have_text(table) 
            # I switched this out for a function
            # because having a nested lambda was hard to read
        )
        print("rows fully populated.")
    except TimeoutException:
        print("rows are not fully populated.")

def stat_scraper(url, driver, season):
   driver.get(url)

   select_all_from_dropdown(driver)

   table_xpath = "//table[contains(@class, 'Crom_table')]"
   table = WebDriverWait(driver, 30).until(
       EC.presence_of_element_located((By.XPATH, table_xpath))
   )
   headers = [header.text.strip() for header in table.find_elements(By.XPATH, ".//thead/tr/th")]
   headers = [header for header in headers if header.strip()]
   print(f"Cleaned Headers: {headers}")
   # you got your headers above from table, but then you just searched the whole document
   # for trbody elements in the wait_for_populated_rows function. it got some other random
   # element. my way to debug it was that i saw that it was getting 30 items but not printing
   # anything. i printed out other information about the acquired rows and found that the class
   # attribute belonged to something random so then i added the constraint that it had to be in the table.
   # before, it was waiting forever since the data was never being populated and then
   # timing out.

   all_data = []
   page_number = 1

   wait_for_populated_rows(driver, table)
   rows = driver.find_elements(By.XPATH, ".//tbody/tr")
   for row in rows:
        cells = row.find_elements(By.XPATH, ".//td")
        all_data.append([cell.text for cell in cells])


   print(f"Total rows: {len(all_data)}")

   cleaned_data = [row[:len(headers)] for row in all_data if len(row) >= len(headers)]
   df = pd.DataFrame(cleaned_data, columns=headers)
   df.to_csv(f'nba_all_player_stats_{season}.csv', index=False)
   print(f"saved to nba_all_player_stats_{season}.csv")


def find_chromedriver_path():
    # Check environment variable first
    chromedriver_env = os.environ.get("CHROMEDRIVER_PATH")
    if chromedriver_env:
        return chromedriver_env
    
    # Load config.yml properly
    try:
        with open("config.yml", "r") as file:
            config_yml = yaml.safe_load(file)  # Read YAML file correctly
            if config_yml and "chrome_path" in config_yml:
                return config_yml["chrome_path"]
            else:
                print("Error: 'chrome_path' not found in config.yml")
    except FileNotFoundError:
        print("Error: config.yml not found in the current directory")
    except yaml.YAMLError as e:
        print(f"Error parsing config.yml: {e}")

    # If neither method works, return None
    print("Chromedriver path not found. Set CHROMEDRIVER_PATH or provide a valid config.yml.")
    return None


if __name__=='__main__':
    chromedriver_path = find_chromedriver_path()
    service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=service)
    season = 2025
    urls = ["https://www.nba.com/stats/players/traditional?sort=PTS&dir=-1&Season=2024-25",
    "https://www.nba.com/stats/players/traditional?sort=PTS&dir=-1&Season=2023-24", "https://www.nba.com/stats/players/traditional?sort=PTS&dir=-1&Season=2022-23"]
    for url in urls:
        stat_scraper(url, driver, season)
        season -= 1

    driver.quit()