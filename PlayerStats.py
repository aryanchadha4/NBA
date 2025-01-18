from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import Select
import pandas as pd
import time

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

        WebDriverWait(driver, 20).until(
            EC.presence_of_all_elements_located((By.XPATH, ".//tbody/tr"))
        )
        print("Table updated")
    except Exception as e:
        print(f"Failed: {e}")

def wait_for_populated_rows(driver, timeout=100):
    try:
        WebDriverWait(driver, timeout).until(
            lambda d: all(
                len(row.find_elements(By.XPATH, ".//td")) > 0
                and all(cell.text.strip() for cell in row.find_elements(By.XPATH, ".//td"))
                for row in d.find_elements(By.XPATH, ".//tbody/tr")
            )
        )
        print("rows fully populated.")
    except TimeoutException:
        print("rows are not fully populated.")

def stat_scraper(url, driver, season):
   driver.get(url)

   try:
       cookie = WebDriverWait(driver, 5).until(
           EC.presence_of_element_located((By.CLASS_NAME, "ot-sdk-row"))
       )
       driver.execute_script("arguments[0].click();", cookie)
       print("Cookie banner dismissed.")
   except Exception:
       print("No cookie banner")

   select_all_from_dropdown(driver)

   table_xpath = "//table[contains(@class, 'Crom_table')]"
   table = WebDriverWait(driver, 30).until(
       EC.presence_of_element_located((By.XPATH, table_xpath))
   )
   headers = [header.text.strip() for header in table.find_elements(By.XPATH, ".//thead/tr/th")]
   headers = [header for header in headers if header.strip()]
   print(f"Cleaned Headers: {headers}")

   all_data = []
   page_number = 1

   wait_for_populated_rows(driver)
   rows = driver.find_elements(By.XPATH, ".//tbody/tr")
   for row in rows:
       cells = row.find_elements(By.XPATH, ".//td")
       all_data.append([cell.text for cell in cells])


   print(f"Total rows: {len(all_data)}")

   cleaned_data = [row[:len(headers)] for row in all_data if len(row) >= len(headers)]


   df = pd.DataFrame(cleaned_data, columns=headers)

   df.to_csv(f'nba_all_player_stats_{season}.csv', index=False)
   print(f"saved to nba_all_player_stats_{season}.csv")


service = Service("/usr/local/bin/chromedriver")
driver = webdriver.Chrome(service=service)
season = 2025
urls = ["https://www.nba.com/stats/players/traditional?sort=PTS&dir=-1&Season=2024-25",
"https://www.nba.com/stats/players/traditional?sort=PTS&dir=-1&Season=2023-24", "https://www.nba.com/stats/players/traditional?sort=PTS&dir=-1&Season=2022-23"]
for url in urls:
   stat_scraper(url, driver, season)
   season -= 1

driver.quit()

