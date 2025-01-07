from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time

service = Service("/usr/local/bin/chromedriver")
driver = webdriver.Chrome(service=service)

url = "https://www.nba.com/stats/players/traditional/?sort=PTS&dir=-1"
driver.get(url)

try:
    cookie = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.CLASS_NAME, "ot-sdk-row"))
    )
    driver.execute_script("arguments[0].click();", cookie)
    print("Cookie banner dismissed.")
except Exception:
    print("No cookie banner or overlay found.")

table_xpath = "//table[contains(@class, 'Crom_table')]"

table = WebDriverWait(driver, 30).until(
    EC.presence_of_element_located((By.XPATH, table_xpath))
)
headers = [header.text.strip() for header in table.find_elements(By.XPATH, ".//thead/tr/th")]
headers = [header for header in headers if header.strip()]
print(f"Cleaned Headers: {headers}")

all_data = []
i = 0
while True:
    rows = driver.find_elements(By.XPATH, ".//tbody/tr")
    for row in rows:
        cells = row.find_elements(By.XPATH, ".//td")
        all_data.append([cell.text for cell in cells])

    try:
        i+=1
        print(f"iterations: {i}")
        next_button = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.XPATH, "//button[@data-pos='next']"))
        )
        if "disabled" in next_button.get_attribute("class"):
            print("Reached last page.")
            break
        driver.execute_script("arguments[0].click();", next_button)
        time.sleep(3)
    except Exception:
        print("No more pages")
        break

print(f"rows: {len(all_data)}")

cleaned_data = [row[:len(headers)] for row in all_data if len(row) >= len(headers)]

df = pd.DataFrame(cleaned_data, columns=headers)

df.to_csv('nba_all_player_stats.csv', index=False)

print("All player stats have been saved")

driver.quit()










