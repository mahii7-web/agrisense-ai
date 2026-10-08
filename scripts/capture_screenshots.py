"""
Automated Screenshot Capture Script for AgriSense AI
Uses Selenium with Chrome/Edge to capture clean high-resolution screenshots
of all sections of the running Streamlit application.
"""

import os
import time
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

BASE_DIR = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = BASE_DIR / "assets" / "screenshots"
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

def capture_all():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1280,950")

    driver = webdriver.Chrome(options=chrome_options)
    url = "http://localhost:8501"

    try:
        print("[1/7] Navigating to Home...")
        driver.get(url)
        time.sleep(4)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "01_home.png"))
        print("      -> Saved 01_home.png")

        # Navigate to Crop Analysis
        print("[2/7] Navigating to Crop Analysis...")
        radio_labels = driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="stRadio"] label')
        for r in radio_labels:
            if "Crop Analysis" in r.text:
                r.click()
                break
        time.sleep(3)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "02_crop_analysis.png"))
        print("      -> Saved 02_crop_analysis.png")

        # Click Viva Preset & Analyze button
        print("[3/7] Clicking Preset & Analyzing Conditions...")
        buttons = driver.find_elements(By.TAG_NAME, "button")
        for b in buttons:
            if "Monsoon Wetland" in b.text:
                b.click()
                break
        time.sleep(2)

        # Click submit button
        submit_btns = driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="stFormSubmitButton"] button')
        if submit_btns:
            submit_btns[0].click()
            time.sleep(4)
            driver.save_screenshot(str(SCREENSHOTS_DIR / "03_recommendation.png"))
            print("      -> Saved 03_recommendation.png")

        # Navigate to Cluster Analysis
        print("[4/7] Navigating to Cluster Analysis...")
        radio_labels = driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="stRadio"] label')
        for r in radio_labels:
            if "Cluster Analysis" in r.text:
                r.click()
                break
        time.sleep(3)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "04_cluster_analysis.png"))
        print("      -> Saved 04_cluster_analysis.png")

        # Navigate to Data Exploration
        print("[5/7] Navigating to Data Exploration...")
        radio_labels = driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="stRadio"] label')
        for r in radio_labels:
            if "Data Exploration" in r.text:
                r.click()
                break
        time.sleep(3)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "05_data_exploration.png"))
        print("      -> Saved 05_data_exploration.png")

        # Navigate to ML Model
        print("[6/7] Navigating to ML Model Evaluation...")
        radio_labels = driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="stRadio"] label')
        for r in radio_labels:
            if "ML Model" in r.text:
                r.click()
                break
        time.sleep(3)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "06_ml_evaluation.png"))
        print("      -> Saved 06_ml_evaluation.png")

        # Capture PCA plot from Recommendation / Analysis
        print("[7/7] Capturing 07_pca.png...")
        # Crop analysis recommendation page contains 2D PCA plot
        radio_labels = driver.find_elements(By.CSS_SELECTOR, 'div[data-testid="stRadio"] label')
        for r in radio_labels:
            if "Crop Analysis" in r.text:
                r.click()
                break
        time.sleep(3)
        driver.execute_script("window.scrollTo(0, 750);")
        time.sleep(1)
        driver.save_screenshot(str(SCREENSHOTS_DIR / "07_pca.png"))
        print("      -> Saved 07_pca.png")

        print("All screenshots successfully captured!")

    finally:
        driver.quit()

if __name__ == "__main__":
    capture_all()
