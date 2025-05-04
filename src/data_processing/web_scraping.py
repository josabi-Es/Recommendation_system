import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
import os

def scrape_hacker_news(url="https://hn.algolia.com", num_pages=2, output_path="../data/DataWebScraping.csv"):
    """
    Scrapes data from Hacker News and saves it to a CSV file.

    Args:
        url (str, optional): The URL to start scraping from.
                           Defaults to "https://hn.algolia.com".
        num_pages (int, optional): The number of pages to scrape.
                                 Defaults to 2.
        output_path (str, optional): The path to save the output CSV file.
                                     Defaults to "../data/DataWebScraping.csv".

    Returns:
        pandas.DataFrame: A DataFrame containing the scraped data
                        with columns "title", "points", and "num_pages".
    """

    driver = webdriver.Chrome()
    driver.get(url)
    print(driver.title)
    time.sleep(5)

    data = {"title": [], "points": [], "num_pages": []}

    for i in range(num_pages):
        elements = driver.find_elements(By.CLASS_NAME, "Story_data")
        for elem in elements:
            title = elem.find_element(By.CLASS_NAME, "Story_title")
            meta = elem.find_element(By.CLASS_NAME, "Story_meta")
            points = meta.find_element(By.TAG_NAME, "span")
            points = int(points.text.split(" ")[0])

            data["title"].append(title.text)
            data["points"].append(points)
            data["num_pages"].append(i + 1)

        next_page = driver.find_element(
            By.XPATH,
            "//li[@class='Pagination_item Pagination_item-current']/following-sibling::li",
        )
        next_page.click()
        time.sleep(5)

    df = pd.DataFrame(data)
    driver.close()

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_csv(output_path, encoding="utf-8", index=False)
    print(f"Scraped data saved to: {output_path}")
    return df


if __name__ == "__main__":
    # This block is executed only if the script is run directly (not imported)
    scraped_data = scrape_hacker_news()
    print(scraped_data) # Print the dataframe to console