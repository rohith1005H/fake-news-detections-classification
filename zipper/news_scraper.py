import os
import requests
from bs4 import BeautifulSoup, FeatureNotFound
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
URL_DIRECTORY = '/home/rohtih/mini/urls'  # Directory containing your .txt files
OUTPUT_CSV = 'news_dataset.csv'  # Name for the output dataset file
REQUEST_DELAY = 0  # No delay for faster scraping, but be cautious with server overload
MAX_WORKERS = 10  # Number of threads for concurrent requests

# --- Helper Function to Scrape a Single URL ---
def scrape_news_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        try:
            soup = BeautifulSoup(response.content, 'lxml')
        except FeatureNotFound:
            soup = BeautifulSoup(response.content, 'html.parser')

        headline = None
        possible_headlines = soup.find_all(['h1', 'h2'])
        if possible_headlines:
            h1_headlines = [h.get_text(strip=True) for h in possible_headlines if h.name == 'h1']
            if h1_headlines:
                headline = h1_headlines[0]
            else:
                headline = possible_headlines[0].get_text(strip=True)
        if not headline:
            meta_title = soup.find("meta", property="og:title") or soup.find("meta", {"name": "title"})
            if meta_title and meta_title.get("content"):
                headline = meta_title.get("content", "").strip()
        if not headline:
            title_tag = soup.find('title')
            if title_tag:
                headline = title_tag.get_text(strip=True)

        body_text = ""
        article_body = soup.find('article') or \
                       soup.find('div', class_=lambda x: x and 'content' in x.lower()) or \
                       soup.find('div', class_=lambda x: x and 'post-content' in x.lower()) or \
                       soup.find('div', class_=lambda x: x and 'entry-content' in x.lower()) or \
                       soup.find('div', id=lambda x: x and 'content' in x.lower()) or \
                       soup.find('main')

        if article_body:
            paragraphs = article_body.find_all('p')
            body_text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        else:
            all_paragraphs = soup.find('body').find_all('p') if soup.find('body') else []
            body_text = "\n".join([p.get_text(strip=True) for p in all_paragraphs if p.get_text(strip=True)])

        if headline and body_text:
            return {'headline': headline, 'body': body_text}
        else:
            return None

    except Exception:
        return None

# --- Main Scraping Logic with ThreadPoolExecutor ---
all_data = []

print(f"Scanning directory: {URL_DIRECTORY}")

for filename in os.listdir(URL_DIRECTORY):
    if filename.endswith(".txt"):
        filepath = os.path.join(URL_DIRECTORY, filename)
        print(f"\nProcessing file: {filename}")

        parts = filename.replace('.txt', '').split('_')
        if len(parts) >= 3:
            news_type = parts[0]
            label = parts[1]
            print(f"  -> Type: {news_type}, Label: {label}")
        else:
            print(f"  -> Skipping file (unexpected format): {filename}")
            continue

        try:
            with open(filepath, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"  -> Error reading file {filename}: {e}")
            continue

        print(f"  -> Found {len(urls)} URLs.")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {executor.submit(scrape_news_content, url): url for url in urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    content = future.result()
                    if content:
                        all_data.append({
                            'url': url,
                            'headline': content['headline'],
                            'body': content['body'],
                            'news_type': news_type,
                            'label': label
                        })
                    else:
                        all_data.append({
                            'url': url,
                            'headline': None,
                            'body': None,
                            'news_type': news_type,
                            'label': label
                        })
                except Exception as e:
                    print(f"  -> Error scraping {url}: {e}")

# --- Create and Save DataFrame ---
if all_data:
    df = pd.DataFrame(all_data)
    print(f"\nScraping complete. Total records collected: {len(df)}")
    df.dropna(subset=['headline', 'body'], inplace=True)
    print(f"Dataset size after removing failed scrapes: {len(df)}")
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
    print("Dataset saved.")
else:
    print("\nNo data was successfully scraped.")
