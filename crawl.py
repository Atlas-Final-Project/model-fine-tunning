import feedparser
import pandas as pd
import time
from urllib.parse import quote
from datetime import datetime, timedelta
import os

# 1) 라벨별 키워드 정의
label_keywords = {
    0: ["earthquake", "fire", "flood", "typhoon", "volcano"],
    1: ["shooting", "murder", "robbery", "kidnapping", "arson"],
    2: ["covid", "outbreak", "infection", "quarantine"],
    3: ["tourism", "festival", "sports", "economy", "culture"],
    4: ["traffic accident", "flight delay", "politics", "weather"]
}

# 2) 지정 기간(start_date~end_date)을 'window' 일 단위로 쪼개 크롤링
def fetch_news_by_date_window(keyword, label, start_date, end_date, window=7):
    collected = []
    cursor = start_date
    while cursor < end_date:
        window_end = min(cursor + timedelta(days=window), end_date)
        # Google News 고급검색 구문: after:YYYY-MM-DD before:YYYY-MM-DD
        query = f'{keyword} after:{cursor.strftime("%Y-%m-%d")} before:{window_end.strftime("%Y-%m-%d")}'
        url = f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
        print(f"[INFO] {keyword!r} | {cursor}~{window_end} 요청 URL:", url)

        feed = feedparser.parse(url)
        for entry in feed.entries:
            text = entry.title.strip()
            if text and not any(d["text"] == text for d in collected):
                collected.append({"text": text, "label": label})
        time.sleep(1)
        cursor = window_end  # 다음 기간으로 이동

    print(f"[OK] '{keyword}' → 총 {len(collected)}건 수집")  
    return collected

# 3) 전체 기간(예: 2025-01-01 부터 오늘까지) 설정
START = datetime(2025, 1, 1)
END   = datetime.today()

all_data = []
for label, kws in label_keywords.items():
    for kw in kws:
        all_data.extend(fetch_news_by_date_window(
            keyword=kw,
            label=label,
            start_date=START,
            end_date=END,
            window=7          # 7일 치씩 잘라서 요청
        ))
        print("-" * 60)

# 4) DataFrame 생성 및 중복 제거
df = pd.DataFrame(all_data).drop_duplicates(subset=["text"]).reset_index(drop=True)
print(f"[RESULT] 총 {len(df)}건 수집 완료")

# 5) CSV로 저장
os.makedirs("dataset", exist_ok=True)
df.to_csv("dataset/news_dataset.csv", index=False, encoding="utf-8-sig")
print("[DONE] dataset/news_dataset.csv 저장")
