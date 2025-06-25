from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
import re
import requests
import nltk
nltk.download('vader_lexicon')

finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'GOOG', 'MSFT']

news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    print(response)
    html = BeautifulSoup(response, features='html.parser')
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

parsed_data = []

for ticker, news_table in news_tables.items():

    # Eine Variable, um das Datum über die Zeilen hinweg zu "merken"
    letztes_datum = None

    for row in news_table.findAll('tr'):
        title = row.a.text
        link = row.a['href']
        
        # Säubern Sie den Text, bevor Sie ihn aufteilen
        date_data = row.td.text.strip().split(' ')

        # Überprüfen, ob die Zelle sowohl Datum als auch Uhrzeit enthält
        if len(date_data) == 2:
            letztes_datum = date_data[0] # Das Datum für die nächsten Zeilen speichern
            time = date_data[1]
        # Überprüfen, ob die Zelle nur die Uhrzeit enthält
        elif len(date_data) == 1:
            time = date_data[0] # Die Uhrzeit nehmen und das gespeicherte Datum verwenden
        else:
            # Ungültige Zeile überspringen
            continue

        # Nur hinzufügen, wenn wir ein gültiges Datum haben
        if letztes_datum:
            parsed_data.append([ticker, letztes_datum, time, title, link])

df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title', 'link'])

vader = SentimentIntensityAnalyzer()

f = lambda title: vader.polarity_scores(title)['compound']
df['compound'] = df['title'].apply(f)
print(df.head(5))



# Schritt 1: Konvertiere die Spalte und wandle Fehler in 'NaT' (Not a Time) um, anstatt abzustürzen.
# Wir verwenden df['date'] statt df.date, das ist sicherer.
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Schritt 2: Entferne alle Zeilen, bei denen die Datumsumwandlung fehlgeschlagen ist (also wo jetzt 'NaT' steht).
df.dropna(subset=['date'], inplace=True)

# Schritt 3: Da jetzt alle Daten gültig sind, extrahiere sicher nur den Datumsteil.
df['date'] = df['date'].dt.date

# --- Durchschnittlichen Compound-Wert pro Ticker und Tag berechnen und plotten ---
print("\nBerechne den durchschnittlichen Sentiment-Score pro Tag für jeden Ticker...")

# Nach Ticker UND Datum gruppieren, EXPLIZIT den Durchschnitt der 'compound'-Spalte berechnen,
# und dann die Ticker zu Spalten umformen. Das vermeidet den TypeError.
mean_df = df.groupby(['ticker', 'date'])['compound'].mean().unstack(level='ticker')

print("Ergebnis der täglichen Durchschnitts-Scores pro Ticker:")
print(mean_df)

# Einen gruppierten Balken-Chart erstellen.
# Pandas macht das automatisch, wenn der DataFrame die richtige Form hat (Daten im Index, Ticker als Spalten).
mean_df.plot(kind='bar', figsize=(14, 7))

# Titel und Achsenbeschriftungen für bessere Lesbarkeit hinzufügen
plt.title('Durchschnittlicher Sentiment-Score pro Tag, getrennt nach Ticker')
plt.xlabel('Datum')
plt.ylabel('Durchschnittlicher Compound Score')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title='Ticker')
plt.tight_layout()
plt.show()