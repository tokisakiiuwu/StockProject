import requests


ALPHAVANTAGE_API_KEY = "QQDJEE91SKUM45X2"

def get_stock_data():
    ticker = 'NVDA'
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={ALPHAVANTAGE_API_KEY}'
    r = requests.get(url)

    return r.json()


