from django.shortcuts import render
import yfinance as yf
import json
from django.http import JsonResponse
import random
from .models import ListAmericanCompanies


def get_other_stocks(main_ticker):
    """
    Fetch related stocks based on industry, sector, or exchange.
    Falls back to the same exchange if industry/sector is unavailable.
    """
    try:
        main_stock = yf.Ticker(main_ticker)
        stock_info = main_stock.info

        industry = stock_info.get('industry')
        sector = stock_info.get('sector')
        exchange = stock_info.get('exchange')  # Example: "NASDAQ"

        # Fetch potential related stocks
        similar_stocks = ListAmericanCompanies.objects.exclude(ticker=main_ticker)[:50]

        other_stocks = []
        for stock_entry in similar_stocks:
            try:
                stock = yf.Ticker(stock_entry.ticker)
                stock_info = stock.info

                same_industry = industry and stock_info.get('industry') == industry
                same_sector = sector and stock_info.get('sector') == sector
                same_exchange = exchange and stock_info.get('exchange') == exchange

                # Prioritize industry, then sector, then exchange
                if same_industry or same_sector or same_exchange:
                    today_data = stock.history(period="1d")
                    current_price = round(today_data['Close'].iloc[-1], 2) if not today_data.empty else "N/A"
                    open_price = today_data['Open'].iloc[-1] if not today_data.empty else None
                    change = round(((current_price - open_price) / open_price * 100), 2) if open_price else "N/A"

                    other_stocks.append({
                        'ticker': stock_entry.ticker,
                        'name': stock_entry.title,
                        'price': current_price,
                        'change': change
                    })

                if len(other_stocks) >= 4:
                    break

            except Exception as e:
                print(f"Error fetching data for {stock_entry.ticker}: {e}")
                continue

        return other_stocks

    except Exception as e:
        print(f"Error fetching main stock data: {e}")
        return []


def stock_search(request):
    """
    Main stock search view. Fetches stock price, chart data, and related stocks.
    """
    context = {}
    ticker = request.GET.get('ticker', 'AAPL').upper()  # Default to AAPL

    if request.method == 'POST':
        ticker = request.POST.get('ticker', 'AAPL').upper().split(' - ')[0]

    try:
        stock = yf.Ticker(ticker)
        today_data = stock.history(period="1d")
        yesterday_data = stock.history(period="2d")
        hist_data = stock.history(period="2y")

        # Compute 50-day Exponential Moving Average
        hist_data['EMA50'] = hist_data['Close'].ewm(span=50, adjust=False).mean()
        hist_data.dropna(inplace=True)
        hist_data = hist_data.tail(252)  # Limit to last year

        # Chart Data for Frontend
        chart_data = {
            'dates': hist_data.index.strftime('%Y-%m-%d').tolist(),
            'closes': hist_data['Close'].round(2).tolist(),
            'ema50': hist_data['EMA50'].round(2).tolist(),
        }

        # Stock Info
        current_price = round(today_data['Close'].iloc[-1], 2) if not today_data.empty else "N/A"
        todays_open = round(today_data['Open'].iloc[-1], 2) if not today_data.empty else "N/A"
        yesterdays_close = round(yesterday_data['Close'].iloc[-2], 2) if len(yesterday_data) > 1 else "N/A"
        pe_ratio = round(stock.info.get('trailingPE', 0), 2) if stock.info.get('trailingPE') else "N/A"
        volume = f"{today_data['Volume'].iloc[-1]:,}" if not today_data.empty else "N/A"

        # Get related stocks
        other_stocks = get_other_stocks(ticker)

        context = {
            'ticker': ticker,
            'Company': stock.info.get('longName', ticker),
            'current_price': current_price,
            'chart_data': json.dumps(chart_data),
            'info': {
                'Today\'s Open': todays_open,
                'P/E Ratio': pe_ratio,
                'Yesterday\'s Close': yesterdays_close,
                'Volume': volume
            },
            'other_stocks': other_stocks
        }

        # Return JSON for AJAX requests
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'chart_data': chart_data})

    except Exception as e:
        context['error'] = f"Error fetching data for {ticker}: {str(e)}"

    return render(request, 'index.html', context)
