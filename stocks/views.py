from django.shortcuts import render
from django.http import JsonResponse
from django.core.cache import cache
import yfinance as yf
import json
import time

from .models import ListAmericanCompanies


def get_stock_history_cached(ticker, period='2y', cache_timeout=60):
    """
    Fetch stock history for (ticker, period) from cache if available;
    otherwise, request from yfinance and store the result in cache.
    """
    cache_key = f"stock_history_{ticker}_{period}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    try:
        data = yf.Ticker(ticker).history(period=period)
    except Exception as e:
        # Check if the exception says "Too Many Requests" or "Rate limited"
        if "Too Many Requests" in str(e) or "Rate limited" in str(e):
            return None  # Return None to indicate a rate-limit issue
        else:
            raise  # Some other error, raise it

    cache.set(cache_key, data, cache_timeout)
    return data


def get_stock_info_cached(ticker, cache_timeout=60):
    """
    Fetch stock info (metadata) from cache if available;
    otherwise, request from yfinance and store it in cache.
    """
    cache_key = f"stock_info_{ticker}"
    cached_info = cache.get(cache_key)
    if cached_info is not None:
        return cached_info

    try:
        info = yf.Ticker(ticker).info
    except Exception as e:
        if "Too Many Requests" in str(e) or "Rate limited" in str(e):
            return None
        else:
            raise

    cache.set(cache_key, info, cache_timeout)
    return info


def get_other_stocks(main_ticker):
    """
    Fetch related stocks based on industry, sector, or exchange.
    Include a small delay to avoid burst requests to Yahoo.
    """
    try:
        main_info = get_stock_info_cached(main_ticker)
        if not main_info:
            return []  # Possibly rate-limited or error

        industry = main_info.get('industry')
        sector = main_info.get('sector')
        exchange = main_info.get('exchange')  # e.g. "NASDAQ"

        # Fetch up to 50 potential related stocks
        similar_stocks = ListAmericanCompanies.objects.exclude(ticker=main_ticker)[:50]

        other_stocks = []
        for entry in similar_stocks:
            time.sleep(0.2)  # Throttle slightly between each request

            s_info = get_stock_info_cached(entry.ticker)
            if not s_info:
                continue

            same_industry = industry and (s_info.get('industry') == industry)
            same_sector = sector and (s_info.get('sector') == sector)
            same_exchange = exchange and (s_info.get('exchange') == exchange)

            if same_industry or same_sector or same_exchange:
                # Get today's data
                day_data = get_stock_history_cached(entry.ticker, period="1d", cache_timeout=60)
                if day_data is not None and not day_data.empty:
                    current_price = round(day_data['Close'].iloc[-1], 2)
                    open_price = day_data['Open'].iloc[-1]
                    change = round(((current_price - open_price) / open_price) * 100, 2)
                else:
                    current_price = "N/A"
                    change = "N/A"

                other_stocks.append({
                    'ticker': entry.ticker,
                    'name': entry.title,
                    'price': current_price,
                    'change': change
                })

            if len(other_stocks) >= 4:
                break

        return other_stocks

    except Exception as e:
        print(f"Error fetching related stocks for {main_ticker}: {e}")
        return []


def stock_search(request):
    """
    Main stock search view using caching & graceful handling of rate-limit errors.
    """
    context = {}
    ticker = request.GET.get('ticker', 'AAPL').upper()

    if request.method == 'POST':
        ticker = request.POST.get('ticker', 'AAPL').upper().split(' - ')[0]

    period = request.GET.get('period', '2y')  # e.g., 3mo, 6mo, 1y, etc.

    try:
        # Fetch daily data
        today_data = get_stock_history_cached(ticker, period='1d')
        # Fetch data for 2 days to get yesterday's close
        yesterday_data = get_stock_history_cached(ticker, period='2d')
        # Main historical data
        hist_data = get_stock_history_cached(ticker, period=period)

        # If hist_data is None, it likely indicates a rate-limit error
        if hist_data is None:
            context['error'] = "Yahoo Finance rate limit exceeded. Please try again in a minute."
            return render(request, 'index.html', context)

        # Compute 50-day EMA
        hist_data['EMA50'] = hist_data['Close'].ewm(span=50, adjust=False).mean()
        hist_data.dropna(inplace=True)

        # Prepare chart data
        chart_data = {
            'dates': hist_data.index.strftime('%Y-%m-%d').tolist(),
            'closes': hist_data['Close'].round(2).tolist(),
            'ema50': hist_data['EMA50'].round(2).tolist(),
        }

        # Fetch stock info
        info = get_stock_info_cached(ticker)
        if info is None:
            context['error'] = "Yahoo Finance rate limit exceeded. Please try again in a minute."
            return render(request, 'index.html', context)

        # Summarize daily stats
        if today_data is not None and not today_data.empty:
            current_price = round(today_data['Close'].iloc[-1], 2)
            todays_open = round(today_data['Open'].iloc[-1], 2)
            volume = f"{int(today_data['Volume'].iloc[-1]):,}"
        else:
            current_price = "N/A"
            todays_open = "N/A"
            volume = "N/A"

        if yesterday_data is not None and len(yesterday_data) > 1:
            yesterdays_close = round(yesterday_data['Close'].iloc[-2], 2)
        else:
            yesterdays_close = "N/A"

        trailing_pe = info.get('trailingPE')
        pe_ratio = round(trailing_pe, 2) if trailing_pe else "N/A"

        # Fetch other/related stocks
        other_stocks = get_other_stocks(ticker)

        # Build context for template
        context = {
            'ticker': ticker,
            'Company': info.get('longName', ticker),
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

        # If AJAX request (chart period buttons), return JSON only
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'chart_data': chart_data})

    except Exception as e:
        error_msg = str(e)
        # Check if it's rate-limit
        if "Too Many Requests" in error_msg or "Rate limited" in error_msg:
            context['error'] = "Yahoo Finance rate limit exceeded. Please try again in a minute."
        else:
            context['error'] = f"Error fetching data for {ticker}: {error_msg}"

    return render(request, 'index.html', context)
