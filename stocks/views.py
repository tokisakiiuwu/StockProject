from django.shortcuts import render
from django.http import JsonResponse
from django.core.cache import cache
import pandas as pd
import yfinance as yf
import json
import time

from .models import ListAmericanCompanies

from scipy.optimize import minimize
import numpy as np


def optimize_portfolio(request):
    """
    This view takes the tickers and allocations submitted by the user,
    performs a basic Minimum Variance Optimization, and returns the optimized portfolio.
    """
    if request.method == 'POST':
        try:
            # Parse the incoming JSON request data
            data = json.loads(request.body)
            tickers = data.get('tickers')
            allocations = data.get('allocations')

            if len(tickers) != len(allocations):
                return JsonResponse({'error': 'Mismatch between tickers and allocations lengths'}, status=400)

            # Download stock price data for the selected tickers
            stock_data = fetch_stock_data(tickers)

            # Convert allocation percentages to decimal (e.g., 50% -> 0.5)
            allocations = np.array(allocations) / 100

            # Calculate the covariance matrix of the returns
            returns = stock_data.pct_change().mean()  # Mean daily returns
            cov_matrix = stock_data.pct_change().cov()  # Covariance matrix

            # Perform optimization (minimum variance portfolio)
            # Calculate the portfolio variance for different weights
            num_stocks = len(tickers)
            portfolio_variances = []
            portfolio_returns = []
            for i in range(10000):  # Randomly simulate 10,000 portfolios
                weights = np.random.random(num_stocks)
                weights /= np.sum(weights)  # Normalize the weights to sum to 1

                # Portfolio return and variance
                port_return = np.sum(weights * returns)
                port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

                portfolio_returns.append(port_return)
                portfolio_variances.append(port_variance)

            # Find the minimum variance portfolio
            min_variance_idx = np.argmin(portfolio_variances)
            optimal_weights = np.random.random(num_stocks)
            optimal_weights /= np.sum(optimal_weights)

            # Create an optimized portfolio result
            optimized_portfolio = {
                tickers[i]: optimal_weights[i] * 100 for i in range(len(tickers))
            }

            return JsonResponse({'portfolio': optimized_portfolio})

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

def portfolio(request):
    if request.method == 'POST':
        tickers = request.POST.getlist('tickers')  # Get selected stocks from form
        risk_level = request.POST.get('risk_level', 'neutral')

        if not tickers:
            return render(request, 'portfolio.html', {'error': 'Please select at least one stock.'})

        # Fetch stock data
        stock_data = fetch_stock_data(tickers)

        # Compute daily returns
        returns = stock_data.pct_change().dropna()

        # Compute mean returns and covariance matrix
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # Number of assets
        num_assets = len(tickers)

        # Initial weights
        init_guess = np.ones(num_assets) / num_assets

        # Constraints: Weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        # Bounds: Each weight between 0 and 1
        bounds = tuple((0, 1) for _ in range(num_assets))

        # Objective function: Minimize portfolio variance (risk-averse)
        def portfolio_variance(weights, cov_matrix):
            return np.dot(weights.T, np.dot(cov_matrix, weights))

        # Adjust optimization based on risk level
        if risk_level == 'averse':
            result = minimize(portfolio_variance, init_guess, args=(cov_matrix,), method='SLSQP', bounds=bounds,
                              constraints=constraints)
        else:
            # If risk-neutral or seeking returns, maximize Sharpe ratio
            def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.02):
                portfolio_return = np.dot(weights, mean_returns)
                portfolio_volatility = np.sqrt(portfolio_variance(weights, cov_matrix))
                return - (portfolio_return - risk_free_rate) / portfolio_volatility

            result = minimize(neg_sharpe_ratio, init_guess, args=(mean_returns, cov_matrix), method='SLSQP',
                              bounds=bounds, constraints=constraints)

        # Extract optimal weights
        optimal_weights = result.x if result.success else init_guess

        # Construct portfolio allocation
        portfolio_allocation = {tickers[i]: round(optimal_weights[i] * 100, 2) for i in range(num_assets)}

        return render(request, 'portfolio.html',
                      {'portfolio_allocation': portfolio_allocation, 'tickers': tickers, 'risk_level': risk_level})

    return render(request, 'portfolio.html')

def portfolio(request):
    return render(request, 'portfolio.html')

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


def fetch_stock_data(tickers):
    """
    Fetch historical stock data for the provided tickers and return as a DataFrame.
    """
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        df = stock.history(period="1y")["Close"]
        data[ticker] = df
    return pd.DataFrame(data)