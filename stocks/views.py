from django.shortcuts import render
import yfinance as yf
import json
from django.http import JsonResponse
# import random
# from .models import ListAmericanCompanies


def stock_search(request):
    context = {}

    # Default values
    ticker = "AAPL"
    period = request.GET.get('period', '1y')  # Default to 1 year if no period is provided

    if request.method == 'POST':
        ticker = request.POST.get('ticker', 'AAPL').upper().split(' - ')[0]

    try:
        stock = yf.Ticker(ticker)
        today_data = stock.history(period="1d")
        yesterday_data = stock.history(period="2d")
        hist_data = stock.history(period="max" if period == "max" else "2y")

        hist_data['EMA50'] = hist_data['Close'].ewm(span=50, adjust=False).mean()

        hist_data.dropna(inplace=True)

        hist_data = hist_data.tail({'1y': 252, '5y': 1260, 'max': len(hist_data)}[period])

        # Extract chart data
        chart_data = {
            'dates': hist_data.index.strftime('%Y-%m-%d').tolist(),
            'closes': hist_data['Close'].round(2).tolist(),
            'ema50': hist_data['EMA50'].round(2).tolist(),

        }

        # Retrieve and round values safely
        current_price = round(today_data['Close'].iloc[-1], 2) if not today_data.empty else "N/A"
        todays_open = round(today_data['Open'].iloc[-1], 2) if not today_data.empty else "N/A"
        yesterdays_close = round(yesterday_data['Close'].iloc[-2], 2) if len(yesterday_data) > 1 else "N/A"
        pe_ratio = round(stock.info.get('trailingPE', 0), 2) if stock.info.get('trailingPE') else "N/A"

        context = {
            'ticker': ticker,
            'Company': stock.info['longName'],
            'current_price': current_price,
            'chart_data': json.dumps(chart_data),
            'info': {
                'Today\'s Open': todays_open,
                'P/E Ratio': pe_ratio,
                'Yesterday\'s Close': yesterdays_close,
                'Volume': f"{today_data['Volume'].iloc[-1]:,}" if not today_data.empty else "N/A"
            }
        }

        # If it's an AJAX request, return JSON response
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'chart_data': chart_data})

    except Exception as e:
        context['error'] = f"Error fetching data for {ticker}: {str(e)}"

    return render(request, 'index.html', context)


#def get_other_stocks():
#    all_stocks = list(ListAmericanCompanies.objects.all())
#    selected_stocks = random.sample(all_stocks, min(len(all_stocks), 4))  # Pick 4 random stocks
#
#    other_stocks = []
#    for stock_entry in selected_stocks:
#        try:
#            stock = yf.Ticker(stock_entry.ticker)
#            today_data = stock.history(period="1d")
#
#            current_price = round(today_data['Close'].iloc[-1], 2) if not today_data.empty else "N/A"
#            open_price = today_data['Open'].iloc[-1] if not today_data.empty else None
#
#            change = round(((current_price - open_price) / open_price * 100), 2) if open_price else "N/A"
#
#            other_stocks.append({
#                'ticker': stock_entry.ticker,
#                'name': stock_entry.title,  # Use title from the database
#                'price': current_price,
#                'change': change
#            })
#        except Exception as e:
#            print(f"Error fetching data for {stock_entry.ticker}: {e}")
#            continue  # Skip to next stock instead of breaking
#
#    return other_stocks