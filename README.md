To run the whole project, make sure to install all the packages in 
- stocks/views.py
- utils/pytorch_models.py
- utils/train_pytorch.py

When launching the Project and the Predictions is not appearing, then:
(Also advisable if you are willing to see better predictions using new data)

1. Delete the whole models folder
2. In terminal, run the model again as follows:

> python manage.py shell

> from utils.train_pytorch import train_model
> from stocks.views import get_stock_history_cached
> df = get_stock_history_cached("AAPL", period="2y")
> train_model(df)

**Disclaimer**: This project and report was created with the assistance of AI tools, including ChatGPT, to help in structuring and drafting the content. While the AI contributed to the writing process, all technical details, analysis, and final content have been carefully reviewed and validated by the authors. Every effort has been made to ensure the accuracy and reliability of the information presented. 
