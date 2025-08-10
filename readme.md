# GenAI Stock Analyzer

A Streamlit-based web app that leverages LangGraph, LangChain, and OpenAI's GPT-4o-mini model to provide fundamental stock analysis using preprocessed stock price trends, technical indicators, and financial metrics.

---

## Features

- Fetches historical stock price data and computes technical indicators such as RSI, MACD, and VWAP.
- Retrieves key financial metrics (P/E ratio, Price-to-Book ratio, Debt-to-Equity ratio, Profit Margins) using Yahoo Finance.
- Integrates an AI-powered fundamental analyst powered by LangGraph and LangChain OpenAI tools to generate concise, structured analysis reports.
- Allows users to input a stock symbol, choose a time duration (1 day, 1 week, 1 month, 1 year), and ask custom questions.
- Presents analysis in a clear, tabbed interface with both textual summary and interactive stock price charts.
- Uses caching to optimize repeated stock data fetches.
- Modular tool design for extensibility and clear separation of concerns.

---

## Installation

1. Clone the repository:

   git clone https://github.com/amany4864/ai-financer/
   
   cd ai-financer

2. Install dependencies:

   pip install -r requirements.txt

3. Set up environment variables for OpenAI API key and others (optional, depending on your setup):

   Create a `.env` file with:

   OPENAI_API_KEY=your_openai_api_key_here

---

## Usage

Run the Streamlit app:

   streamlit run app.py

- Enter a valid stock ticker symbol (e.g., TSLA).
- Select a duration for historical data analysis.
- Enter a question to ask the AI analyst about the stock.
- Click **Analyze** to generate the report and view charts.

---

## Code Structure

- `app.py`: Main Streamlit application integrating LangGraph for managing AI workflow and user interface.
- `tools`: Functions decorated as `@tool` provide data fetching and summarization capabilities:
  - `get_stock_prices`: Fetches historical stock prices and calculates RSI, MACD, VWAP.
  - `get_financial_metrics`: Retrieves fundamental financial ratios.
  - `news_sentiment` (placeholder): Intended for integrating news sentiment analysis.
- `FUNDAMENTAL_ANALYST_PROMPT`: System prompt guiding the AI on how to interpret stock and financial data.
- `graph_builder`: Defines the LangGraph state machine for orchestrating calls between user input, AI, and tools.
- `plot_stock.py` (imported): Contains plotting logic for visualizing stock prices.

---

## Dependencies

- Streamlit — Interactive web UI
- yfinance — Yahoo Finance API
- pandas — Data manipulation
- matplotlib — Plotting
- ta — Technical analysis indicators
- langgraph — AI workflow orchestration
- langchain — LLM chaining
- python-dotenv — Environment variable management
- OpenAI Python SDK (via `langchain_openai`)

---

## Notes

- The app currently uses the GPT-4o-mini model via LangChain's OpenAI wrapper.
- The news sentiment tool is a placeholder and needs implementation for actual news scraping and sentiment analysis.
- Stock data visualization uses a custom `plot_stock_data` function (in `plot_stock.py`).
- Ensure proper API keys and environment variables are set for OpenAI and any other services used.
- Streamlit's caching (`@st.cache_data`) is used to improve performance on repeated data fetches.

---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or contributions, please open an issue or contact the maintainer.

---

*Enjoy analyzing stocks with AI-powered insights!*
