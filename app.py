# app.py
from langgraph.graph import START, END # type: ignore
import streamlit as st
import pandas as pd
import datetime as dt
import yfinance as yf
from statistics import mean
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volume import volume_weighted_average_price
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from typing import Union, Dict, Annotated, TypedDict
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
import matplotlib.pyplot as plt
import dotenv
# Add these imports at the top if not already there
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
from plot_stock import plot_stock_data

dotenv.load_dotenv()


# ========== PROMPT ==========
# 3. **news_sentiment**: Fetches company-specific news and world events (like wars, geopolitics, and economic conditions) to analyze their impact on the stock.
# - Always call the news_sentiment tool for every analysis, and always include its output as the "news_sentiment" field in your response.

FUNDAMENTAL_ANALYST_PROMPT = """
You are a fundamental analyst specializing in evaluating a company's performance (whose stock symbol is {company}) using **preprocessed** stock price trends, technical indicators, and financial metrics.

You have access to the following tools:
1. **get_stock_prices**: Provides summarized stock price insights and preprocessed technical indicators like RSI trend, MACD signal, and VWAP average.
2. **get_financial_metrics**: Returns summarized key financial metrics such as P/E ratio, price-to-book, debt-to-equity, and profitability classification.
### Your Objective:
Analyze the data returned by these tools and provide a concise, structured analysis report.

### Guidelines:
- The raw data is **already preprocessed** and summarized for you. Focus your reasoning on interpreting these summaries, not on computing metrics.
- Be objective, concise, and avoid speculation.
- Mention any tool failures explicitly if data is missing.
 
### Output Format:
"stock": "<Stock Symbol>",
"price_analysis": "<Brief summary of recent price trends and volatility>",
"technical_analysis": "<Summary based on RSI trend, MACD signal, VWAP>",
"financial_analysis": "<Financial health based on summarized metrics>",
"final Summary": "<Conclusion from all above components>",
"Asked Question Answer": "<Answer user's question using the analysis above>"

Your goal is to provide professional, decision-relevant insights based on preprocessed and condensed data.
"""

# ========== TOOLS ==========


@tool
def get_stock_prices(ticker: str, days: int = 365) -> Union[Dict, str]:
    """
    Fetch summarized stock price and technical indicator insights to reduce token usage.
    Also returns raw data for plotting in Streamlit (not sent to LLM).
    """
    try:
        start_date = dt.datetime.today() - dt.timedelta(days=days)
        end_date = dt.datetime.today()

        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            interval='1d',
            auto_adjust=True
        )

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        if data.empty or len(data) < 20:
            return f"Only {len(data)} rows — not enough historical stock data."

        df = data.copy()
        df.reset_index(inplace=True)

        latest_close = df['Close'].iloc[-1]
        previous_close = df['Close'].iloc[-2]
        percent_change = round((latest_close - previous_close) / previous_close * 100, 2)
        close_avg = round(df['Close'].mean(), 2)

        price_summary = {
            "latest_close": round(latest_close, 2),
            "previous_close": round(previous_close, 2),
            "percent_change_1d": percent_change,
            "average_close": close_avg
        }

        rsi_series = RSIIndicator(df['Close'], window=14).rsi().dropna()
        rsi_trend = "Overbought" if rsi_series.iloc[-1] > 70 else "Oversold" if rsi_series.iloc[-1] < 30 else "Neutral"

        macd_obj = MACD(df['Close'])
        macd_vals = macd_obj.macd().dropna()
        macd_signal_vals = macd_obj.macd_signal().dropna()
        macd_trend = (
            "Bullish crossover" if macd_vals.iloc[-1] > macd_signal_vals.iloc[-1]
            else "Bearish crossover" if macd_vals.iloc[-1] < macd_signal_vals.iloc[-1]
            else "Neutral"
        )

        vwap_series = volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume']).dropna()
        recent_vwap = round(mean(vwap_series.iloc[-3:]), 2)

        indicators = {
            "RSI_trend": rsi_trend,
            "MACD_trend": macd_trend,
            "recent_VWAP_avg": recent_vwap
        }

        # ✅ Return also raw price data for plotting
        plot_data = df[['Date', 'Close']].tail(min(90, len(df))).to_dict(orient='records')
        return {
        'price_summary': price_summary,
        'indicators': indicators,
        'plot_data': plot_data
        }

    except Exception as e:
        return f"Error fetching price data: {str(e)}"


@tool
def get_financial_metrics(ticker: str) -> Union[Dict, str]:
    """
    Financial summary with basic evaluation.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        pe = info.get('forwardPE')
        pb = info.get('priceToBook')
        dte = info.get('debtToEquity')
        pm = info.get('profitMargins')

        return {
            "pe_ratio": pe,
            "pe_evaluation": "High" if pe and pe > 40 else "Low" if pe and pe < 10 else "Moderate",
            "price_to_book": pb,
            "pb_evaluation": "High" if pb and pb > 5 else "Reasonable",
            "debt_to_equity": dte,
            "dte_evaluation": "High Leverage" if dte and dte > 2 else "Low/Moderate Leverage",
            "profit_margins": pm,
            "profitability": "Good" if pm and pm > 0.1 else "Low" if pm and pm < 0.05 else "Average"
        }

    except Exception as e:
        return f"Error fetching financial metrics: {str(e)}"
    
@tool
def news_sentiment(ticker: str) -> str:
    """
    Fetches company and world news and returns a sentiment summary string.
    """
    company_news = get_yfinance_news(ticker, max_items=5)
    world_news = get_google_news("war OR conflict OR geopolitics OR economy", max_items=5)
    prompt = build_news_sentiment_prompt(company_news, world_news)
    # Actually run the LLM on this prompt and return the result!
    return llm.invoke([SystemMessage(content=prompt)]).content

# ========== LANGGRAPH SETUP ==========
class State(TypedDict):
    messages: Annotated[list, add_messages]
    stock: str
    days: int

tools = [get_stock_prices, get_financial_metrics]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tool = llm.bind_tools(tools)

def fundamental_analyst(state: State):
    prompt = FUNDAMENTAL_ANALYST_PROMPT.format(company=state['stock'], days=state['days'])
    messages = [SystemMessage(content=prompt)] + state['messages']
    return {'messages': llm_with_tool.invoke(messages)}

graph_builder = StateGraph(State)
graph_builder.add_node("fundamental_analyst", fundamental_analyst)
graph_builder.add_edge(START, "fundamental_analyst")
graph_builder.add_node(ToolNode(tools))
graph_builder.add_conditional_edges("fundamental_analyst", tools_condition)
graph_builder.add_edge("tools", "fundamental_analyst")
graph_builder.add_edge('tools', 'fundamental_analyst')
graph_builder.add_edge('fundamental_analyst', END)    # ← add this
graph = graph_builder.compile()



import streamlit as st
import pandas as pd
import datetime as dt
import yfinance as yf
import json

st.set_page_config(page_title='Stock Analyzer', layout='centered')
st.title('📊 GenAI Stock Analyzer')

ticker = st.text_input('Stock Symbol (e.g. TSLA)')
duration = st.selectbox('Duration', ['1 Day', '1 Week', '1 Month', '1 Year'])
dmap = {'1 Day': 1, '1 Week': 7, '1 Month': 30, '1 Year': 365}
days = dmap[duration]
query = st.text_input('Your Question')

@st.cache_data(show_spinner=False)
def fetch_stock_data(ticker, days):
    start_date = dt.datetime.today() - dt.timedelta(days=days)
    end_date = dt.datetime.today()
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval='1d',
        auto_adjust=True,
        progress=False
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.reset_index(inplace=True)
    return df

if st.button('Analyze'):
    if not ticker or not query:
        st.warning('Enter both symbol and question')
        st.stop()
    with st.spinner('Running analysis...'):
        # Fetch stock data
        df = fetch_stock_data(ticker, days)
        npoints = len(df)
        # Call the LLM via LangGraph
        events = graph.stream(
            {'messages': [('user', query)], 'stock': ticker, 'days': days},
            stream_mode='values'
        )
        msg = None
        for e in events:
            if 'messages' in e:
                msg = e['messages'][-1].content
        if not msg:
            st.error('No response')
            st.stop()
        try:
            analysis = json.loads(msg)
        except Exception:
            st.text_area('AI Response', msg, height=200)
            st.stop()
        st.success('✅ Analysis Complete')

    tab1, tab2 = st.tabs(['📋 Report', '📈 Chart & Data'])

    with tab1:
        st.header(f"{analysis['stock']} Analysis Summary")
        st.markdown(f"**Data Window:** Last {days} days ({npoints} trading days)")
        st.subheader('Price Analysis')
        st.write(analysis.get('price_analysis', 'N/A'))
        st.subheader('Technical Analysis')
        st.write(analysis.get('technical_analysis', 'N/A'))
        st.subheader('Financial Analysis')
        st.write(analysis.get('financial_analysis', 'N/A'))
        # st.subheader('News Sentiment (Company & World)')
        # st.write(analysis.get('news_sentiment', 'N/A'))
        # st.subheader('Conclusion')
        # st.info("ℹ️ The AI analysis uses both company-specific news and world events (such as war, geopolitics, and economy) to inform its suggestions.")
        st.info(analysis.get('final Summary', 'N/A'))
        st.subheader('Answer to Your Question')
        st.write(analysis.get('Asked Question Answer', 'N/A'))

    with tab2:
        if npoints > 0:
            st.subheader('Underlying Closing Prices')
            display_df = df[['Date', 'Close']].copy()
            display_df['Close'] = display_df['Close'].round(2)
            st.dataframe(display_df.rename(columns={'Close': 'Close ($)'}), height=300)
            # To add your custom chart, import and call your plot_stock_data function here.
            from plot_stock import plot_stock_data
            fig = plot_stock_data(df, ticker, days, npoints)
            st.pyplot(fig)
        else:
            st.info('ℹ️ No data available to display.')