import yfinance as yf
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.regime_switching import markov_autoregression

end_date = pd.Timestamp.now().date()
start_date = '2020-01-01'
tickers_list = ['AAPL', 'AMZN', 'MSFT']

data = yf.download(tickers_list, start=start_date, end=end_date)['Adj Close']
log_returns = np.log(data / data.shift(1))

# Estimando volatilidade usando Markov-Switching Autoregressive
markov_volatility = {}
for ticker in tickers_list:
    model = markov_autoregression.MarkovAutoregression(log_returns[ticker].dropna(), k_regimes=2, order=1, trend='c',
                                                       switching_ar=True)
    res = model.fit(disp="off")
    markov_volatility[ticker] = res.smoothed_marginal_probabilities[1]

markov_volatility_df = pd.DataFrame(markov_volatility)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Volatilidade Markov-Switching Autoregressive de Ações"),
    dcc.Dropdown(
        id='ticker-dropdown',
        options=[{'label': ticker, 'value': ticker} for ticker in tickers_list],
        value='AAPL'
    ),
    dcc.Graph(id='markov-graph')
])


@app.callback(
    Output('markov-graph', 'figure'),
    [Input('ticker-dropdown', 'value')]
)
def update_graph(ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=markov_volatility_df.index,
                             y=markov_volatility_df[ticker],
                             mode='lines',
                             name=ticker,
                             line=dict(color='purple')))

    median_volatility = markov_volatility_df[ticker].median()

    fig.add_shape(
        type="line",
        x0=markov_volatility_df.index[0],
        x1=markov_volatility_df.index[-1],
        y0=median_volatility,
        y1=median_volatility,
        line=dict(dash="dot", color="gold"),
    )

    above_median = markov_volatility_df[ticker] > median_volatility
    for i, val in enumerate(above_median):
        if val:
            if i == 0 or not above_median[i - 1]:
                x_start = markov_volatility_df.index[i]
            if i == len(above_median) - 1 or not above_median[i + 1]:
                x_end = markov_volatility_df.index[i]
                fig.add_shape(
                    type="rect",
                    x0=x_start,
                    x1=x_end,
                    y0=0,
                    y1=1,
                    fillcolor="blue",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                )

    fig.update_layout(
        title=f'Volatilidade Markov-Switching para {ticker}',
        xaxis_title='Data',
        yaxis_title='Probabilidade do Regime de Alta Volatilidade',
        xaxis=dict(color='white'),
        yaxis=dict(color='white'),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)