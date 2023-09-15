import yfinance as yf
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from arch import arch_model

end_date = pd.Timestamp.now().date()
start_date = '2020-01-01'
tickers_list = ['BTC-USD', 'ETH-USD']

data = yf.download(tickers_list, start=start_date, end=end_date)['Adj Close']
log_returns = np.log(data / data.shift(1))

# Estimando volatilidade usando GARCH(1,1)
garch_volatility = {}
for ticker in tickers_list:
    model = arch_model(log_returns[ticker].dropna(), vol="Garch", p=1, q=1)
    res = model.fit(disp="off")
    garch_volatility[ticker] = res.conditional_volatility

garch_volatility_df = pd.DataFrame(garch_volatility)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Volatilidade GARCH de Ações"),
    dcc.Dropdown(
        id='ticker-dropdown',
        options=[{'label': ticker, 'value': ticker} for ticker in tickers_list],
        value='AAPL'
    ),
    dcc.Graph(id='garch-graph')
])


@app.callback(
    Output('garch-graph', 'figure'),
    [Input('ticker-dropdown', 'value')]
)
def update_graph(ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=garch_volatility_df.index,
                             y=garch_volatility_df[ticker],
                             mode='lines',
                             name=ticker))

    median_volatility = garch_volatility_df[ticker].median()

    fig.add_shape(
        type="line",
        x0=garch_volatility_df.index[0],
        x1=garch_volatility_df.index[-1],
        y0=median_volatility,
        y1=median_volatility,
        line=dict(dash="dot"),
    )

    above_median = garch_volatility_df[ticker] > median_volatility
    for i, val in enumerate(above_median):
        if val:
            if i == 0 or not above_median[i - 1]:
                x_start = garch_volatility_df.index[i]
            if i == len(above_median) - 1 or not above_median[i + 1]:
                x_end = garch_volatility_df.index[i]
                fig.add_shape(
                    type="rect",
                    x0=x_start,
                    x1=x_end,
                    y0=0,
                    y1=2 * median_volatility,
                    fillcolor="lightgray",
                    opacity=0.5,
                    layer="below",
                    line_width=0,
                )

    fig.update_layout(title=f'Volatilidade GARCH para {ticker}',
                      xaxis_title='Data',
                      yaxis_title='Volatilidade')

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)