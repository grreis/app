#!/usr/bin/env python
# coding: utf-8

# In[45]:


import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

import pypfopt
from pypfopt import black_litterman, risk_models
from pypfopt import BlackLittermanModel, plotting
from pypfopt import EfficientFrontier, objective_functions


# In[ ]:


st.write("""
# Black-Litterman Asset Allocation
Model-based asset allocation with investor views
""")

st.sidebar.header('Investor Return Views')

def user_input_features():
    Petro = st.sidebar.slider('PETR4',0.0,0.25,0.05)
    Itau = st.sidebar.slider('ITUB4',0.0,0.25,0.05)
    Magalu = st.sidebar.slider('MGLU3',0.0,0.25,0.15)
    Ambev = st.sidebar.slider('ABEV3',0.0,0.25,0.1)
    Vivo = st.sidebar.slider('VIVT4',0.0,0.25,0.05)
    data = {'PETR4.SA': Petro,
           'ITUB4.SA': Itau,
           'MGLU3.SA': Magalu,
           'ABEV3.SA': Ambev,
           'VIVT4.SA': Vivo}
    features = pd.DataFrame(data,index=[0])
    return features

df = user_input_features()

def allocate(df):
    tickers = [df.columns[k] for k in range(df.shape[1])]

    ohlc = yf.download(tickers,start="2010-01-01",end="2020-01-01")
    prices = ohlc["Adj Close"]

    market_prices = yf.download("^BVSP", start="2010-01-01",end="2020-01-01")["Adj Close"]

    mcaps = {}
    for t in tickers:
        stock = yf.Ticker(t)
        mcaps[t] = stock.info["marketCap"]
    
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    delta = black_litterman.market_implied_risk_aversion(market_prices)

    market_prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)

    bl = BlackLittermanModel(S, pi="market", market_caps=mcaps, risk_aversion=delta,
                            absolute_views=df.to_dict('records')[0])

    ret_bl = bl.bl_returns()
    S_bl = bl.bl_cov()

    ef = EfficientFrontier(ret_bl, S_bl)
    ef.add_objective(objective_functions.L2_reg)
    ef.max_sharpe()
    weights = ef.clean_weights()
    
    return weights
    
weights = pd.Series(allocate(df))
labels = [weights.index[k].rstrip(".SA") for k in range(weights.size)]
sizes = weights.values

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal') 

st.subheader('B-L Optimal Portfolio Allocation')
st.pyplot(fig1)

st.markdown('Gustavo Reis, 2020')


# In[ ]:




