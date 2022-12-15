import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric

# -------------- Settings ----------------
# page_title = "GMV Tracker"
# page_icon = ":money_with_wings:"

page_title = '<p style="font-family:Serif; font-weight:bold; color:maroon; font-size: 50px;">GMV Tracker </p>'
st.markdown(page_title, unsafe_allow_html=True)

# title
# st.title(page_title + " " + " " + page_icon)
# image of abhibus
st.sidebar.image("abhibus_logo.png", use_column_width=False)

# getting files 
data = st.sidebar.file_uploader('Browse Files',type='csv')

# date picker
date = st.sidebar.date_input("Select date", value=None, min_value=None, max_value=None, key=None)

if data is not None:
    # saving data in variable
    data = pd.read_csv(data)

    # changing column names for fbprophet
    data.columns = ["ds", "y"]
    # changing ds column which is date column to datetime
    data["ds"] = pd.to_datetime(data["ds"], errors="coerce")

    # initializing model
    m = Prophet()
    m.fit(data)

    # prediction
    future = m.make_future_dataframe(periods=730, freq="D")
    fcst = m.predict(future)
    forecast = fcst[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast['yhat'] = forecast['yhat'].div(45).round(2)
    forecast['yhat_lower'] = forecast['yhat_lower'].div(45).round(2)
    forecast['yhat_upper'] = forecast['yhat_upper'].div(45).round(2)


    new_forecast = forecast.copy()
    future_data = new_forecast.columns = ['Date', 'Revenue', 'Revenue Lower', 'Revenue Upper']
    


    # converting datetime to str for comparison
    forecast['ds'] = forecast['ds'].astype('str')

    # displaying date
    new_d = date.strftime("%Y-%m-%d")
    # print(forecast.dtypes)
    # print(type(new_d))
    # print(forecast)

    # forecast button
    if st.sidebar.button("Forecast"):
        # if new_d in forecast['ds'].unique():
        for i, j in zip(forecast['ds'], forecast['yhat']):
            if new_d == i:
                # j = j/50
                st.markdown(f'<p style="font-family:Serif; color:Brown; font-size: 30px;">Forecasting on {date.strftime("%d %B %Y")}</p>', unsafe_allow_html=True)
                st.text(f'Revenue estimated: ₹ {round(j, 2)}')
                st.markdown('<p style="font-family:Serif; color:Brown; font-size: 30px;">Future Data</p>', unsafe_allow_html=True)
                st.dataframe(new_forecast)
            
    # visualisation
    st.markdown('<p style="font-family:Serif; color:Brown; font-size: 30px;">Weekly Trend Analysis</p>', unsafe_allow_html=True)
    fig_2 = plot_components_plotly(m, fcst)
    fig_2

    st.markdown('<p style="font-family:Serif; color:Brown; font-size: 30px;">Financial Analysis</p>', unsafe_allow_html=True)
    fig = plot_plotly(m, forecast)
    fig
