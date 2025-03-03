import streamlit as st 
import json 
from prophet.serialize import model_from_json 
import pandas as pd 
from prophet.plot import plot_plotly 

# Function to load the model
def load_model():
    try:
        with open('model_03_prophet.json', 'r') as file_in:
            model = model_from_json(json.load(file_in))
            return model 
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Load the model
model = load_model()
if model is None:
    st.stop()  # Stops execution if the model fails to load

# Streamlit Layout 
st.title('Ozone (O3) Level Forecasting Using the Prophet Library')

st.caption('''This project uses the Prophet library to predict ozone levels in ug/m3. The model
           was trained with data up to 05/05/2023 and has a forecast error (RMSE - Root Mean Squared Error) of 17.43 on the test data.
           The user can input the number of days for which they want the forecast, and the model will generate an interactive chart
           containing estimates based on historical O3 concentration data.
           Additionally, a table will be displayed with the estimated values for each day.''')

st.subheader('Enter the number of forecast days:')
days = st.number_input('Number of days', min_value=1, value=1, step=1)

# Initialize session state
if 'forecast_ok' not in st.session_state:
    st.session_state['forecast_ok'] = False
    st.session_state['forecast_data'] = None

# Button to predict
if st.button('Predict'):
    try:
        future = model.make_future_dataframe(periods=days, freq='D')
        forecast = model.predict(future)
        if forecast is not None and not forecast.empty:
            st.session_state['forecast_data'] = forecast
            st.session_state['forecast_ok'] = True
        else:
            st.error("The forecast failed or returned empty data.")
            st.session_state['forecast_ok'] = False
    except Exception as e:
        st.error(f"Error generating the forecast: {e}")
        st.session_state['forecast_ok'] = False

# Display the chart and table only if the forecast is valid
if st.session_state['forecast_ok'] and st.session_state['forecast_data'] is not None:
    # Chart
    fig = plot_plotly(model, st.session_state['forecast_data'])
    fig.update_layout({
        'plot_bgcolor': 'rgba(255, 255, 255, 1)',
        'paper_bgcolor': 'rgba(255, 255, 255, 1)',
        'title': {'text': "Ozone Forecast", 'font': {'color': 'black'}},
        'xaxis': {'title': 'Date', 'title_font': {'color': 'black'}, 'tickfont': {'color': 'black'}},
        'yaxis': {'title': 'Ozone Level (O3 μg/m3)', 'title_font': {'color': 'black'}, 'tickfont': {'color': 'black'}}
    })
    st.plotly_chart(fig)

    # Table
    forecast = st.session_state['forecast_data']
    forecast_sheet = forecast[['ds', 'yhat']].tail(days)
    forecast_sheet.columns = ['Date (Day/Month/Year)', 'O3 (ug/m3)']
    forecast_sheet['Date (Day/Month/Year)'] = forecast_sheet['Date (Day/Month/Year)'].dt.strftime('%d-%m-%Y')
    forecast_sheet['O3 (ug/m3)'] = forecast_sheet['O3 (ug/m3)'].round(2)
    forecast_sheet.reset_index(drop=True, inplace=True)
    st.write(f'Table containing ozone (ug/m3) forecasts for the next {days} days:')
    st.dataframe(forecast_sheet, height=300)

    # Download
    csv = forecast_sheet.to_csv(index=False)
    st.download_button(
        label='Download table as CSV',
        data=csv,
        file_name='ozone_forecast.csv',
        mime='text/csv'
    )
else:
    st.warning("Click 'Predict' to generate the forecast.")