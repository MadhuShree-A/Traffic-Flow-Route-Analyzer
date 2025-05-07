import streamlit as st
import requests
import folium
import pandas as pd
import plotly.express as px
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from streamlit_folium import folium_static

# TomTom API Key (Replace with actual key)
TOMTOM_API_KEY = "GJbicJXmKQQBGI48bMnfDJd1NseDtnw7"

# Set random seed for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

set_seed()

# Get coordinates from place name
def get_coordinates(place):
    url = f"https://api.tomtom.com/search/2/geocode/{place}.json?key={TOMTOM_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "results" in data and len(data["results"]):
            position = data["results"][0]["position"]
            return f"{position['lat']},{position['lon']}"
    st.error(f"Could not find coordinates for {place}")
    return None

def get_traffic(lat, lon):
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={TOMTOM_API_KEY}&point={lat},{lon}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["flowSegmentData"].get("currentSpeed", None)
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching real-time traffic: {e}")
        return None


# Get real-time traffic speed
def get_real_time_traffic(lat, lon):
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={TOMTOM_API_KEY}&point={lat},{lon}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data
    return None
# Generate simulated historical traffic data
def generate_historical_traffic(real_time_speed):
    congestion_levels = [max(10 if 7 <= hour <= 9 or 17 <= hour <= 20 else 5, real_time_speed - random.randint(0, 15)) for hour in range(24)]
    return pd.DataFrame({"Time": [f"{i}:00" for i in range(24)], "Congestion Level": congestion_levels})

# Train LSTM Model
def train_lstm_model(historical_traffic, scaler):
    data = historical_traffic.copy()
    data['Time_Index'] = np.arange(len(data))
    data_scaled = scaler.fit_transform(data[['Congestion Level']])

    X_train, y_train = [], []
    for i in range(len(data_scaled) - 6):
        X_train.append(data_scaled[i:i+6, 0])
        y_train.append(data_scaled[i+6, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(6, 1)),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, verbose=0)

    return model

# Predict Future Traffic
def predict_future_traffic(model, scaler, historical_traffic):
    data_scaled = scaler.transform(historical_traffic[['Congestion Level']].values)
    last_6_hours = data_scaled[-6:].reshape(1, 6, 1)

    predictions = []
    for _ in range(6):
        next_hour_pred = model.predict(last_6_hours, verbose=0)[0][0]
        predictions.append(next_hour_pred)
        last_6_hours = np.roll(last_6_hours, -1)
        last_6_hours[0, -1, 0] = next_hour_pred

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    last_time = int(historical_traffic.iloc[-1]['Time'].split(':')[0])
    future_times = [f"{(last_time + i) % 24}:00" for i in range(1, 7)]
    
    return pd.DataFrame({"Time": future_times, "Predicted Congestion": predictions})

# Get alternative routes
def get_alternative_routes(source, destination):
    url = f"https://api.tomtom.com/routing/1/calculateRoute/{source}:{destination}/json?key={TOMTOM_API_KEY}&traffic=true"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    st.error(f"Error fetching routes: {response.status_code}")
    return None

# Streamlit UI
st.title("Traffic Flow & Route Analyzer")

source_place = st.text_input("Enter Source (City, State)", "Coimbatore, Tamil Nadu")
destination_place = st.text_input("Enter Destination (City, State)", "Trichy, Tamil Nadu")

if st.button("Analyze Traffic & Show Route"):
    source = get_coordinates(source_place)
    destination = get_coordinates(destination_place)

    if source and destination:
        src_lat, src_lon = map(float, source.split(","))
        dest_lat, dest_lon = map(float, destination.split(","))

        source_1 = get_real_time_traffic(src_lat, src_lon)
        destination_1 = get_real_time_traffic(dest_lat, dest_lon)

        if source_1 and destination_1:
            st.subheader("Real-Time Traffic Data For Source")
            
            st.write(f"Road Speed: {source_1['flowSegmentData']['currentSpeed']} km/h")
            st.write(f"Free Flow Speed: {source_1['flowSegmentData']['freeFlowSpeed']} km/h")
            st.write(f"Traffic Delay: {source_1['flowSegmentData']['currentTravelTime']} secs")

            st.subheader("Real-Time Traffic Data For Destination")
            
            st.write(f"Road Speed: {destination_1['flowSegmentData']['currentSpeed']} km/h")
            st.write(f"Free Flow Speed: {destination_1['flowSegmentData']['freeFlowSpeed']} km/h")
            st.write(f"Traffic Delay: {destination_1['flowSegmentData']['currentTravelTime']} secs")
            
            src_speed = get_traffic(src_lat, src_lon)
            dest_speed = get_traffic(dest_lat, dest_lon)

            # Real-time speed bar plot
            real_time_df = pd.DataFrame({"Location": ["Source", "Destination"], "Speed (km/h)": [src_speed, dest_speed]})
            fig_real_time = px.bar(real_time_df, x="Location", y="Speed (km/h)", title="Real-Time Traffic Speed", color="Location", text="Speed (km/h)", barmode="group")
            st.plotly_chart(fig_real_time)
            
            scaler = MinMaxScaler()
            historical_traffic_source = generate_historical_traffic(src_speed)
            historical_traffic_dest = generate_historical_traffic(dest_speed)
            
            # Plot historical traffic data
            fig_hist_src = px.line(historical_traffic_source, x="Time", y="Congestion Level", title="Historical Traffic (Source)", markers=True)
            st.plotly_chart(fig_hist_src)
            fig_hist_dest = px.line(historical_traffic_dest, x="Time", y="Congestion Level", title="Historical Traffic (Destination)", markers=True)
            st.plotly_chart(fig_hist_dest)
            
            model_source = train_lstm_model(historical_traffic_source, scaler)
            model_dest = train_lstm_model(historical_traffic_dest, scaler)
            future_traffic_source = predict_future_traffic(model_source, scaler, historical_traffic_source)
            future_traffic_dest = predict_future_traffic(model_dest, scaler, historical_traffic_dest)
            
            fig_src = px.line(future_traffic_source, x="Time", y="Predicted Congestion", title="Future Traffic (Source)", markers=True)
            st.plotly_chart(fig_src)
            fig_dest = px.line(future_traffic_dest, x="Time", y="Predicted Congestion", title="Future Traffic (Destination)", markers=True)
            st.plotly_chart(fig_dest)

        route_data = get_alternative_routes(source, destination)
        if route_data and "routes" in route_data:
                m = folium.Map(location=[src_lat, src_lon], zoom_start=10)
                folium.Marker([src_lat, src_lon], popup="Source", icon=folium.Icon(color="blue")).add_to(m)
                folium.Marker([dest_lat, dest_lon], popup="Destination", icon=folium.Icon(color="red")).add_to(m)
                for route in route_data["routes"]:
                    points = [(p["latitude"], p["longitude"]) for p in route["legs"][0]["points"]]
                    folium.PolyLine(points, color="blue", weight=5, opacity=0.7).add_to(m)
                folium_static(m)
