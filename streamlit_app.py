import streamlit as st
import joblib

# Title of the application
st.title("TV Prediction App")

# Header
st.header("Estimate the price of your TV.")

# Subheader
st.subheader("Welcome to our TV Price Estimator! This application allows you to estimate the market value of your television based on various features.")

st.image("TV.jpg", caption="TV", width=600)

st.write("Simply select the brand, resolution, size, operating system, and input the original price and rating. Our model will then provide you with an estimated price for your TV. Once you have entered all the details, click on the 'Predict the price' button to get an estimated market value for your TV. This tool helps you understand the approximate value of your television in the current market, making it easier for you to make informed decisions whether you're buying, selling, or simply curious about your TV's worth.")

top_6_feature_names = joblib.load("top_6_feature_names.joblib") 

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Örnek veri
sample_one = [{
    'Brand': 'TOSHIBA',
    'Resolution': 'Ultra HD LED',
    'Size': '55 inches',
    'Operating System': 'VIDAA 4.0',
    'Rating': 4.3,
    'Original Price': 54990
}]

# Örnek veri DataFrame'e dönüştürme
df_sample = pd.DataFrame(sample_one)

# Marka ve diğer özellikler için seçenekler
brands = ['TOSHIBA', 'Samsung', 'LG', 'Sony', 'Panasonic', 'Philips', 'Sharp', 'Hisense', 'Vizio', 'TCL']
resolutions = ['Ultra HD LED', 'Full HD', 'HD Ready', '4K Ultra HD', '8K Ultra HD']
sizes = ['32 inches', '40 inches', '43 inches', '50 inches', '55 inches', '60 inches', '65 inches', '75 inches', '85 inches']
operating_systems = ['VIDAA 4.0', 'Android TV', 'WebOS', 'Tizen', 'Roku TV']

# Label Encoding
labelEncoder_cols = ['Brand', 'Resolution', 'Size', 'Operating System']
le_dict = {}

for column in labelEncoder_cols:
    le = LabelEncoder()
    le.fit(brands if column == 'Brand' else
           resolutions if column == 'Resolution' else
           sizes if column == 'Size' else
           operating_systems)
    df_sample[column] = le.transform(df_sample[column])
    le_dict[column] = le

# Fiyat tahmini fonksiyonu (örnek)
def predict_price(sample):
    # Burada basit bir formül ile fiyat tahmini yapıyoruz, gerçek bir model kullanabilirsiniz
    base_price = 20000  # Temel fiyat (örnek)
    brand_weight = 1000
    resolution_weight = 2000
    size_weight = 1500
    os_weight = 500
    rating_weight = 3000
    original_price_weight = 0.8

    price = (
        base_price +
        sample['Brand'][0] * brand_weight +
        sample['Resolution'][0] * resolution_weight +
        sample['Size'][0] * size_weight +
        sample['Operating System'][0] * os_weight +
        sample['Rating'][0] * rating_weight +
        sample['Original Price'][0] * original_price_weight
    )
    return round(price, 2)

# Streamlit arayüzü
st.title("Television Price Predict")

# Kullanıcı giriş alanları
brand = st.selectbox('Brand', brands)
resolution = st.selectbox('Resolution', resolutions)
size = st.selectbox('Size', sizes)
os = st.selectbox('Operating Systems', operating_systems)
rating = st.slider('Rating', min_value=0.0, max_value=5.0, step=0.1, value=4.3)
original_price = st.number_input('Original Price', min_value=0, value=54990)

# Kullanıcı girişlerini DataFrame'e dönüştürme
sample_data = {
    'Brand': [brand],
    'Resolution': [resolution],
    'Size': [size],
    'Operating System': [os],
    'Rating': [rating],
    'Original Price': [original_price]
}
sample_df = pd.DataFrame(sample_data)

# Label Encoding uygulama
for column in labelEncoder_cols:
    le = le_dict[column]
    sample_df[column] = le.transform(sample_df[column])

# Fiyat tahmini yapma
predicted_price = predict_price(sample_df)

# Tahmin sonucunu gösterme
if st.button('Predict the price'):
    st.write(f"Estimated price: {predicted_price:.2f} TL")
    st.title(f"{predicted_price:.2f} TL")



