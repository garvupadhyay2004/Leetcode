import streamlit as st
import base64
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Set background
def set_background(image_file_path):
    with open(image_file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    background_style = f'''
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .custom-box {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 0 10px rgba(0,0,0,0.3);
    }}
    .custom-title {{
        font-size: 36px;
        font-weight: bold;
        color: black;
        text-align: center;
        margin-bottom: 10px;
    }}
    .custom-label {{
        color: black;
        font-size: 20px;
        font-weight: bold;
    }}
    .stButton > button {{
        color: white;
        background-color: #8B4513;
        font-weight: bold;
        font-size: 18px;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
    }}
    .stButton > button:hover {{
        background-color: #A0522D;
        cursor: pointer;
    }}
    </style>
    '''
    st.markdown(background_style, unsafe_allow_html=True)

set_background("C:\\Users\\rsent\\AppData\\Local\\Temp\\7965ae32-5cf7-4665-b7eb-2794e33d67ae_DataScience_Project_by_Gaurav_and_Ayush_Sharma[1].zip.7ae\iX.jpg")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\rsent\\AppData\\Local\\Temp\\03ce4df7-03c5-48f2-8c32-e6f8f15b9e6f_DataScience_Project_by_Gaurav_and_Ayush_Sharma[1].zip.e6f\Automobile_dataset.csv")
    selected_features = ['Engine HP', 'Engine Cylinders', 'highway MPG', 'city mpg', 'MSRP']
    df = df.dropna(subset=selected_features)
    return df, selected_features

df, selected_features = load_data()
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[selected_features])
model = NearestNeighbors(n_neighbors=5, metric='euclidean')
model.fit(data_scaled)

# Tabs for dashboard
tab1, tab2, tab3 = st.tabs(["🏠 Overview", "🔍 More Info", "👨‍💻 Developer"])

with tab1:
    st.markdown('<div class="custom-box"><div class="custom-title">🚗 Car Recommendation System</div><p class="custom-label">Get car suggestions based on your preferences in Indian Rupees.</p></div>', unsafe_allow_html=True)

    # Input box function
    def input_box(label, widget):
        st.markdown(f'<div style="background-color: #ffffffdd; padding: 10px; border-radius: 10px; margin-bottom: 10px;"><span style="color: black; font-weight: bold; font-size: 18px;">{label}</span></div>', unsafe_allow_html=True)
        return widget

    st.markdown('<div class="custom-box">', unsafe_allow_html=True)

    hp = input_box("Horsepower (Engine HP)", st.slider("Horsepower", 50, 1000, 200, label_visibility="collapsed"))
    cylinders = input_box("Number of Cylinders", st.slider("Cylinders", 2, 16, 4, label_visibility="collapsed"))
    highway_mpg = input_box("Highway MPG", st.slider("Highway MPG", 5, 100, 30, label_visibility="collapsed"))
    city_mpg = input_box("City MPG", st.slider("City MPG", 5, 100, 25, label_visibility="collapsed"))
    budget_inr = input_box("Budget (in INR)", st.number_input("Budget INR", min_value=50000, max_value=15000000, value=2500000, step=50000, label_visibility="collapsed"))

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="custom-box">', unsafe_allow_html=True)
    if st.button("Find Cars"):
        usd_inr = 83  # Static conversion rate
        budget_usd = budget_inr / usd_inr
        user_input = np.array([[hp, cylinders, highway_mpg, city_mpg, budget_usd]])
        user_input_scaled = scaler.transform(user_input)
        distances, indices = model.kneighbors(user_input_scaled)
        recommendations = df.iloc[indices[0]].copy()
        recommendations['Similarity Score'] = distances[0]
        recommendations = recommendations[recommendations['MSRP'] <= budget_usd]

        if recommendations.empty:
            st.warning("No cars found within the specified budget. Try increasing your budget.")
        else:
            recommendations = recommendations.sort_values("Similarity Score")
            st.markdown('''
                <div class="custom-box">
                    <div class="custom-title">Top Recommended Cars</div>
            ''', unsafe_allow_html=True)
            recommendations['MSRP (INR)'] = (recommendations['MSRP'] * usd_inr).astype(int)
            st.dataframe(recommendations[['Make', 'Model', 'Year', 'Engine Fuel Type', 'Engine HP', 'MSRP (INR)', 'Similarity Score']].reset_index(drop=True))
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="custom-box"><div class="custom-title">🔍 More Information</div>', unsafe_allow_html=True)
    st.write("""
    This Car Recommendation System uses a **K-Nearest Neighbors (KNN)** model to suggest cars based on your preferences like horsepower, fuel efficiency, and budget (in INR). 
    It standardizes car features and compares your preferences with similar cars in the dataset.
    """)

with tab3:
    st.markdown('<div class="custom-box"><div class="custom-title">👨‍💻 Developer Info</div>', unsafe_allow_html=True)
    st.write("""
    **Developer**: Your Name  
    **Project Title**: Car Recommendation System using KNN  
    **Tools Used**: Python, Streamlit, scikit-learn, Pandas  
    **Year**: 2025  
    """)




