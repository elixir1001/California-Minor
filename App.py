# Importing necessary libraries
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pydeck as pdk

# Load the model and preprocessing pipeline
model = joblib.load('Final Model.pkl')
preprocessing_pipeline = joblib.load('Final_housing_pipeline.pkl')

# Load the housing dataset
housing_main = pd.read_csv('housing.csv')

# Set page configuration
st.set_page_config(
    page_title="California Housing Prediction",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.link_button(
    "GitHub",
    "https://github.com/elixir1001/California-Housing-Prices",
    use_container_width=True
)
st.sidebar.link_button(
    "Research Paper",
    "https://www.spatial-statistics.com/pace_manuscripts/spletters_ms_dir/statistics_prob_lets/html/ms_sp_lets1.html",
    use_container_width=True
)

# Sidebar for user input
st.sidebar.header("Specify input parameters")

# Define function to get user input via sliders
def features_from_user():
    longitude = st.sidebar.slider('Longitude', 
                                  housing_main.longitude.min(), 
                                  housing_main.longitude.max(), 
                                  housing_main.longitude.mean())
    latitude = st.sidebar.slider('Latitude', 
                                 housing_main.latitude.min(), 
                                 housing_main.latitude.max(), 
                                 housing_main.latitude.mean())
    housing_median_age = st.sidebar.slider('Median Age of House', 
                                           housing_main.housing_median_age.min(), 
                                           housing_main.housing_median_age.max(), 
                                           housing_main.housing_median_age.mean())
    total_rooms = st.sidebar.slider('Total Rooms in Block', 
                                    housing_main.total_rooms.min(), 
                                    housing_main.total_rooms.max(), 
                                    housing_main.total_rooms.mean())
    total_bedrooms = st.sidebar.slider('Total Bedrooms in Block', 
                                       housing_main.total_bedrooms.min(), 
                                       housing_main.total_bedrooms.max(), 
                                       housing_main.total_bedrooms.mean())
    population = st.sidebar.slider('Population', 
                                   housing_main.population.min(), 
                                   housing_main.population.max(), 
                                   housing_main.population.mean())
    households = st.sidebar.slider('Number of Houses in block', 
                                   housing_main.households.min(), 
                                   housing_main.households.max(), 
                                   housing_main.households.mean())
    median_income = st.sidebar.slider('Median Income', 
                                      housing_main.median_income.min(), 
                                      housing_main.median_income.max(), 
                                      housing_main.median_income.mean())
    ocean_proximity = st.sidebar.selectbox("Select Ocean Proximity", 
                                           pd.unique(housing_main["ocean_proximity"]))
    
    data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'ocean_proximity': ocean_proximity
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
housing = features_from_user()

# Prepare the input data for prediction
housing_prepared = preprocessing_pipeline.transform(housing)

# Predict housing price
housing_predictions = model.predict(housing_prepared)

# Create Streamlit app with tabs
import streamlit as st

st.markdown(
    """
    <div style='background: linear-gradient(to right,  #CC3B3B ,#ff4a4a); padding: 1px; border-radius: 0px; text-align: center;'>
        <h1 style='color: white; font-family: "Helvetica", sans-serif; font-size: 26px;'>California Housing Price Project</h1>
    </div>
    """,
    unsafe_allow_html=True
)


st.write('---')
# Create tabs
selected_tab = st.sidebar.radio("Select Tab:", ("Predicted Price", "Data Analysis"))

# Display content based on selected tab
if selected_tab == "Predicted Price":

    st.title("Estimated Housing Price:- ")
    estimated_price = housing_predictions[0]
    usd_to_inr_exchange_rate = 75.0
    price_in_inr = estimated_price * usd_to_inr_exchange_rate
    formatted_price = f"<h1 style='color: #E74C3C; text-align: left;'>₹{price_in_inr:,.2f}</h1>"
    st.markdown(formatted_price, unsafe_allow_html=True)
    st.write('---')
    st.write(housing.head())
    new_df = housing[['longitude', 'latitude']]
    st.title("Location of the house:- ")
    st.map(new_df)
 
elif selected_tab == "Data Analysis":
    st.markdown(
    """
    <div style='background: linear-gradient(to right, #CC3B3B ,#ff4a4a); padding: 1px; border-radius: 0px; text-align: center;'>
        <h1 style='color: white; font-family: "Helvetica", sans-serif; font-size: 16px;'>Data Analysis on Actual Housing Data</h1>
    </div>
    """,
    unsafe_allow_html=True
    )
    st.write('---')

    #st.write(housing_main.head())
    #feature = st.selectbox("Select a feature to visualize and compare with price", housing_main.columns)  
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        selected_visualization = st.radio("Select Visualization:", ("Histogram of Age of House", 
                                                                    "Histogram of Total Rooms",
                                                                    "Histogram of Population",
                                                                    "Histogram of Income",
                                                                    "Map of Data"))
        @st.cache_data
        def convert_df(housing_main):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return housing_main.to_csv().encode('utf-8')
        csv = convert_df(housing_main)
        st.download_button(
            label="Download the CSV",
            data=csv,
            file_name="housing.csv",
            mime='text/csv',
            use_container_width=True
        )
    with col2:
        if selected_visualization == "Histogram of Age of House":
            sns.set(style='whitegrid')
            plt.figure(figsize=(20, 12))
            sns.histplot(data=housing_main, x='housing_median_age', kde=True, color='skyblue')
            plt.title('Distribution of Housing Median Age', fontsize=16)
            plt.xlabel('Housing Median Age', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            sns.despine(left=True)
            sns.set_palette('bright')
            st.pyplot(plt)
        if selected_visualization == "Histogram of Total Rooms":
            sns.set(style='whitegrid')
            plt.figure(figsize=(20, 12))
            sns.histplot(data=housing_main, x='total_rooms', kde=True, color='skyblue')
            plt.title('Distribution of Housing Total Rooms', fontsize=16)
            plt.xlabel('Total Rooms', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            sns.despine(left=True)
            sns.set_palette('pastel')
            st.pyplot(plt)
        if selected_visualization == "Histogram of Population":
            sns.set(style='whitegrid')
            plt.figure(figsize=(20, 12))
            sns.histplot(data=housing_main, x='population', kde=True, color='skyblue')
            plt.title('Distribution of Population', fontsize=16)
            plt.xlabel('Population', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            sns.despine(left=True)
            sns.set_palette('pastel')
            st.pyplot(plt)
        if selected_visualization == "Histogram of Income":
            sns.set(style='whitegrid')
            plt.figure(figsize=(20, 12))
            sns.histplot(data=housing_main, x='median_income', kde=True, color='skyblue')
            plt.title('Distribution of Median Income', fontsize=16)
            plt.xlabel('Income', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            sns.despine(left=True)
            sns.set_palette('pastel')
            st.pyplot(plt)
        if selected_visualization == "Map of Data":
            new_df = housing_main[['longitude', 'latitude']]
            st.map(new_df)
