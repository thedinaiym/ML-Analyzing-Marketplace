import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score
import plotly.express as px
import plotly.graph_objects as go

def load_data(uploaded_file):
    """Расширенная загрузка и обработка данных"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        
        # Подробное описание колонок
        st.sidebar.markdown("### Структура данных")
        for column in df.columns:
            st.sidebar.text(f"{column}: {df[column].dtype}")
        
        # Преобразование даты и времени с более гибким подходом
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Обработка времени с более гибким форматом
        try:
            df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S').dt.time
        except ValueError:
            # Если не удалось распознать с точным форматом, используем более общий подход
            df['Time'] = pd.to_datetime(df['Time'], format='mixed').dt.time
        
        # Извлечение дополнительных признаков
        df['hour'] = df['Time'].apply(lambda x: x.hour)
        df['month'] = df['Date'].dt.month
        df['day_of_week'] = df['Date'].dt.day_name()
        
        # Кодирование категориальных признаков
        categorical_columns = ['Branch', 'City', 'Customer type', 'Gender', 'Product line', 'Payment']
        df_encoded = pd.get_dummies(df, columns=categorical_columns)
        
        return df, df_encoded
    return None, None
