from data_preprocessing import *
from segmentation import *

def sales_forecast(df_encoded):
    """Прогнозирование продаж с расширенной визуализацией"""
    st.subheader('💹 Прогнозирование продаж')
    
    # Выбор признаков
    features = [
        'Unit price', 'Quantity', 'hour', 'month', 
        'Branch_A', 'Branch_B', 'Customer type_Member', 
        'Gender_Female'
    ]
    
    X = df_encoded[features]
    y = df_encoded['Total']
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Модель случайного леса
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    
    # Предсказание
    y_pred = rf_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    
    # Визуализация важности признаков
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    fig = px.bar(
        feature_importance, 
        x='feature', 
        y='importance', 
        title='Важность признаков для прогноза продаж'
    )
    st.plotly_chart(fig)
    
    # Сравнение предсказанных и реальных значений
    comparison_df = pd.DataFrame({
        'Реальные значения': y_test,
        'Предсказанные значения': y_pred
    })
    
    fig = px.scatter(
        comparison_df, 
        title='Сравнение реальных и предсказанных значений продаж'
    )
    st.plotly_chart(fig)
    
    st.metric("Средняя квадратичная ошибка", f"{mse:.2f}")