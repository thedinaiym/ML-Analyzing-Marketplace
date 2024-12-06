from data_preprocessing import *

def advanced_customer_segmentation(df_encoded):
    """Расширенная сегментация клиентов с визуализацией"""
    st.subheader('🔍 Продвинутая сегментация клиентов')
    
    # Выбор признаков для кластеризации
    segmentation_features = ['Total', 'Quantity', 'Unit price', 'Rating', 'hour']
    X = df_encoded[segmentation_features]
    
    # Масштабирование
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # K-means кластеризация
    n_clusters = st.slider('Количество кластеров', 2, 6, 4)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_encoded['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Визуализация кластеров
    col1, col2 = st.columns(2)
    
    with col1:
        # 3D scatter plot
        fig = px.scatter_3d(
            df_encoded, 
            x='Total', 
            y='Quantity', 
            z='Rating',
            color='Cluster',
            title='3D Кластеризация клиентов'
        )
        st.plotly_chart(fig)
    
    with col2:
        # Radar chart для характеристик кластеров
        cluster_stats = df_encoded.groupby('Cluster')[segmentation_features].mean()
        
        categories = segmentation_features
        
        fig = go.Figure()
        for cluster in range(n_clusters):
            fig.add_trace(go.Scatterpolar(
                r=cluster_stats.loc[cluster],
                theta=categories,
                fill='toself',
                name=f'Кластер {cluster}'
            ))
        
        fig.update_layout(title='Характеристики кластеров')
        st.plotly_chart(fig)
    
    # Таблица характеристик кластеров
    st.dataframe(cluster_stats)
    
    # Дополнительные метрики
    st.metric(
        "Качество кластеризации (Silhouette Score)", 
        f"{silhouette_score(X_scaled, df_encoded['Cluster']):.2f}"
    )
    
    return df_encoded