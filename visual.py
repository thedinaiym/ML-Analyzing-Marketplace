from data_preprocessing import *
from segmentation import *
from sales_forecast import *

def advanced_visualizations(df):
    """Расширенный анализ визуализаций"""
    st.subheader('📈 Расширенный анализ визуализаций')
    
    # Создаем несколько вкладок для разных типов визуализаций
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        'Распределение продаж', 
        'Категориальный анализ', 
        'Гендерный анализ', 
        'Временной анализ', 
        'Корреляционный анализ',
        'Интерактивные графики'
    ])
    
    with tab1:
        # 1. Распределение общей выручки
        col1, col2 = st.columns(2)
        
        with col1:
            # Гистограмма распределения выручки
            fig, ax = plt.subplots(figsize=(10, 6))
            df['Total'].hist(bins=30, edgecolor='black', ax=ax)
            ax.set_title('Распределение общей выручки')
            ax.set_xlabel('Выручка')
            ax.set_ylabel('Частота')
            st.pyplot(fig)
        
        with col2:
            # Ящик с усами для выручки
            fig, ax = plt.subplots(figsize=(10, 6))
            df.boxplot(column=['Total'], ax=ax)
            ax.set_title('Статистика выручки')
            st.pyplot(fig)
    
    with tab2:
        # 2. Анализ категорий продуктов
        col1, col2 = st.columns(2)
        
        with col1:
            # Круговая диаграмма продаж по категориям
            fig, ax = plt.subplots(figsize=(10, 6))
            category_sales = df.groupby('Product line')['Total'].sum()
            ax.pie(category_sales, labels=category_sales.index, autopct='%1.1f%%')
            ax.set_title('Доля продаж по категориям')
            st.pyplot(fig)
        
        with col2:
            # Средний рейтинг по категориям
            fig, ax = plt.subplots(figsize=(10, 6))
            category_rating = df.groupby('Product line')['Rating'].mean()
            category_rating.plot(kind='bar', ax=ax)
            ax.set_title('Средний рейтинг по категориям')
            ax.set_xticklabels(category_rating.index, rotation=45, ha='right')
            st.pyplot(fig)
    
    with tab3:
        # 3. Гендерный анализ
        col1, col2 = st.columns(2)
        
        with col1:
            # Продажи по полу
            fig, ax = plt.subplots(figsize=(10, 6))
            gender_sales = df.groupby('Gender')['Total'].sum()
            ax.pie(gender_sales, labels=gender_sales.index, autopct='%1.1f%%')
            ax.set_title('Доля продаж по полу')
            st.pyplot(fig)
        
        with col2:
            # Средний чек по полу
            fig, ax = plt.subplots(figsize=(10, 6))
            gender_avg_check = df.groupby('Gender')['Total'].mean()
            gender_avg_check.plot(kind='bar', ax=ax)
            ax.set_title('Средний чек по полу')
            ax.set_xticklabels(gender_avg_check.index, rotation=45)
            st.pyplot(fig)
    
    with tab4:
        # 4. Временной анализ
        col1, col2 = st.columns(2)
        
        with col1:
            # Продажи по месяцам
            fig, ax = plt.subplots(figsize=(10, 6))
            monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Total'].sum()
            monthly_sales.plot(kind='bar', ax=ax)
            ax.set_title('Продажи по месяцам')
            ax.set_xticklabels(monthly_sales.index, rotation=45, ha='right')
            st.pyplot(fig)
        
        with col2:
            # Распределение продаж по часам
            fig, ax = plt.subplots(figsize=(10, 6))
            hourly_sales = df.groupby('hour')['Total'].sum()
            hourly_sales.plot(kind='line', marker='o', ax=ax)
            ax.set_title('Продажи по часам')
            ax.set_xlabel('Час дня')
            ax.set_ylabel('Общая выручка')
            st.pyplot(fig)
    
    with tab5:
        # 5. Корреляционный анализ
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation_columns = ['Unit price', 'Quantity', 'Total', 'Rating', 'Tax 5%', 'hour']
        correlation_matrix = df[correlation_columns].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Корреляционная матрица')
        st.pyplot(fig)
    
    with tab6:
        # 6. Интерактивные графики с Plotly
        st.subheader('Интерактивная визуализация')
        
        # Выбор типа графика
        plot_type = st.selectbox('Выберите тип графика', [
            'Scatterplot продаж',
            'Распределение по категориям',
            'Продажи по дням недели'
        ])
        
        if plot_type == 'Scatter plot продаж':
            fig = px.scatter(
                df, 
                x='Quantity', 
                y='Total', 
                color='Product line', 
                title='Зависимость выручки от количества товаров'
            )
            st.plotly_chart(fig)
        
        elif plot_type == 'Распределение по категориям':
            fig = px.box(
                df, 
                x='Product line', 
                y='Total', 
                title='Распределение выручки по товарным категориям'
            )
            st.plotly_chart(fig)
        
        else:
            day_sales = df.groupby('day_of_week')['Total'].sum()
            fig = px.bar(
                x=day_sales.index, 
                y=day_sales.values, 
                title='Продажи по дням недели'
            )
            st.plotly_chart(fig)