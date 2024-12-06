from data_preprocessing import *
from segmentation import *
from sales_forecast import *
from visual import *
            
def main():
    st.set_page_config(page_title='ML Анализ Супермаркета', layout='wide')
    st.title('🛒 ML-Анализ Данных Супермаркета')
    
    uploaded_file = st.file_uploader("Загрузите CSV файл", type=['csv'])
    
    if uploaded_file is not None:
        df, df_encoded = load_data(uploaded_file)
        
        tabs = st.tabs(['📊 Описательная статистика', 
                        '🔬 Сегментация клиентов', 
                        '💰 Прогнозирование продаж', 
                        '📈 Визуальный анализ'])
        
        with tabs[0]:
            st.dataframe(df.describe())
        
        with tabs[1]:
            advanced_customer_segmentation(df_encoded)
        
        with tabs[2]:
            sales_forecast(df_encoded)
        
        with tabs[3]:
            advanced_visualizations(df)

if __name__ == '__main__':
    main()