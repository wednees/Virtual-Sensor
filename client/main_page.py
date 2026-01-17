import time
from PIL import Image
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from utils import data_preparation

def main_component(model):
    is_forecasting_data = True
    st.title('Glowbyte')
    st.write('Необходимо загрузить данные для прогноза')    

    file_for_forecast = st.file_uploader("Выберите файл", type=["txt", "csv", "xlsx"])
    if (file_for_forecast is not None):
        st.write("Имя файла: ", file_for_forecast.name)
        st.write("Размер файла: ", file_for_forecast.size, "bytes")

        # Display the contents of the file if it's a text file
        if file_for_forecast.type == 'text/plain':
            st.write("File Contents:")
            st.write(file_for_forecast.read())
        # Display the contents of the file if it's a csv or xlsx file
        elif file_for_forecast.type in ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']:
            forecast_data = pd.read_csv(file_for_forecast) if file_for_forecast.type in ['text/csv'] else pd.read_excel(file_for_forecast)

            st.write(forecast_data)
            forecast_data_copy = forecast_data.copy()
            forecast_data['datetime'] = pd.to_datetime(forecast_data['datetime'])
            datetime = forecast_data['datetime']
            forecast_data['year'] = forecast_data['datetime'].dt.year
            forecast_data['month'] = forecast_data['datetime'].dt.month
            forecast_data['day'] = forecast_data['datetime'].dt.day
            forecast_data['hour'] = forecast_data['datetime'].dt.hour
            forecast_data['minute'] = forecast_data['datetime'].dt.minute
            forecast_data.drop('datetime', axis=1, inplace=True)

            title = st.empty()

            title.markdown('Прогнозирование данных...')
            predictions = model.predict(forecast_data)
            title.markdown('')
            target = pd.Series(predictions, name='target')
            tr = pd.concat([target, datetime], axis=1)

            plt.figure(figsize=(10, 5))
            plt.plot(tr['datetime'], tr['target'])
            plt.xlabel('Datetime')
            plt.ylabel('Target')
            plt.title('Target Over Time')
            st.pyplot(plt)

            is_forecasting_data = False

    if (not is_forecasting_data):
        true_file = st.file_uploader("Выберите файл для сравнения данных", type=["txt", "csv", "xlsx"])

        if (true_file is not None):

            if (true_file.type == 'text/plain'):
                st.write("File Contents:")
                st.write(true_file.read())
            # Display the contents of the file if it's a csv or xlsx file
            elif (true_file.type in ['text/csv', 'application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']):
                true_data = pd.read_csv(true_file) if true_file.type in ['text/csv'] else pd.read_excel(true_file)

            preparation_test_X, preparation_test_y, datetime = data_preparation(forecast_data_copy, true_data)

            test_predictions = model.predict(preparation_test_X)
            st.write(preparation_test_y.shape)
            st.write(len(test_predictions))

            mae = mean_absolute_error(preparation_test_y['target'], test_predictions)

            st.write(f'MAE: {mae}')

            combined_df = pd.DataFrame({
                'datetime': datetime,
                'pred_target': test_predictions,
                'true_target': preparation_test_y['target']
            })

            combined_df['datetime'] = pd.to_datetime(combined_df['datetime'])

            st.write(combined_df)
            plt.figure(figsize=(12, 6))
            plt.plot(combined_df['datetime'], combined_df['pred_target'], label='test_predictions', color='red')
            plt.plot(combined_df['datetime'], combined_df['true_target'], label='preparation_test_y', color='blue')
            plt.legend()
            plt.savefig("mygraph.png")
            plt.xlabel('Datetime')
            plt.ylabel('Values')

            image = Image.open('mygraph.png')
            # Отображаем изображение в Steamlit
            st.image(image, caption='Сравнение графиков', use_column_width=True)