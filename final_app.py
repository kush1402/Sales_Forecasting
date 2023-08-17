import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import datetime
import tensorflow
# from keras.layers.convolutional import Conv1D , MaxPooling1D
from keras.utils import plot_model
from keras.models import Sequential , Model
from keras.layers import *
from keras import optimizers
#Ignore Harmless warnings
import warnings
warnings.filterwarnings('ignore')


dataset = pd.read_csv('train.csv' , index_col='date' , parse_dates=True)


with st.sidebar:
    with st.form(key='my_form'):
        st.title('Demand Forecasting')
        store_number = st.number_input('Select Store Number:' , value =0 , min_value = 0 , max_value = 10)
        item_number = st.number_input('Select Item Number:' , value =0 , min_value = 0 , max_value = 50)
        today_year = 2018
        jan_1 = datetime.date(today_year, 1, 1)
        dec_31 = datetime.date(today_year, 12, 31)
        end_date = st.date_input("Forecast Till Date",datetime.date(today_year, 1, 1), jan_1, dec_31,format="YYYY-MM-DD")
        submitted = st.form_submit_button('Forecast')

    
if submitted :
    dataset_series = dataset.loc[(dataset["item"] == item_number) & (dataset["store"] == store_number)]
    dataset_series = dataset_series.drop(['store' , 'item'] , axis =1)

    past_data = 7
    for i in range (past_data):
        new_column_name_1 = f'Sales_{(i+1)*7}'
        dataset_series.loc[: , new_column_name_1] = dataset_series['sales'].shift(+(i+1)*7)

    dataset_series = dataset_series.dropna()

    X = dataset_series.drop(['sales'] , axis =1)
    Y = dataset_series['sales']

    X_train , X_test , Y_train , Y_test = X[:-182], X[-182:], Y[:-182], Y[-182:]

    X_train_series = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_series = X_test.values.reshape((X_test.shape[0], X_test.shape[1], 1))

    
    epochs = 10
    lr = 0.0003
    adam = optimizers.Adam(lr)

    model_dense = Sequential()
    model_dense.add(LSTM(64 , activation ='relu' , input_shape=(X_train_series.shape[1], X_train_series.shape[2])))
    model_dense.add(Dense(1000, activation='relu', input_dim=X_train.shape[1]))
    model_dense.add(Dense(100, activation = 'relu'))
    model_dense.add(Dense(1))
    model_dense.compile(loss='mse', optimizer=adam , metrics = ['mean_squared_error'])
    model_dense.summary()

    model_history = model_dense.fit(X , Y , epochs=epochs, verbose=2)

    new_data = dataset_series.iloc[-49:,0:1]
    predict_dataset = pd.DataFrame()

    past_data = 7
    for i in range (past_data):
        new_column_name_1 = f'Sales_{(i+1)*7}'
        predict_dataset.loc[: , new_column_name_1] = new_data['sales'].shift(+(i+1)*7)

    new_row = {'sales' : 0}
    last_date = new_data.index[-1]+ timedelta(days=1)
    new_data = pd.concat([new_data, pd.DataFrame([new_row],index =[last_date])], ignore_index=False)

    new_data_2= pd.DataFrame()
    for i in range (past_data):
        new_column_name_1 = f'Sales_{(i+1)*7}'
        new_data_2.loc[: , new_column_name_1] = new_data['sales'].shift(+(i+1)*7)

    new_data_2.dropna()

    def get_data(new_data):
        past_data = 7
        new_data_2= pd.DataFrame()
        for i in range (past_data):
            new_column_name_1 = f'Sales_{(i+1)*7}'
            new_data_2.loc[: , new_column_name_1] = new_data['sales'].shift(+(i+1)*7)
            # if(i!=7):
            #     new_column_name_2 = f'Sales_{i+1}'
            #     new_data_2.loc[: , new_column_name_2] = new_data['sales'].shift(+(i+1))
       
        new_data_2 = new_data_2.dropna()
        return new_data_2[-1:]

    t = dataset_series.index[-1]
    days = end_date - t.date()
    days_int = days.days


    for i in range(days_int):
        predict = model_dense.predict(new_data_2[-1:])
        new_row = {'sales' : 0}
        index = new_data.index[-1]
        new_data.loc[index] = predict
        index = new_data.index[-1]+ timedelta(days=1)
        new_data = pd.concat([new_data, pd.DataFrame([new_row],index =[index])], ignore_index=False)
        new_row_2 = get_data(new_data)
        new_data_2 = pd.concat([new_data_2, new_row_2], axis=0)

    new_data_3 = new_data
    new_data_3.rename(columns = {'sales':'Predicted Sales'}, inplace = True)
    new_data_3['Predicted Sales'] = new_data_3['Predicted Sales'].apply(np.ceil)
    print(new_data_3)

    fig, ax = plt.subplots()
    plt.rcParams['figure.figsize']=(10, 6)
    plt.plot(dataset_series['sales'][1412:], label='Original Data', color='red')
    plt.plot(new_data[30:139], label='Model Predictions', color='blue')
    plt.ylabel('sales')
    plt.title('LSTM Model for Time Series Data')
    plt.legend()
    plt.show()
    # ax.plot(df['Date'] , df['Sales'] , marker = 'o')
    # ax.set_xlabel('Date')
    # ax.set_ylabel('Sales')
    # ax.set_title('Demand forecast')

    st.pyplot(fig)

    st.write(new_data_3[49:140],unsafe_allow_html=False)
    
