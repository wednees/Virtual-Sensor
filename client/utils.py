import pandas as pd
from datetime import timedelta
def data_preparation(X_test, y_test):
    X_test['datetime'] = pd.to_datetime(X_test['datetime'])
    y_test['Дата'] = pd.to_datetime(y_test['Дата'])
    X_test = X_test.drop_duplicates(subset=['datetime'])
    X_test = X_test.set_index('datetime').resample('2h').min().reset_index()
    X_test = X_test.interpolate(method='linear', limit_direction='forward')
    end_time = X_test['datetime'].iloc[-1]
    end_time_1 = y_test['Дата'].iloc[-1]
    x_start_time = X_test['datetime'].iloc[0]
    cur_time = y_test['Дата'].iloc[0]
    
    if (end_time < end_time_1):
        end = end_time
    else:
        end = end_time_1
    
    X_test = X_test[X_test['datetime'] > cur_time].reset_index(drop=True)
    X_test = X_test[X_test['datetime'] <= end].reset_index(drop=True)
    
    y_test = y_test.set_index('Дата').resample('2h').min()
    y_test['interpolated'] = y_test['target'].isna()
    y_test = y_test.interpolate(method='linear').reset_index()
    y_test.rename(columns={'Дата': 'datetime'}, inplace=True)
    y_test = y_test[y_test['datetime'] < end].iloc[:-1]
    
    X_test = X_test.merge(y_test[['datetime', 'interpolated']], on='datetime', how='left')
     
    X_test['year'] = X_test['datetime'].dt.year
    X_test['month'] = X_test['datetime'].dt.month
    X_test['day'] = X_test['datetime'].dt.day
    X_test['hour'] = X_test['datetime'].dt.hour
    X_test['minute'] = X_test['datetime'].dt.minute

    
    
   
    X_test = X_test[X_test['interpolated'] != True]
    X_test.drop('interpolated', axis=1, inplace=True)
    y_test = y_test[y_test['interpolated'] != True]
    y_test.drop('interpolated', axis=1, inplace=True)

    datetime = y_test['datetime']
    X_test.drop('datetime', axis=1, inplace=True)
    y_test.drop('datetime', axis=1, inplace=True)
    
    return X_test, y_test, datetime