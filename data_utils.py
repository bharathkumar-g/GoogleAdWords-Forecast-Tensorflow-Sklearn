#python version: 3.5
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import copy as cp
import pandas as pd
import random

START_YEAR = 2015
DAYS_IN_YEAR = 365
MONTH_DAYS = [31,28,31,30,31,30,31,31,30,31,30,31]

class DataWrapper:
    def __init__(self,x,y,data_size,batch_size):
        self.data = x
        self.labels = y
        self.data_size = data_size
        self.batch_size = batch_size
        self.next_batch_ind = 0

    def get_next_batch(self):
        batch_inds = random.sample(range(self.data_size), self.batch_size)
        batch_x = self.data[batch_inds]
        batch_y = self.labels[batch_inds]
        self.next_batch_ind += self.batch_size
        if self.next_batch_ind + self.batch_size > self.data_size:
            self.next_batch_ind = 0
        return batch_x,batch_y

def plot_corr_mat(df):
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(30, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

    sns.set(font_scale=0.8)

    plt.show()

def get_year(date):
    year= date.split('-')[0]
    return int(year)

def get_month(date):
    month = date.split('-')[1]
    return int(month)

def get_day(date):
    day = date.split('-')[2]
    return int(day)

def get_day_of_week(date):
    total_day_count = get_total_day_count(date)
    day_of_week = np.mod(total_day_count+3,7) #0-Monday,6-Sunday
    return day_of_week

def get_total_day_count(date):
    year,month,day_of_month = date.split('-') #['yyyy', 'mm', 'dd']
    day_count = (int(year) - START_YEAR)*DAYS_IN_YEAR + get_day_of_year(year,month,day_of_month)
    return day_count

def get_day_of_year(year,month,day_of_month):
    day_count = 0
    if year == '2016':
        MONTH_DAYS[1] = 29
    for i in range(int(month)-1):
        day_count += MONTH_DAYS[i]
    return day_count + int(day_of_month) -1

def get_working_day(day_of_week):
    if day_of_week <= 4:
        return 1
    else:
        return 0

def get_previous_vals(df,n_features=5,diff=False):
    prev_features = [df['cost'].tolist(), df['clicks'].tolist(), df['impressions'].tolist(),
                     df['avg_position'].tolist(), df['conversions'].tolist()]

    col_names = ['prev_cost', 'prev_clicks', 'prev_impressions','prev_avg_position', 'prev_conversions']

    #Choosing only first n features
    prev_features = prev_features[:n_features]
    col_names = col_names[:n_features]
    for feature, col_name in zip(prev_features, col_names):
        prev_feature = cp.copy(feature)
        prev_feature.insert(0, 0.0)
        prev_feature.pop(-1)
        if diff:
            diff_arr = np.array(feature)-np.array(prev_feature)
            df[col_name] = diff_arr
        else:
            df[col_name] = prev_feature
    return df

def get_next_values(df_col):
    df_col = df_col.tolist()
    df_col.pop(0)
    df_col.append(0)
    return df_col

def get_positive(x):
    if x >= 0:
        return x
    else:
        return 0

def get_positive_vals(arr):
    return [get_positive(x) for x in arr]

def get_moving_avg(col,n):
    mov_avg_col = cp.copy(col)
    for i in range(mov_avg_col.shape[0]):
        if i < n:
            mov_avg_col[i] = 0.0
        else:
            mov_avg_col[i] = np.mean(np.array(col[i-n:i]))
    return mov_avg_col

def check_distribution(mean_array,global_mean,margin=0.3):
    for mean_val in mean_array:
        if np.abs(mean_val - global_mean) > margin:
            return False
    return True

def get_euclidean_error(predictions, labels,round = False,print_arrays=False):
    if round:
        predictions = np.rint(predictions)
    total_error = 0
    for y_pred, label in zip(predictions, labels):
        total_error = total_error + np.abs(y_pred-label)
    if print_arrays:
        results = np.zeros((len(predictions),2))
        results[:,0] = predictions
        results[:,1] = labels
        print(results)
    return total_error / len(predictions)

def get_mean_rel_err(arr_true,arr_pred):
    return np.mean(np.abs((arr_true-arr_pred)/arr_true))*100

def get_processed_dataframe(df_path,output='clicks',raw = False):
    # Read data
    df = pd.read_csv(df_path)

    # Parsing date
    df['year'] = df.date.apply(get_year)
    df['month'] = df.date.apply(get_month)
    df['day'] = df.date.apply(get_day)
    df['day_of_week'] = df.date.apply(get_day_of_week)
    df['total_day_count'] = df.date.apply(get_total_day_count)
    df['working_day'] = df.day_of_week.apply(get_working_day)
    df = df.drop('date', 1)

    # Moving year,month,day columns to front
    df = df[['year', 'month', 'day', 'impressions', 'day_of_week', 'working_day', 'clicks', 'conversions', 'cost',
             'total_conversion_value', 'average_position', 'reservations', 'price']]

    # Renaming columns names
    df = df.rename(index=str,
                   columns={"average_position": "avg_position", "total_conversion_value": "tot_conversion_val"})

    # Creating 2 new columns with values of clicks&conversions of the next day
    df['next_clicks'] = get_next_values(df['clicks'])
    df['next_conversions'] = get_next_values(df['conversions'])

    # Dropping last row, because we won't learn anything from it. We have already extracted the clicks and conversions.
    df = df[:-1]

    # Adding moving average
    df['mov_avg_short'] = get_moving_avg(df['clicks'], n=6)
    df['mov_avg_long'] = get_moving_avg(df['clicks'], n=30)

    # Specifying previous day data to use as features. Use diff to get derivatives(return difference between current and previous feature)
    # df = get_previous_vals(df,n_features=1,diff=True)

    if raw == True:
        return df

    # Shuffling the data, keep the index
    df = df.sample(frac=1)

    # Dividing the set into input features and output,
    if output == 'clicks':
        Y = df[['next_clicks']].astype(float)
    else:
        Y = df[['next_conversions']].astype(float)

    X = df.drop(['next_clicks', 'next_conversions'], 1)

    # Normalizing inputs
    X = (X - X.min() - (X.max() - X.min()) / 2) / ((X.max() - X.min()) / 2)

    return X,Y