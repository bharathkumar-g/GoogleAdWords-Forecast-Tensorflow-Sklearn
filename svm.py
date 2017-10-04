import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.svm import SVR
import copy as cp

class DataWrapper:
    def __init__(self,x,y,data_size,batch_size):
        self.data = x
        self.labels = y
        self.data_size = data_size
        self.batch_size = batch_size
        self.next_batch_ind = 0

    def get_next_batch(self):
        batch_x = self.data[self.next_batch_ind:self.next_batch_ind+batch_size]
        batch_y = self.labels[self.next_batch_ind:self.next_batch_ind+batch_size]
        self.next_batch_ind += self.batch_size
        if self.next_batch_ind + self.batch_size > self.data_size:
            self.next_batch_ind = 0
        #print(batch_x,batch_y)
        return batch_x,batch_y

START_YEAR = 2015
DAYS_IN_YEAR = 365
MONTH_DAYS = [31,28,31,30,31,30,31,31,30,31,30,31]

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

def get_euclides_error(predictions, labels,round = False,print_arrays=False):
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

if __name__=='__main__':
    #Read data
    df = pd.read_csv('ad_data.csv')

    # Parsing date
    df['year'] = df.date.apply(get_year)
    #df['month'] = df.date.apply(get_month)
    #df['day'] = df.date.apply(get_day)
    df['day_of_week'] = df.date.apply(get_day_of_week)
    df['total_day_count'] = df.date.apply(get_total_day_count)
    df['working_day'] = df.day_of_week.apply(get_working_day)
    df = df.drop('date', 1)

    # Moving year,month,day columns to front
    df = df[['year', 'day_of_week','working_day', 'total_day_count', 'impressions', 'clicks', 'conversions', 'cost',
             'total_conversion_value', 'average_position', 'reservations', 'price']]

    # Renaming columns names
    df = df.rename(index=str,
                   columns={"average_position": "avg_position", "total_conversion_value": "tot_conversion_val"})
    #Getting clicks & coversions columns as lists
    clicks = df['clicks'].tolist()
    conversions = df['conversions'].tolist()

    #Moving each element up by removing first element. After that adding 0 to the end so that column size matched DF size
    clicks.pop(0)
    clicks.append(0)
    conversions.pop(0)
    conversions.append(0)

    #Creating 2 new columns with values of clicks&conversions of the next day
    df['next_clicks'] = clicks
    df['next_conversions'] = conversions

    #Dropping last row, because we won't learn anything from it. We have already extracted the clicks and conversions.
    df = df[:-1]

    #Specifying previous day data to use as features. Use diff to get derivatives(return difference between current and previous feature)
    #df = get_previous_vals(df,n_features=3,diff=True)

    #Shuffling the data, keep the index
    df = df.sample(frac=1)

    #Dividing the set into input features and output,
    Y = df.ix[:,'next_clicks'].astype(float)
    X = df.drop(['next_clicks','next_conversions'],1)

    #print(X.head(10))

    #Normalizing inputs
    X = (X - X.min() - (X.max() - X.min()) / 2) / ((X.max() - X.min()) / 2)

    train_data_size = 600
    val_data_size = 130
    stddev = 0.01
    epochs = 3000
    batch_size = 32
    steps_in_epoch = 530//10
    learning_rate = 0.001

    #Splitting into train,val,test sets
    X_dataset = X
    Y_dataset = Y
    X_train = np.array(X[:train_data_size])
    Y_train = np.array(Y[:train_data_size])
    X_val = np.array(X[train_data_size:])
    Y_val = np.array(Y[train_data_size:])
    # X_test = np.array(X[train_data_size+val_data_size:])
    # Y_test = np.array(Y[train_data_size+val_data_size:])

    train_data = DataWrapper(X_train,Y_train,train_data_size,batch_size)
    val_data = DataWrapper(X_val, Y_val, val_data_size, batch_size)

    # Defining our SVR
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)

    classifiers = {"types":[svr_lin,svr_rbf,svr_poly],"kernel_names": ["Linear","RBF","Polynomial"]}

    for classifier,kernel_name in zip(classifiers["types"],classifiers["kernel_names"]):
        print("SVM with",kernel_name,"kernel:")
        classifier.fit(train_data.data, train_data.labels)

        # Calculating Train set error
        predictions_train = classifier.predict(train_data.data)
        get_positive_vals = lambda x: x if x >= 0 else 0
        predictions_train = [get_positive_vals(y) for y in predictions_train]
        train_error_float = get_euclides_error(predictions_train, train_data.labels,round=False)
        train_error_int = get_euclides_error(predictions_train, train_data.labels,round=True)
        print("Train Error: Float", train_error_float,", Rounded:",train_error_int)

        # Calcualting Val set error
        predictions_val = classifier.predict(val_data.data)
        get_positive_vals = lambda x: x if x >= 0 else 0
        predictions_val = [get_positive_vals(y) for y in predictions_val]
        val_error_float = get_euclides_error(predictions_val, val_data.labels,round=False)
        val_error_int = get_euclides_error(predictions_val, val_data.labels,round=True)
        print("Val Error: Float", val_error_float,", Rounded:",val_error_int)