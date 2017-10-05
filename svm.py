import pandas as pd
from sklearn.svm import SVR
import random
from data_utils import *

train_data_size = 600
val_data_size = 130
total_data_size = train_data_size + val_data_size
stddev = 0.01
epochs = 3000
batch_size = 32
steps_in_epoch = 530 // 10
learning_rate = 0.001

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
if __name__=='__main__':
    #Read data
    df = pd.read_csv('ad_data.csv')

    # Parsing date
    df['year'] = df.date.apply(get_year)
    df['month'] = df.date.apply(get_month)
    df['day'] = df.date.apply(get_day)
    df['day_of_week'] = df.date.apply(get_day_of_week)
    df['total_day_count'] = df.date.apply(get_total_day_count)
    df['working_day'] = df.day_of_week.apply(get_working_day)
    df = df.drop('date', 1)

    # Moving year,month,day columns to front
    # df = df[['year', 'day_of_week','working_day', 'total_day_count', 'impressions', 'clicks', 'conversions', 'cost',
    #          'total_conversion_value', 'average_position', 'reservations', 'price']]

    df = df[['year','month', 'day','impressions','day_of_week','working_day', 'clicks', 'conversions', 'cost',
             'total_conversion_value', 'average_position', 'reservations', 'price']]

    # Renaming columns names
    df = df.rename(index=str,
                   columns={"average_position": "avg_position", "total_conversion_value": "tot_conversion_val"})

    #Creating 2 new columns with values of clicks&conversions of the next day
    df['next_clicks'] = get_next_values(df['clicks'])
    df['next_conversions'] = get_next_values(df['conversions'])


    #Dropping last row, because we won't learn anything from it. We have already extracted the clicks and conversions.
    df = df[:-1]

    # Adding moving average
    df['mov_avg_short'] = get_moving_avg(df['clicks'], n=6)
    df['mov_avg_long'] = get_moving_avg(df['clicks'], n=30)
    #Specifying previous day data to use as features. Use diff to get derivatives(return difference between current and previous feature)
    #df = get_previous_vals(df,n_features=1,diff=True)

    #Shuffling the data, keep the index
    df = df.sample(frac=1)

    #Dividing the set into input features and output,
    Y = df[['next_clicks']].astype(float)
    X = df.drop(['next_clicks','next_conversions'],1)

    #Normalizing inputs
    X = (X - X.min() - (X.max() - X.min()) / 2) / ((X.max() - X.min()) / 2)

    #Splitting into train,val,test sets
    X = np.array(X)
    Y = np.array(Y)

    #Placeholders for average scores
    train_error_arr = np.zeros(3)
    val_error_arr = np.zeros(3)
    best_val_error = 1000
    best_error_info = {"kernel_name":"","train_error":0,"val_error:":0}
    num_eval = 1000

    for i in range(num_eval):
        #Random splitting data into train,val sets
        train_data_inds = random.sample(range(total_data_size), train_data_size)
        val_data_inds = list(set(range(total_data_size))- set(train_data_inds))
        X_train = X[train_data_inds]
        Y_train = Y[train_data_inds]
        X_val = X[val_data_inds]
        Y_val = Y[val_data_inds]

        # Creating SVR classifiers with linear,gaussian and polynomial kernels
        # svr_lin = SVR(kernel='linear', C=1000)
        #svr_rbf = SVR(kernel='rbf', C=1000, gamma=0.01)
        svr_poly = SVR(kernel='poly', C=1000, degree=3,gamma=0.05)

        #classifiers = {"types":[svr_lin,svr_rbf,svr_poly],"kernel_names": ["Linear","RBF","Polynomial"]}
        classifiers = {"types": [svr_poly], "kernel_names": ["Polynomial"]}

        for classifier,kernel_name,id in zip(classifiers["types"],classifiers["kernel_names"],range(3)):
            #print("SVM with",kernel_name,"kernel:")
            classifier.fit(X_train, Y_train)

            # Calculating Train set error
            predictions_train = classifier.predict(X_train)
            get_positive_vals = lambda x: x if x >= 0 else 0
            predictions_train = [get_positive_vals(y) for y in predictions_train]
            train_error_int = get_euclidean_error(predictions_train, Y_train,round=True)
            train_error_arr[id] += train_error_int

            # Calcualting Val set error
            predictions_val = classifier.predict(X_val)
            get_positive_vals = lambda x: x if x >= 0 else 0
            predictions_val = [get_positive_vals(y) for y in predictions_val]
            val_error_int = get_euclidean_error(predictions_val, Y_val,round=True)
            val_error_arr[id] += val_error_int

            avg_err = (train_error_int+val_error_int)/2
            if val_error_int < best_val_error and val_error_int >= train_error_int:
                best_error_info ={"kernel_name":kernel_name,"train_error":train_error_int,"val_error":val_error_int}
                best_val_error = val_error_int

    train_error_arr = train_error_arr/num_eval
    val_error_arr = val_error_arr / num_eval
    
    print("Evaluation finished, showing average results from",num_eval,"evaluations.")
    for train_error,val_error,kernel_name in zip(train_error_arr,val_error_arr,classifiers['kernel_names']):
        print("SVM with",kernel_name,": Train error",train_error,", Val error:",val_error)
    print(
        "Best result for",best_error_info["kernel_name"],"kernel",
        ", train error:",best_error_info["train_error"],
        ", val error:", best_error_info["val_error"]
    )