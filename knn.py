import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
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

def get_euclides_error(predictions, labels, round=False, print_arrays=False):
    if round:
        predictions = np.rint(predictions)
    total_error = 0
    for y_pred, label in zip(predictions, labels):
        total_error = total_error + np.abs(y_pred - label)
    if print_arrays:
        results = np.zeros((len(predictions), 2))
        results[:, 0] = predictions
        results[:, 1] = labels
        print(results)
    return total_error / len(predictions)

if __name__ == '__main__':
    # Read data
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

    # Specifying previous day data to use as features. Use diff to get derivatives(return difference between current and previous feature)
    # df = get_previous_vals(df,n_features=1,diff=True)

    # Shuffling the data, keep the index
    df = df.sample(frac=1)

    # Dividing the set into input features and output,
    Y = df[['next_clicks']].astype(float)
    X = df.drop(['next_clicks', 'next_conversions'], 1)

    # Normalizing inputs
    X = (X - X.min() - (X.max() - X.min()) / 2) / ((X.max() - X.min()) / 2)

    # Splitting into train,val,test sets
    X = np.array(X)
    Y = np.array(Y)

    num_eval = 1000
    num_n_neighbours = 50

    scores = np.zeros(num_n_neighbours)

    for n_neighbours in range(1,num_n_neighbours+1):

        for i in range(num_eval):
            # Random splitting data into train,val sets
            train_data_inds = random.sample(range(total_data_size), train_data_size)
            val_data_inds = list(set(range(total_data_size)) - set(train_data_inds))
            X_train = X[train_data_inds]
            Y_train = Y[train_data_inds]
            X_val = X[val_data_inds]
            Y_val = Y[val_data_inds]

            knn_classifier = KNeighborsRegressor(n_neighbors=n_neighbours, weights='distance')
            knn_classifier.fit(X_train, Y_train)
            predictions_train = knn_classifier.predict(X_train)
            train_error = get_euclides_error(predictions_train, Y_train)
            predictions_val = knn_classifier.predict(X_val)
            val_error = get_euclides_error(predictions_val,Y_val)
            scores[n_neighbours-1] += val_error
            #print("Train error:",train_error,", Val error:",val_error)

    scores = scores/num_eval
    print(scores)
    print("Best avg score for n = ",np.argmin(scores)+1,",avg score = ",np.min(scores))

    plt.plot(scores)
    plt.show()