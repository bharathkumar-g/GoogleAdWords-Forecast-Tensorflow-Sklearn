from sklearn.neighbors import KNeighborsRegressor
import random
from data_utils import *

train_data_size = 600
val_data_size = 70
test_data_size = 60
total_data_size = train_data_size + val_data_size +test_data_size
num_eval = 5
num_n_neighbours = 1
start_n_neighbours = 5
val_errors = np.zeros(num_n_neighbours)
train_errors = np.zeros(num_n_neighbours)
best_val_error = 1000
opt_train_error = 1000
test_error = 1000

if __name__ == '__main__':
    #Getting fully preprocessed dataframes as numpy arrays, specify output = 'conversions' for conversions
    X,Y = get_processed_dataframe('ad_data.csv',output='conversions')

    # Converting to numpy arrays
    X = np.array(X)
    Y = np.array(Y)

    for n,n_neighbours in enumerate(range(start_n_neighbours,start_n_neighbours+num_n_neighbours)):
        n_eval = 0
        #This loop is while because of distribution check
        while n_eval < num_eval:
            # Random splitting data into train,val sets
            train_data_inds = random.sample(range(total_data_size), train_data_size)
            test_val_data_inds = list(set(range(total_data_size)) - set(train_data_inds))
            X_train = X[train_data_inds]
            Y_train = Y[train_data_inds]
            X_val = X[test_val_data_inds[:val_data_size]]
            Y_val = Y[test_val_data_inds[:val_data_size]]
            X_test = X[test_val_data_inds[val_data_size:]]
            Y_test = Y[test_val_data_inds[val_data_size:]]

            #Checking data distribution
            if not check_distribution(mean_array=[np.mean(Y_train), np.mean(Y_val), np.mean(Y_test)],
                                      global_mean=np.mean(Y), margin=0.5):
                continue
            
            #Training Knn regressor
            knn_regressor = KNeighborsRegressor(n_neighbors=n_neighbours, weights='uniform')
            knn_regressor.fit(X_train, Y_train)

            #Making predictions on train set
            predictions_train = knn_regressor.predict(X_train)
            train_error = get_euclidean_error(predictions_train, Y_train)
            train_errors[n] += train_error

            # Making predictions on val set
            predictions_val = knn_regressor.predict(X_val)
            val_error = get_euclidean_error(predictions_val,Y_val)

            #If this is the best val error so far, save the error and get test error
            if val_error < best_val_error and val_error > train_error:
                best_val_error = val_error
                opt_train_error = train_error
                predictions_test = knn_regressor.predict(X_test)
                predictions_test = get_positive_vals(predictions_test)
                test_error = get_euclidean_error(predictions_test, Y_test, round=True)

            val_errors[n] += val_error
            n_eval += 1

    print("Test error:", test_error)

    val_errors = val_errors/num_eval
    train_errors = train_errors/num_eval
    print("Best error: Train:",opt_train_error,", Val:",best_val_error)
    print("Best avg val error for k = ",np.argmin(val_errors)+start_n_neighbours,",avg val error = ",np.min(val_errors))

    plt.plot(train_errors,label='avg_train')
    plt.plot(val_errors,label='avg_test')
    plt.xlabel('k parameter', fontsize=16)
    plt.ylabel('Error', fontsize=16)
    plt.legend()
    plt.show()