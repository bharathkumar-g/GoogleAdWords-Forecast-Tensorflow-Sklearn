#python version: 3.5
from sklearn.svm import SVR
from data_utils import *

train_data_size = 600
val_data_size = 70
test_data_size = 60
total_data_size = train_data_size + val_data_size + test_data_size

best_error_info = {"kernel_name": "", "train_error": 0, "val_error": 0}

# Placeholders for average scores
train_error_arr = np.zeros(3)
val_error_arr = np.zeros(3)
best_val_error = 1000
test_error = 1000

num_eval = 1

if __name__=='__main__':
    #Getting fully processed dataframe
    df_X,df_Y = get_processed_dataframe('ad_data.csv',output='clicks',raw=False)
    df_raw = get_processed_dataframe('ad_data.csv',output='clicks',raw=True)

    # Converting to numpy arrays
    X = np.array(df_X)
    Y = np.array(df_Y)

    #print(Y[0,0],Y[0,1],Y[:3,:])

    n_eval = 0
    all_predictions = None

    while n_eval < num_eval:
        #Random splitting data into train,val sets
        train_data_inds = random.sample(range(total_data_size), train_data_size)
        test_val_data_inds = list(set(range(total_data_size))- set(train_data_inds))
        X_train = X[train_data_inds]
        Y_train = Y[train_data_inds,1]
        X_val = X[test_val_data_inds[:val_data_size]]
        Y_val = Y[test_val_data_inds[:val_data_size],0]
        X_test = X[test_val_data_inds[val_data_size:]]
        Y_test = Y[test_val_data_inds[val_data_size:],0]

        #Checking distribution of data
        #if not check_distribution(mean_array=[np.mean(Y_train),np.mean(Y_val),np.mean(Y_test)],global_mean=np.mean(Y),margin=0.5):
        #   continue

        # Creating SVR classifiers with linear,gaussian and polynomial kernels
        svr_lin = SVR(kernel='linear', C=1000)
        svr_rbf = SVR(kernel='rbf', C=1000, gamma=0.03)
        svr_poly = SVR(kernel='poly', C=1000, degree=2,gamma=0.05)

        classifiers = {"types":[svr_lin,svr_rbf,svr_poly],"kernel_names": ["Linear","RBF","Polynomial"]}

        for id,(classifier,kernel_name) in enumerate(zip(classifiers["types"],classifiers["kernel_names"])):
            #print("SVM with",kernel_name,"kernel:")
            classifier.fit(X_train, Y_train)

            # Calculating Train set error
            predictions_train = classifier.predict(X_train)
            predictions_train = get_positive_vals(predictions_train)
            train_error_int = get_euclidean_error(predictions_train, Y_train,round=True)
            train_error_arr[id] += train_error_int

            # Calcualting Val set error
            predictions_val = classifier.predict(X_val)
            predictions_val = get_positive_vals(predictions_val)
            val_error_int = get_euclidean_error(np.exp(predictions_val), Y_val,round=True)
            val_error_arr[id] += val_error_int

            if val_error_int < best_val_error and val_error_int >= train_error_int:
                best_error_info ={"kernel_name":kernel_name,"train_error":train_error_int,"val_error":val_error_int}
                best_val_error = val_error_int
                predictions_test = classifier.predict(X_test)
                predictions_test = get_positive_vals(predictions_test)
                test_error = get_euclidean_error(np.exp(predictions_test), Y_test, round=True)
                all_predictions = classifier.predict(X)
        n_eval += 1

    train_error_arr = train_error_arr/num_eval
    val_error_arr = val_error_arr / num_eval

    #Calculating moving averages
    mov_avg_pred7 = get_moving_avg(np.exp(all_predictions),7)
    mov_avg_pred14 = get_moving_avg(np.exp(all_predictions), 14)
    mov_avg_pred21 = get_moving_avg(np.exp(all_predictions), 21)
    mov_avg7 = get_moving_avg(df_raw['clicks'], n=7)
    mov_avg14 = get_moving_avg(df_raw['clicks'], n=14)
    mov_avg21 = get_moving_avg(df_raw['clicks'], n=21)

    #Calculating mean relative error
    mean_rel_err7 = get_mean_rel_err(mov_avg_pred7,mov_avg7)
    mean_rel_err14 = get_mean_rel_err(mov_avg_pred14, mov_avg14)
    mean_rel_err21 = get_mean_rel_err(mov_avg_pred21, mov_avg21)

    #print("Moving averages: n = 7:",mean_rel_err7,"%, n=14:",mean_rel_err14,"%,n=21:",mean_rel_err21,"%")

    print("Test error:",test_error)

    y_test_pred_real = np.exp(predictions_test)
    print(np.column_stack((y_test_pred_real,Y_test)))

    print("Evaluation finished, showing average results from",num_eval,"evaluations.")
    for train_error,val_error,kernel_name in zip(train_error_arr,val_error_arr,classifiers['kernel_names']):
        print("SVM with",kernel_name,": Train error",train_error,", Val error:",val_error)
    print(
        "Best result for",best_error_info["kernel_name"],"kernel",
        ", train error:",best_error_info["train_error"],
        ", val error:", best_error_info["val_error"]
    )

    plt.plot(y_test_pred_real)
    plt.plot(Y_test)
    plt.show()