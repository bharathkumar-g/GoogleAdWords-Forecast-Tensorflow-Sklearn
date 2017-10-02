import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

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

if __name__=='__main__':
    #Read data
    df = pd.read_csv('ad_data.csv')

    #Parsing date
    df['year'] = df.date.apply(get_year)
    df['month'] = df.date.apply(get_month)
    df['day'] = df.date.apply(get_day)
    df = df.drop('date',1)

    #Moving year,month,day columns to front
    df = df[['year', 'month', 'day','impressions', 'clicks', 'conversions', 'cost', 'total_conversion_value', 'average_position', 'reservations', 'price']]

    #Renaming columns names
    df = df.rename(index=str, columns={"average_position": "avg_position", "total_conversion_value": "tot_conversion_val"})

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

    #Plotting correlation matrix
    #plot_corr_mat(df)

    #Dropping last row, because we won't learn anything from it. We have already extracted the clicks and conversions.
    df = df[:-1]

    # Shuffling the data, keep the index
    df = df.sample(frac=1)

    #Dividing the set into input features and output,
    Y = df.ix[:,'next_clicks']
    X = df.drop(['next_clicks','next_conversions'],1)

    #Normalizing inputs
    X = (X - X.min() - (X.max() - X.min()) / 2) / ((X.max() - X.min()) / 2)

    #Splitting into train,val,test sets
    X_train = np.array(X[:530])
    Y_train = np.array(Y[:530])
    X_val = np.array(X[530:630])
    Y_val = np.array(Y[530:630])
    X_test = np.array(X[630:])
    Y_test = np.array(Y[630:])

    train_data = DataWrapper(X_train,Y_train,530,10)
    val_data = DataWrapper(X_val, Y_val, 100, 10)
    test_data = DataWrapper(X_test, Y_test, 100, 10)

    # Defining input/output placeholders
    X = tf.placeholder(dtype=tf.float32, shape=[None, 11])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])
    keep_prob = tf.placeholder(tf.float32)

    stddev = 0.01
    learning_rate = 0.001
    epochs = 100
    batch_size = 10
    steps_in_epoch = 530//10
    num_eval_imgs = 500
    # Defining weights and biases
    W = {
        "fc": tf.Variable(tf.random_normal([11, 30], stddev=stddev)),
        "out": tf.Variable(tf.random_normal([30, 1], stddev=stddev)),
    }

    B = {
        "fc": tf.Variable(tf.random_normal([30], stddev=stddev)),
        "out": tf.Variable(tf.random_normal([1], stddev=stddev)),
    }

    # Defining our model
    #fc = tf.reshape(X, [-1, W['fc'].get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(X, W['fc']), B['fc'])
    fc = tf.nn.relu(fc)

    # Apply Dropout
    #fc = tf.nn.dropout(fc, keep_prob)

    # Output
    output = tf.add(tf.matmul(fc, W['out']), B['out'])

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.abs(output-Y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    #correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
    #total_correct = tf.reduce_sum(tf.cast(correct_pred, tf.int32))
    #accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        best_val_loss = 1000
        val_acc_his = np.zeros(epochs)
        train_acc_his = np.zeros(epochs)
        for epoch in range(epochs):
            train_loss = sess.run(loss_op, feed_dict={X: train_data.data, Y: train_data.labels, keep_prob: 1.0})
            train_acc_his[epoch] = train_loss
            val_loss = sess.run(loss_op, feed_dict={X: val_data.data, Y: val_data.labels, keep_prob: 1.0})
            val_acc_his[epoch] = val_loss
            if val_loss < best_val_loss and epoch != 0:
                best_val_loss = val_loss
                acc_info = "val acc has improved! Model saved."
            else:
                acc_info = ""
            print("Accuracy after", str(epoch), "epochs: Train set:", train_loss, ", Val set:", val_loss, acc_info)
            for step in range(steps_in_epoch):
                batch_x, batch_y = train_data.get_next_batch()
                #print(batch_x,batch_y)
                # Run optimization op (backprop)
                _, pred, loss_ = sess.run([train_op, output, loss_op],
                                          feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
                #print(pred,loss_)
        test_loss = sess.run(loss_op, feed_dict={X: test_data.data, Y: test_data.labels, keep_prob: 1.0})
        print("Validation loss:",best_val_loss,",Test loss:", test_loss)
        plt.plot(train_acc_his)
        plt.plot(val_acc_his)
        plt.show()
        print("Optimization Finished!")





