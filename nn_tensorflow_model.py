import pandas as pd
import tensorflow as tf
from data_utils import *

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
    df = df[['year','month','day', 'day_of_week','working_day', 'total_day_count', 'impressions', 'clicks', 'conversions', 'cost',
             'total_conversion_value', 'average_position', 'reservations', 'price']]

    # Renaming columns names
    df = df.rename(index=str,
                   columns={"average_position": "avg_position", "total_conversion_value": "tot_conversion_val"})

    #Creating 2 new columns with values of clicks&conversions of the next day
    df['next_clicks'] = get_next_values(df['clicks'])
    df['next_conversions'] = get_next_values(df['conversions'])

    #Dropping last row, because we won't learn anything from it. We have already extracted the clicks and conversions.
    df = df[:-1]

    #Specifying previous day data to use as features. Use diff to get derivatives(return difference between current and previous feature)
    #df = get_previous_vals(df,n_features=3,diff=True)

    #Shuffling the data, keep the index
    df = df.sample(frac=1)

    #Dividing the set into input features and output,
    Y = df.ix[:,'next_clicks']
    X = df.drop(['next_clicks','next_conversions'],1)

    print(X.head(10))

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

    train_data = DataWrapper(X_train,Y_train,train_data_size,batch_size)
    val_data = DataWrapper(X_val, Y_val, val_data_size, batch_size)

    input_features = 14
    fc1_dim = 30
    fc2_dim = 30
    fc3_dim = 60
    out_layer_dim = fc2_dim
    # Defining input/output placeholders
    X = tf.placeholder(dtype=tf.float32, shape=[None, input_features])
    Y = tf.placeholder(dtype=tf.float32, shape=[None])
    keep_prob = tf.placeholder(tf.float32)
    #learning_rate = tf.placeholder(tf.float32)

    # Defining weights and biases
    W = {
        "fc1": tf.get_variable("W1", shape=[input_features, fc1_dim], initializer=tf.contrib.layers.xavier_initializer()),
        "fc2": tf.get_variable("W2", shape=[fc1_dim, fc2_dim], initializer=tf.contrib.layers.xavier_initializer()),
        "fc3": tf.get_variable("W3", shape=[fc2_dim, fc3_dim], initializer=tf.contrib.layers.xavier_initializer()),
        "out": tf.get_variable("W4", shape=[out_layer_dim, 1], initializer=tf.contrib.layers.xavier_initializer()),
    }

    B = {
        "fc1": tf.get_variable("B1", shape=[fc1_dim], initializer=tf.zeros_initializer()),
        "fc2": tf.get_variable("B2", shape=[fc2_dim], initializer=tf.zeros_initializer()),
        "fc3": tf.get_variable("B3", shape=[fc3_dim], initializer=tf.zeros_initializer()),
        "out": tf.get_variable("B4", shape=[1], initializer=tf.zeros_initializer()),
    }

    # Defining our model
    #fc = tf.reshape(X, [-1, W['fc'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(X, W['fc1']), B['fc1'])
    fc1 = tf.nn.relu(fc1)
    fc2 = tf.add(tf.matmul(fc1, W['fc2']), B['fc2'])
    fc2 = tf.nn.relu(fc2)
    #fc3 = tf.add(tf.matmul(fc2, W['fc3']), B['fc3'])
    #fc3 = tf.nn.relu(fc3)

    # Apply Dropout
    fc_drop = tf.nn.dropout(fc2, keep_prob)

    # Output
    output = tf.add(tf.matmul(fc_drop, W['out']), B['out'])
    output = tf.nn.relu(output)
    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.squared_difference(output,Y))
    loss_euclides = tf.reduce_mean(tf.abs(output-Y))
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,)
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss_op)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        best_val_loss = 1000
        best_train_loss = 1000
        val_acc_his = np.zeros(epochs)
        train_acc_his = np.zeros(epochs)
        for epoch in range(epochs):
            train_loss = sess.run(loss_op, feed_dict={X: train_data.data, Y: train_data.labels, keep_prob: 1.0})
            train_acc_his[epoch] = train_loss
            if train_loss < best_train_loss and epoch != 0:
                best_train_loss = train_loss
                acc_info = "train acc has improved!"
            else:
                acc_info = ""
            val_loss = sess.run(loss_op, feed_dict={X: val_data.data, Y: val_data.labels, keep_prob: 1.0})
            val_acc_his[epoch] = val_loss
            if val_loss < best_val_loss and epoch != 0:
                best_val_loss = val_loss
                acc_info += "val acc has improved! Model saved."
                saver.save(sess, "/tmp/model.ckpt")
            else:
                acc_info += ""
            print("Accuracy after", str(epoch), "epochs: Train set:", train_loss, ", Val set:", val_loss, acc_info)
            for step in range(steps_in_epoch):
                batch_x, batch_y = train_data.get_next_batch()
                #print(batch_x,batch_y)
                # Run optimization op (backprop)
                _, pred, loss_ = sess.run([train_op, output, loss_op],
                                          feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
                #print(pred,loss_)
        saver.restore(sess, "/tmp/model.ckpt")
        print("Model loaded.")
        euclides_loss = sess.run(loss_euclides, feed_dict={X: X_dataset, Y: Y_dataset, keep_prob: 1.0})
        squared_loss = sess.run(loss_op, feed_dict={X: X_dataset, Y: Y_dataset, keep_prob: 1.0})
        print("Total loss: Euclides:",euclides_loss,", Squared", squared_loss)
        plt.plot(train_acc_his)
        plt.plot(val_acc_his)
        plt.show()
        print("Optimization Finished!")





