import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats as st
import tensorflow as tf
# from sklearn.naive_bayes import GaussianNB
# from sklearn.neural_network import 



# Load the data
wine_red = pd.read_csv("data/winequality-red.csv", sep = ";")
wine_white = pd.read_csv("data/winequality-white.csv", sep = ";")

# # View first five samples of the Red wine data

# print(wine_red_df.head())
# print(wine_white_df.head())

# # Split the the whole data into test and train
# X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(wine_red_df.iloc[:, :-1], wine_red_df.iloc[:, -1],
#                                                     test_size=0.2, random_state=1)
# X_train_white, X_test_white, y_train_white, y_test_white = train_test_split(wine_white_df.iloc[:, :-1], wine_white_df.iloc[:, -1],
#                                                     test_size=0.2, random_state=1)

# Adding the new column wine type 
def boxcox_trans(data):
    for i in range(data.shape[1]):
        data.iloc[:, i], _ = st.boxcox(data.iloc[:, i])
    return data

# boxcox
red_trans = wine_red.copy(deep = True)
# for i in range(red_trans.iloc[:,:-1].shape[1]):
#     red_trans.iloc[:,:-1].iloc[:,i]=st.boxcox(red_trans.iloc[:,:-1].iloc[:,i])
boxcox_trans()
white_trans = wine_white.copy(deep = True)

for i in range(white_trans.iloc[:,:-1].shape[1]):
    red_trans.iloc[:,:-1].iloc[:,i]=st.boxcox(red_trans.iloc[:,:-1].iloc[:,i])
    

red_trans["wine type"] = 0
white_trans["wine type"] = 1

# Concatenate both the dataset
wine = pd.concat([red_trans, white_trans], axis = 0, ignore_index = False)

# Split the the whole data into test and train
X_train, X_test, y_train, y_test = train_test_split(wine.iloc[:, :-1], wine.iloc[:, -1], test_size=0.25, random_state=1)               


# using neural networks

# parameters initialization
learning_rate = 0.001
batch_size = X_train.shape[0] // 10
epochs = 1000
num_features = X_train.shape[1]
epoch_list = []
epochs_to_print = epochs // 10
hidden_layer_units = 30
avg_cost_list = []

X_placeholder = tf.placeholder(tf.float32, [None, num_features], name='X')
y_placeholder = tf.placeholder(tf.float32, [None, 2], name='y')
# one_hot_encoder y_train_one_hot =
label_one_hot = []
for lable in y_train:
    index = [1]*2
    index[lable] = 0
    label_one_hot.append(index)

y_train_one_hot = label_one_hot

label_one_hot = []
for lable in y_test:
    index = [1]*2
    index[lable] = 0
    label_one_hot.append(index)

y_test_one_hot = label_one_hot

# use 2 layer neural network



merged_summaries = tf.summary.merge_all()
# The tensorflow session is stared here:
# with tf.Session() as sess:
#     tf.global_variables_initializer().run
#     cost = 0
#     for i in range(epochs):
#         # x batch y batch
#         sample = np.random.choice(np.array())


#         # Feeder
#         feed_dict = {X_placeholder: X_batch, y_placeholder: y_batch}
#         # Compute the cost
#         _, current_cost = sess.run([training_step, cost], feed_dict)
#         # Sum the overall cost
#         cost += current_cost