import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow import keras


#Read in data, need to tab seperapte since content_string has commas in it
data1 = pd.read_csv("data/RC-ML-MODEL-DATA.txt", sep = "\t", encoding = 'ISO-8859-1')
data2 = pd.read_csv("data/RC-ML-MODEL-DATA-02.txt", sep = "\t", encoding = 'ISO-8859-1')
tickets = pd.concat([data1,data2])

#drop a bunch of columns that would be made after the identification and resolution ratings
cols_to_select = list(range(1, 21))
tickets = tickets.iloc[:, cols_to_select]

# one-hot encode note aggregations and content_keyword string
tickets[['note_aggregation_level1', 'note_aggregation_level2', 'content_keyword_string']] = tickets[['note_aggregation_level1', 'note_aggregation_level2', 'content_keyword_string']].apply(lambda x: x.str.replace(' ', '').str.split('+'))
for i in ['note_aggregation_level1', 'note_aggregation_level2', 'content_keyword_string']:
  one_hot = pd.get_dummies(tickets[i].apply(pd.Series).stack()).sum(level=0)
  tickets = pd.concat([tickets, one_hot], axis=1)
tickets.drop(columns = ['note_aggregation_level1', 'note_aggregation_level2', 'content_keyword_string'], inplace=True)

#get data ready for models
X = tickets.drop(['problem_rating_identifitication', 'problem_rating_resolution', 'content_string', 'category_id', 'category', 'user_id', 'created_on', 'created_year', 'created_month', 'created_day', 'created_dayname',
                  'language', 'sentiment', 'note', 'response_first'], axis =1)
#If content_keyword_string or note_aggregation_level1 or note_aggregation_level2 is null then zero for all categories
X = X.fillna(0).astype('int')
Y = tickets[['problem_rating_identifitication', 'problem_rating_resolution']]

#train test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=43)


#set network structure
keras.utils.set_random_seed(14)
network = keras.models.Sequential([
      keras.layers.Dense(256, activation = 'relu',input_shape=(50,)),
      keras.layers.Dense(256, activation = "relu"),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(128, activation = 'relu'),
      keras.layers.Dropout(0.3),
      keras.layers.Dense(6, activation="softmax")
])


#compile network
network.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.Adam(learning_rate=0.005),
                metrics = ["accuracy"])

# X_train.shape, Y_train['problem_rating_identifitication'].shape

#fit network
history = network.fit(X_train, Y_train['problem_rating_identifitication'],
                      epochs=10, batch_size=32,
                      validation_data=(X_valid, Y_valid['problem_rating_identifitication']))

network.save('models/identification_model.h5')


# for resolution
#set network stucture
keras.utils.set_random_seed(2)
network = keras.models.Sequential([
      keras.layers.Dense(256, activation = 'relu',input_shape=(50,)),
      keras.layers.Dense(256, activation = "relu"),
      keras.layers.Dropout(0.5),
      keras.layers.Dense(128, activation = 'relu'),
      keras.layers.Dropout(0.3),
      keras.layers.Dense(6, activation="softmax")
])

#compile network
network.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.Adam(learning_rate=0.005),
                metrics = ["accuracy"]) ##errors,algorithm,accuracy


#fit network
history = network.fit(X_train, Y_train['problem_rating_resolution'],
                      epochs=15, batch_size=32,
                      validation_data=(X_valid, Y_valid['problem_rating_resolution']))

network.save('models/resolution_model.h5')