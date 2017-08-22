import argparse
import sys
import tempfile

import pandas as pd
import tensorflow as tf

import sklearn.cross_validation

def build_estimator(model_dir, model_type):
    """Build an estimator."""
 
# Sparse base columns. Creates a Sparse Column with hashed bucket configuration.
# output_id = Hash(input_feature_string) % bucket_size.
    sparse_columns = list(map(lambda x : tf.contrib.layers.sparse_column_with_hash_bucket(x, hash_bucket_size=1000),CATEGORICAL_COLUMNS[:-1]))
    
# Continuous base columns. Creates a Real Valued Column for dense numeric data.
# A default normalizer function takes the input tensor as its argument, and returns the output tensor.
    continuous_columns = list(map(lambda x : tf.contrib.layers.real_valued_column(x),CONTINUOUS_COLUMNS))

# Transformations
# Creates a Bucketized Column for discretizing dense input.
    age_buckets = tf.contrib.layers.bucketized_column(continuous_columns[0],boundaries=[18, 25, 30, 35, 40, 45,50, 55, 60, 65,70, 75, 80, 85, 90, 95])

# Wide columns and deep columns

# wide columns, Creates a Crossed Column for performing feature crosses.
    wide_columns1 = [age_buckets,
    tf.contrib.layers.crossed_column([sparse_columns[0], sparse_columns[2]],hash_bucket_size=int(1e4)),
    tf.contrib.layers.crossed_column([age_buckets, sparse_columns[2], sparse_columns[0],sparse_columns[1]],hash_bucket_size=int(1e6)),
    tf.contrib.layers.crossed_column([sparse_columns[3], sparse_columns[4],sparse_columns[5]],hash_bucket_size=int(1e4))]
    
    wide_columns = wide_columns1 + sparse_columns
    
# deep columns, Creates an Embedding Column for feeding sparse data into a DNN.
# embedding values are l2-normalized to the value of max_norm.
    deep_columns1 = list(map(lambda x : tf.contrib.layers.embedding_column(x, dimension=8),sparse_columns))        
    deep_columns = deep_columns1 + continuous_columns
        
# selects model type    
    if model_type == "wide":
# Train a linear model to classify instances into one of multiple possible classes. 
# When number of possible classes is 2, this is binary classification
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,feature_columns=wide_columns,
            optimizer=tf.train.FtrlOptimizer(learning_rate=0.1,l1_regularization_strength=1.0,l2_regularization_strength=1.0))
    elif model_type == "deep":
# Classifier for TensorFlow DNN models
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,feature_columns=deep_columns,
        hidden_units=[100, 50])
    else:
# classifier for TensorFlow Linear and DNN joined training models
        m = tf.contrib.learn.DNNLinearCombinedClassifier(model_dir=model_dir,linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50],
        fix_global_step_increment_bug=True)
    return m


def input_fn(df):
    """Input builder function."""
    
# Creates a dictionary mapping from each continuous feature column name (k) to
# the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    
# Creates a dictionary mapping from each categorical feature column name (k)
# to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(indices=[[i, 0] for i in range(df[k].size)],values=df[k].values,dense_shape=[df[k].size, 1]) for k in CATEGORICAL_COLUMNS}
    
# Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    
# Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    
# Returns the feature columns and the label.
    return feature_cols, label


def train_and_eval(model_dir, model_type, train_steps,dataset):
    """Train and evaluate the model."""
# Creates dataframe from input dataset
    data_df = pd.read_csv(dataset)
    
# Create a list of all categorical columns from dataframe
    global CATEGORICAL_COLUMNS 
    CATEGORICAL_COLUMNS = data_df.select_dtypes(include = ['object']).columns.values.tolist()
    
# Create a list of all continuous columns from dataframe
    global CONTINUOUS_COLUMNS
    CONTINUOUS_COLUMNS = data_df.select_dtypes(exclude = ['object']).columns.values.tolist()
    
# Selects last element of categorical columns as label
    global LABEL_COLUMN 
    LABEL_COLUMN = CATEGORICAL_COLUMNS[-1]
    
# Remove NaN elements
    data_df = data_df.dropna(how='any', axis=0)
    
# Converts elements of label column in binary values
    data_df[LABEL_COLUMN] = (data_df[LABEL_COLUMN].apply(lambda x: "yes" in x)).astype(int)
    
    
# Splits input dataset in to training and testing dataset
    train_data,test_data= sklearn.cross_validation.train_test_split(data_df,test_size=0.33,random_state=42)
    
# Training dataset
    df_train = train_data
    
# Testing  dataset 
    df_test = test_data
  
# Creates temporary model directory
    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    print("model directory = %s" % model_dir)

# Calls build estimator function 
    m = build_estimator(model_dir, model_type)
    
# Fit data to the classifier model
    m.fit(input_fn = lambda: input_fn(df_train), steps = train_steps)
    
# Evaluate and gets the result
    results = m.evaluate(input_fn = lambda: input_fn(df_test), steps=1)
    
# Prints the result
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))
    
    
FLAGS = None


def main(_):
    train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,FLAGS.dataset)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
    "--model_dir",
    type=str,
    default="",
    help="Base directory for output models."
    )
    parser.add_argument(
    "--model_type",
    type=str,
    default="wide",
    help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
    )
    parser.add_argument(
    "--train_steps",
    type=int,
    default=200,
    help="Number of training steps."
    )
    parser.add_argument(
    "--dataset",
    type=str,
    default="",
    help="Path to the dataset."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
