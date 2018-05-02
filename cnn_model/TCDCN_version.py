from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def cnn_model_fn(features, labels, mode):
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # AFWL, LFW and net images are cutted to 60x60 pixels, and have RGB color channel
  input_layer = tf.reshape(features["x"], [-1, 60, 60, 3])

  # Convolutional Layer #1
  # Input Tensor Shape: [batch_size, 60, 60, 1]
  # Output Tensor Shape: [batch_size, 56, 56, 20]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=20,           # Computes 32 features using a 5x5 filter with ReLU activation.
      kernel_size=[5, 5],
      padding="valid",       # valid Padding
      #strides=(1, 1), # by default strides=1
      activation=tf.nn.relu,
      use_bias = True,
      bias_initializer=create_biases(20))

  # Pooling Layer #1
  # First max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 56, 56, 20]
  # Output Tensor Shape: [batch_size, 28, 28, 20]
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2
  # Input Tensor Shape: [batch_size, 28, 28, 20]
  # Output Tensor Shape: [batch_size, 24, 24, 48]
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=48,             # Computes 64 features using a 5x5 filter.
      kernel_size=[5, 5],
      padding="valid",
      activation=tf.nn.relu,
      use_bias=True,
      bias_initializer=create_biases(20))

  # Pooling Layer #2
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 24, 24, 48]
  # Output Tensor Shape: [batch_size, 12, 12, 48]
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Convolutional Layer #3
  # Input Tensor Shape: [batch_size, 12 ,12 , 48]
  # Output Tensor Shape: [batch_size, 10, 10, 64]
  conv3 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,  # Computes 64 features using a 5x5 filter.
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu,
      use_bias=True,
      bias_initializer=create_biases(20))


  # Pooling Layer #3
  # Second max pooling layer with a 2x2 filter and stride of 2
  # Input Tensor Shape: [batch_size, 10, 10, 64]
  # Output Tensor Shape: [batch_size, 5, 5, 64]
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

  # Convolutional Layer #4
  # Input Tensor Shape: [batch_size, 5 ,5 , 48]
  # Output Tensor Shape: [batch_size, 3, 3, 64]
  conv4 = tf.layers.conv2d(
      inputs=pool3,
      filters=80,  # Computes 64 features using a 5x5 filter.
      kernel_size=[3, 3],
      padding="valid",
      activation=tf.nn.relu,
      use_bias=True,
      bias_initializer=create_biases(20))


  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 3, 3, 80]
  # Output Tensor Shape: [batch_size, 3 * 3 * 80]
  conv4_flat = tf.reshape(conv4, [-1, 3 * 3 * 80])

  # Dense Layer:  Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 3 * 3 * 80]
  # Output Tensor Shape: [batch_size, 256]
  dense = tf.layers.dense(inputs=conv4_flat, units=256, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 256]
  # Output Tensor Shape: [batch_size, 4]  as we have2 classes smile, no smile) or ( male, female)
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes): cross entropy
  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
  #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  # Create the Estimator
  classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/model_tcdcn") #output dir

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  classifier.train(
      input_fn=train_input_fn,
      steps=2000,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()