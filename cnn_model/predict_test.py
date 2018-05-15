import tensorflow as tf
import numpy as np
import os,cv2
import sys
import load_dataset


# First, pass the path of the image
test_path1= '../dataset/dataset_smile_test/'
test_path2= '../dataset/dataset_gender_test/'
model_path1= './smile-model-cascade/'
model_path2= './gender-model-cascade/'
model_path3= './smile-model-tcdcn/'
model_path4= './gender-model-tcdcn/'
graph1 = 'smile.ckpt.meta'
graph2 = 'gender.ckpt.meta'

image_size =  150 # 331
num_channels = 3
classes1 = ['not_smiling','smiling'] # 0 = no smiling, 1=smiling
classes2 = ['female','male'] # 0 = female, 1=male

data1 = load_dataset.read_test_set(test_path1,image_size,classes1)
#data2 = load_dataset.read_test_set(test_path2,image_size,classes2)
batch_size = 2995

#The input to the network is of shape [None image_size image_size num_channels]:
#x_batch = images.reshape(1, image_size,image_size,num_channels)

x_batch, y_true_batch, _, cls_batch = data1.test.next_batch(batch_size)

# Restore the saved model
session = tf.Session()
# Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph(model_path1+graph1)
# Load the weights saved using the restore method.
saver.restore(session, tf.train.latest_checkpoint(model_path1))
# Accessing the default graph which we have restored
graph = tf.get_default_graph()
# Now, let's get hold of the op that we can be processed to get the output.
y_pred = graph.get_tensor_by_name("y_pred:0")
# Feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_true_cls = tf.argmax(y_true, axis=1)

# Creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {x: x_batch,  y_true: y_true_batch}
result=session.run(y_pred, feed_dict=feed_dict_testing) # result format: [probabiliy_of_smile probability_of_non-smiling]
y_pred_cls = tf.argmax(result, axis=1)
# Accuracy
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc = session.run(accuracy, feed_dict=feed_dict_testing)
msg = "Testing Accuracy of "+model_path1+"  {0:>6.1%}"
print(msg.format(acc))