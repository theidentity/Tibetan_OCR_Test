import tensorflow as tf
import numpy as np

a = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(100, 100)).astype(np.float32),name='a')
b = tf.constant(np.random.normal(loc=0.0, scale=1.0, size=(100, 100)).astype(np.float32),name='b')
c = tf.matmul(a, b)
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
