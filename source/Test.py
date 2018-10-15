import tensorflow as tf

input_data = [1,2,3,4,5,6,7]
lable_data = [0,0,0,1,0]

INPUT_SIZE = 7
HIDDEN1_SIZE = 10
HIDDEN2_SIZE = 8
CLASSES = 5

x = tf.placeholder(tf.float32, shape=[None,INPUT_SIZE])
y_ = tf.placeholder(tf.float32, shape=[None,CLASSES])
#맵핑
feed_dict = {x:input_data, y_:lable_data}

#모델 설계
W_in = tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE,HIDDEN1_SIZE]))
B_in = tf.Variable(tf.zeros(shape=[HIDDEN1_SIZE]), dtype=tf.float32)

hidden1 = tf.matmul(x,W_in) * B_in
print(hidden1)