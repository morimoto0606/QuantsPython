import tensorflow as tf
import numpy as np

tf.reset_default_graph()
a = tf.placeholder(tf.float32, name='a')
b = tf.placeholder(tf.float32, name='b')
c = tf.placeholder(tf.float32, name='c')
d = tf.placeholder(tf.float32, name='d')
e = a * b
f = c * d
g = e + f
ga, gb, gc, gd, ge, gf = tf.gradients(g, [a, b, c, d, e, f])
feed_dict = {a: 3, b: 2, c: 4, d: 5}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([a, b, c, d, e, f, g], feed_dict))
    # 赤色の勾配計算
    print(sess.run([ga, gb, gc, gd, ge, gf], feed_dict))

tf.reset_default_graph()
x = tf.placeholder(tf.float32, name='x')
vol = tf.placeholder(tf.float32, name='vol')

y = tf.exp(x) * vol
dydx, dydvol = tf.gradients(y, [x, vol])
feed_dict = {x: 0.0, vol: 0.3}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run([dydx, dydvol], feed_dict))
