import theano
from theano import In
import numpy as np
import theano.tensor as T
from theano import function, pp
from pprint import pprint
from utilbox import *
import tensorflow as tf

a = T.dscalar('a')
b = T.dscalar('b')
c = a + b

f = function([a, b], c)

res = f(2.33, 4.5)

a = np.asarray([[1, 2], [3, 4], [5, 6]])
print(a[1, 0])
b = 2
a = a * b
print(a)
seperateLine()
m1 = T.dmatrix('m1')
m2 = T.dmatrix('m2')
ms = m1 + m2
f = function([m1, m2], ms)
pprint(f)
pprint(f([[1, 2], [3, 4]], [[10, 20], [30, 40]]))

x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
x_squared = x ** 2
x_abs = abs(x)
logistic = function([x], [s, x_squared, x_abs])
m1 = np.asarray([[0, 6], [-6, 0]])
pprint(logistic(m1))
seperateLine()
node1 = tf.constant(3, dtype=tf.float32, name='node1')
node2 = tf.constant(2.0, name='node2')
print(node1)
print(node2)
node3 = tf.add(node1, node2, name='add1')
node4 = tf.multiply(node2, node3, name='multiply1')
with tf.Session() as sess:
  writer = tf.summary.FileWriter("output", sess.graph)
  print(sess.run(node4))
  writer.close()

a = tf.placeholder(tf.double, name='ph1')
b = tf.placeholder(tf.double, name='ph2')
adder_node = a + b
add_and_triple = adder_node * 3
with tf.Session() as sess:
  writer = tf.summary.FileWriter("output", sess.graph)
  print(sess.run(add_and_triple, {a: [[1, 2], [3, 4]], b: 8.001}))
  writer.close()

W = tf.Variable([.10], dtype=tf.float32)
b = tf.Variable([-.10], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = tf.multiply(W, x) + b
init = tf.global_variables_initializer()

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

# fixW = tf.assign(W, [-1.])
# fixb = tf.assign(b, [1.])

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

with tf.Session() as sess:
  writer = tf.summary.FileWriter("output", sess.graph)
  sess.run(init)
  for i in range(10000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
  seperateLine()
  print(sess.run([W, b]))
  writer.close()
