import theano
import numpy
from theano import tensor as T
from theano import function

a = T.dscalar('a')
b = T.dscalar('b')
c = a + b

f = function([a,b],c)

res = f(2.33, 4.5)
print(res)
print(f)
print(type(a))

