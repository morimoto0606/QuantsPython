import numpy
import scipy.stats

a = [1, 2, 3, 4, 5, 6]
print(a[1::2])

squares = [x ** 2 for x in a]
print(squares)

value = [len(x) for x in open('my_file.txt')]
print(value)

it = (len(x) for x in open('my_file.txt'))
print(it)
print(next(it))
print(next(it))
print(next(it))

flavor_list = ['vanilla', 'chocolate', 'pecan', 'strawberry']
for flavor in flavor_list:
    print('%s is delicious' % flavor)

for i, flavor in enumerate(flavor_list, 1):
    print('%d: %s' % (i, flavor))
