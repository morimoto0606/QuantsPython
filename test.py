import unittest
from urllib.parse import parse_qs

class Test(unittest.TestCase):
    def setUp(self):
        self.a = 0

    def test_item4(self):
        my_values = parse_qs('red=5&blue=0&green=',
                             keep_blank_values=True)
        print(repr(my_values))
        print(my_values)

        red = my_values.get('red', [''])[0] or 0
        green = my_values.get('green', [''])[0] or 0
        opacity = my_values.get('opacity', [''])[0] or 0
        print('Red: %r' % red)
        print('Green: %r' % green)
        print('Opacity: %r' % opacity)


