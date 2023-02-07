import numpy as np

class Data_Bundle:
    def __init__(self, events, reviews):
        self.events = events
        self.reviews = reviews

    def get_data(self):
        return self.events, self.reviews

'''
a1 = np.array([[2,3,4],[1,0,1]])
b1 = np.array([[2,1], [1,7]])
a2 = np.array([[2,9,4],[1,1,1]])
b2 = np.array([[5,1], [2,7]])
#print(type(a))

data1 = Data_Bundle(a1, b1)
data2 = Data_Bundle(a2, b2)
x1, x2 = data1.get_data()

li = [data1, data2]
ar = np.array(li)
print(ar)
'''