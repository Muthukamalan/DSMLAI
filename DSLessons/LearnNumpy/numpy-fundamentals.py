import numpy as np

# Reading the example code
a = np.arange(6)
print(a,a.shape)
a2 = a[np.newaxis,:]
print(a2,a2.shape)

# What is an array
a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(a[0])


print(np.zeros(2))
print(np.ones(2))
print(np.ones(2,dtype=np.int32)) # specifyinh our datatype
print(np.empty(2))  # inital content is random 

print(np.linspace(0,5,num=20))

arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])
print(np.sort(arr))


# Adding, removing, and sorting elements
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
print(np.concatenate((a, b)))
print(np.concatenate((a, b),axis=0))


# How do you know the shape and size of an array?

array_example:np.array= np.array([
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7]
    ],
    [
        [0, 1, 2, 3],
        [4, 5, 6, 7]
    ],
    [
        [0 ,1 ,2, 3],
        [4, 5, 6, 7]
    ]
])
print(array_example)
print(array_example.ndim)
print(array_example.size)
print(array_example.shape)


# Can you reshape an array?
a = np.arange(6)
print(a)
b = a.reshape(3,2)
print(b)

print(np.reshape(b,newshape=(1,6),order='C'))


# How to convert a 1D array into a 2D array (how to add a new axis to an array)
a = np.array([1,2,3,4,5,6])
print(a.shape)

row_vector = a[np.newaxis,:]
print(row_vector.shape)

col_vector = a[:,np.newaxis]
print(col_vector)

print(np.expand_dims(a=a,axis=0))
print(np.expand_dims(a=a,axis=1))



# Indexing and Slicing

data = np.array([1,2,3])
print(data[1])
print(data[0:2])
print(data[1:])
print(data[-2:])

a = np.array([[1 , 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a[a<5])
five_up = (a>=5)
divisible_by_2 = (a%2==0)
put_condition = ((a > 2) & (a < 11))


# Index collapsed
print(a[five_up])
print(a[divisible_by_2])
print(a[put_condition])  


print(((a > 5) | (a == 5))) # Boolean ARRAY

print(np.nonzero(a<5)) #Return the indices of the elements that are non-zero

a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
b = np.nonzero(a < 5)
print(b)

list_of_coordinates = list(zip(b[0],b[1]))
for coord in list_of_coordinates:
    print(coord)

print(a[b])
print(np.nonzero(a == 42))



# How to create an array from existing data
a1 = np.array([[1, 1],[2, 2]])
a2 = np.array([[3, 3],[4, 4]])

print(np.hstack((a1,a2)))
print(np.vstack((a1,a2)))

x =  np.arange(1, 25).reshape(2, 12)
print(x)
print(np.hsplit(x,3)) # three equally shaped arrays


print(np.hsplit(ary=x,indices_or_sections=(3,4)))


# Basic array operations
data = np.array([1, 2])
ones = np.ones(2, dtype=int)
print(data+ones)
print(data-ones)
print(data*data)
print(data/data)
print(data.sum())
b = np.array([[1,1],[2,3]])
print(b.sum(axis=0))
print(b.sum(axis=1))



# Broadcasting
# There are times when you might want to carry out an operation between an array and a single number (also called an operation between a vector and a scalar) or between arrays of two different sizes.

data = np.array([1.0,2.])
print(data*1.6)

# More useful array operations
print(data.sum())
print(data.min())
print(data.max())
data = np.array([[1, 2], [5, 3], [4, 6]])
print(data.max(axis=1))


# Generating random numbers
rng = np.random.default_rng() 
print(rng.integers(low=5,high=2934,size=(2,4)))


# How to get unique items and counts
# NOTE array will be flattern
a = np.array([[[11, 11, 12, 13, 14, 15,17,23,9],[ 16, 17, 12, 13, 11, 14, 18, 19, 20]]])
print(np.unique(a))
unique_val,index_list = np.unique(a,return_index=True)
print(index_list)
unique_val,occur_count = np.unique(a,return_counts=True)
print(occur_count)



# Transposing and reshaping a matrix
print(a.shape)
print(a.T,a.T.shape)
print(a.reshape(1,-1,3,3).shape)


# How to reverse an array
print(a)
print(np.flip(a))


# Reshaping and flattening multidimensional arrays
print(a.flatten())   #changes to your new array won't change the parent array
a1 = a.flatten()
a1[0]=100
print(a)
print(a1)

print(a.ravel())   #changes to your new array will change the parent array
a1 = a.ravel()
a1[0]=100
print(a)
print(a1)


# Working with mathematical formulas
# MSE
n = 10 # number_of_samples
predictions= np.array([10,20,30,40,50])
labels     = np.array([9,21,29,41,49])
err = (1/n) * np.sum(np.square(predictions-labels))


# How to save and load NumPy objects
a = np.array([1, 2, 3, 4, 5, 6])
np.save(file="a.npy",arr=a)
b = np.load("a.npy")
print(b)

# Plotting
from matplotlib import pyplot as plt
x = np.linspace(0, 5, 20)
y = np.linspace(0, 10, 20)
plt.plot(x, y, 'purple') # line
plt.plot(x, y, 'o')      # dots
plt.show()