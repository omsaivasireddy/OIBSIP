# Import the necessary python libraries
import pandas as pa
import numpy as nu
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data

classify = pa.read_csv("Iris.csv")
print("\n____All the data points are showed in a matrix form____\n")
print(classify)

# Visualize the data
print("\n____The statistical information about the dataset____\n")
print(classify.describe()) 
# this method returns the statistics of the dataset

classify=classify.drop("Id",axis=1)
# this method remove the Id column which is not useful for us
print("\n____Dataset after removing the Id column____\n")
print(classify.head())
# this method will returns the top 5 data points

print("\nThe array of unique outputs that we may get : ",end="")
print(classify.Species.unique())
# this method returns the unique outputs that can be achieved
print("\nThe count of each output are :")
print(classify.Species.value_counts())
# This method returns the count of each output

sns.pairplot(classify,hue="Species")
# This method allows user to plot the graph pairwise of each pair of columns
plt.show()
# This method will display the above method's graphs.

print("By the above data graphs, we can say that Iris-setosa are seperated from the other two types")
