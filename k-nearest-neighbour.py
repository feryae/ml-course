# Example of getting neighbors for an instance
from math import sqrt

# calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance += (row1[i] - row2[i])**2
	return sqrt(distance)

# Locate the most similar neighbors
def get_neighbors(train, test_row, num_neighbors):
	distances = list()
	for train_row in train:
		dist = euclidean_distance(test_row, train_row)
		distances.append((train_row, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(num_neighbors):
		i = i + 1
		neighbors.append(distances[i][0])
	return neighbors

# Test distance function
dataset = [[6207,1.0,0],
	[43851,1.4,1],
	[38237,5.8,2],
	[319,0.6,0],
	[9073,6.7,2],
	[1908,2.3,1],
	[23555,0.8,1],
	[95117,6.6,2],
	[31642,3.4,1],
	[21216,2.5,1],
    [27657,5.5,2],
    [1999,2.5,1],
    [9043,2.9,1],
    [10614,0.1,0]]
testset = [[20226,3.0],
    [28320,1.7],
    [5655,0.1],
    [1998,3.0],
    [9890,9.4],
    [10953,3.9]]
for i in range(len(testset)):
	neighbors = get_neighbors(dataset, testset[i], 1)
	for neighbor in neighbors:
		print(testset[i])
		if neighbor[2] == 0 :
			print("rejected")
		elif neighbor[2] == 1 :
			print("considered")
		elif neighbor[2] == 2 :
			print("accepted")

