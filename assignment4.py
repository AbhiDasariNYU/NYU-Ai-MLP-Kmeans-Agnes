import numpy as np
import math




### Assignment 4 ###

class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi)
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)




class FCLayer:
	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b

	def forward(self, input):
		#Write forward pass here
		self.x=input
		res = np.dot(input, self.w) + self.b
		return res



	def backward(self, gradients):
		self.gradients=gradients
		w_cap   = np.dot(self.x.T, gradients)
		x_cap  = np.dot(gradients, self.w.T)

		self.w = self.w-self.lr * w_cap
		self.b = self.b-self.lr * gradients
		return x_cap


class Sigmoid:
	def __init__(self):
		None
	def sigmoid(self,Z):
		A = 1 / (1 + np.exp(-Z))
		return A

	def forward(self, input):
		# Write forward pass here
		self.x = input
		self.result = self.sigmoid(input)
		return self.result

	def backward(self, gradients):
		# Write backward pass here

		sig = self.sigmoid(self.x)
		return np.multiply(np.multiply(sig, (1 - sig)), gradients)


class K_MEANS:

	def __init__(self, k, t):
		#k_means state here
		#Feel free to add methods
		# t is max number of iterations
		# k is the number of clusters
		self.k = k
		self.t = t

	def distance(self, centroids, datapoint):
		#print(centroids)
		#print(datapoint)
		diffs = (centroids - datapoint)**2
		#print(diffs)
		return np.sqrt(diffs.sum(axis=0))







	def train(self, X):
			points = {}
			randomlist = np.random.randint(0,len(X), self.k)
			for i in range(self.k):
				points[i] = X[randomlist[i]]
			for _ in range(self.t):
					classes = {}
					for i in range(self.k):
						classes[i] = []
					for row in X:
						distances=[]
						for point in points:
							distance=self.distance(points[point],row)
							distances.append(distance)
						minimum = distances.index(min(distances))
						classes[minimum].append(row)
					for c in classes:
						points[c] = np.average(classes[c], axis=0)

			#print(classes)
			clusters=[]
			found=False
			for row  in X:
				#print(datapoint)
				#print("_______________")
				found=False
				if(not found):
					for i in classes.keys():
						if (not found):
							for j in classes[i]:
								if (not found):
									if (row==j).all():
										clusters.append(i)
										found=True


			return clusters



class AGNES:
	#Use single link method(distance between cluster a and b = distance between closest
	#members of clusters a and b
	def __init__(self, k):
		#agnes state here
		#Feel free to add methods
		# k is the number of clusters
		self.k = k
	def mirage(self,L):
		if isinstance(L, list):
			ret = []
			for i in L:
				ret.append(self.mirage(i))
		elif isinstance(L, (int, float, type(None), str, bool)):
			ret = L
		else:
			raise ValueError("Unexpected type for mirage function")

		return ret

	def distance(self, a, b):
		diffs = (a - b)**2
		return np.sqrt(diffs.sum())

	def minmax(self,a,b):
		x=min(a,b)
		y=max(a,b)
		return x,y



	def minimum(self,distances):
		minimum = 10000000
		a = 0;b = 0
		i=0
		while i <len(distances):
			j=0
			while j < i:
				if minimum > distances[i][j]:
					minimum = distances[i][j]
					a = i;b = j
				j+=1
			i+=1
		x,y=self.minmax(a,b)
		return x,y

	def indexer(self,clusters):
		result = []
		counter = 0
		for item in clusters:
			result.append((counter, item))
			counter += 1
		return result

	def update_cluster_ids(self,cluster_ids, row, column):
		# We know than row value < column value
		cluster_A = cluster_ids[row]
		cluster_B = cluster_ids[column]
		combine_cluster = []

		for cluster_id in cluster_A:
			combine_cluster.append(cluster_id)
		for cluster_id in cluster_B:
			combine_cluster.append(cluster_id)

		cluster_ids.pop(row)
		cluster_ids.pop(column - 1)
		cluster_ids.insert(row, combine_cluster)

		return cluster_ids

	def train(self, X):
		clusters = []
		i=0
		while i < (len(X)):
			cluster = []
			cluster.append(i)
			clusters.append(cluster)
			i+=1

		limit = len(clusters)

		distances = []
		i=0
		while i < len(X):
			row_matrix = [float("inf")] * len(X)
			j=0
			while j < i:
				row_value = X[i]
				column_value = X[j]
				distance = self.distance(row_value, column_value)
				row_matrix[j] = distance
				j+=1
			distances.append(row_matrix)
			i+=1



		while limit > self.k:
			x, y = self.minimum(distances)
			old = self.mirage(distances)
			merge_row_values = distances[x]
			merge_column_values = distances[y]
			# Update merge row values
			row = [100000000] * len(distances)
			for i in range(x):
				new = min(merge_row_values[i], merge_column_values[i])
				row[i] = new
			# Delete merge row and replace with revised values
			distances.pop(x)
			distances.insert(x, row)
			mini=0
			# Update column values
			for i in range(len(old)):
				if i > x:
					mini = min(old[y][i],old[i][y])
					new = min(old[i][x], mini)
					distances[i][x] = new
				# Delete the merge column in each row
				distances[i].pop(y)
			# Delete entire merge row; here merge column which is actually one of the row
			distances.pop(y)
			A = clusters[x]
			B = clusters[y]
			merge = []
			for i in A:
				merge.append(i)
			for j in B:
				merge.append(j)
			clusters.pop(x)
			clusters.pop(y - 1)
			clusters.insert(x, merge)
			clusters = clusters
			limit = len(clusters)
		cluster = {}
		for i, j in self.indexer(clusters):
			for i in j:
				cluster[i] = j[0]
		res = []
		for key in sorted(cluster):
			res.append(cluster[key])
		return res



