import sys
import os
import numpy as np
from matplotlib import pyplot as plt

class Perceptron:
	def __init__(self):
		self.weights = None
		self.report = Report()

	def predict(self, data: np.array):
		dot = np.sum(np.dot(data, self.weights))
		if dot > 0:
			return 1.0
		else:
			return -1.0

	def fit(self, training_sample: np.array, desired_value):
		prediction= self.predict(training_sample)
		if prediction > desired_value:
			self.weights -= training_sample
		elif prediction < desired_value:
			self.weights += training_sample

	def predict_all(self, data: np.array, labels: np.array):
		error = 0
		for row, expected in zip(data, labels):
			actual = self.predict(row)
			if actual != expected:
				error = np.absolute(actual - expected)
		return error

	def init(self):
		raw_data = np.loadtxt(sys.argv[1], delimiter=',')
		# init bias
		rows = raw_data.shape[0]
		bias = np.ones(rows)
		bias.shape = (rows, 1)
		# inputs w1 and w2
		data = raw_data[:, [0, 1]]
		# add bias to data
		data = np.hstack((bias, data))
		# get the labels
		output = raw_data[:, [2]].flatten()
		# w2, w1 and w0 (w0 is 1) y = w2*x + w1*y + b (1*b or wo*b)
		self.weights = [0 for i in range(len(list(set(output))) + 1)]
		#print(weights)
		return data, output

	def main(self):
		data, output = self.init()
		error = None
		while error != 0:
			for (x, y) in zip(data, output):
				self.fit(x, y)
			self.report.create(self.weights)
			error = self.predict_all(data, output.T)
		return data, output

class Report():
	def __init__(self):
		self.file_name = "output.csv"
		if os.path.exists(self.file_name):
			os.remove(self.file_name)

	def create(self, weights):
		out = open(self.file_name, "a")
		out.write("%d, %d, %d\n"%(weights[1], weights[2], weights[0]))
		out.close()

class GraphOutputter():
	def process(self, p: Perceptron, data: np.array, expected_labels: np.array, labels: np.array):
		colormap = np.array(['b', 'r'])
		ixs = [0 if x == 1 else 1 for x in expected_labels]
		xs = data[:, [1]]
		ys = data[:, [2]]
		plt.scatter(xs.flatten(), ys.flatten(), c=colormap[ixs])
		w = p.weights
		xx = np.linspace(min(xs), max(xs))
		a = -w[1] / w[2]
		yy = a * xx - (w[0]) / w[2]
		plt.plot(xx, yy, 'k-')
		plt.show()

if __name__ == "__main__":
	perceptron = Perceptron()
	data, output = perceptron.main()
	pl = GraphOutputter()
	pl.process(perceptron, data, output, np.array([[1], [-1]]))