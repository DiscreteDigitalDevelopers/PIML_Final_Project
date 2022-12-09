import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

def main():
	model = 'random_forest'

	with open('results.pkl', 'rb') as fp:
		results = pkl.load(fp)

	with open('results_gb.pkl', 'rb') as fp:
		preds = pkl.load(fp)

	# preds = results[0]
	labels = results[1]
	adiffs = results[2]
	diffs = preds - labels

	bins = np.linspace(0, 16, 100)

	hist = np.append(0, np.histogram(np.square(diffs), bins = bins)[0])
	plt.hist(np.square(diffs), bins, label = 'Squared Error')
	# plt.legend(loc='upper right')
	plt.xlabel('Squared Error')
	plt.ylabel('Number of Predictions')
	plt.title('Squared Error Distribution')
	plt.savefig(f'media/{model}/squared_error.png', dpi=200)
	plt.clf()

	plt.hist(diffs, np.linspace(-5, 5, 100), label = 'Error')
	# plt.legend(loc='upper right')
	plt.xlabel('Error')
	plt.ylabel('Number of Predictions')
	plt.title('Error Distribution')
	plt.savefig(f'media/{model}/error.png', dpi=200)
	plt.clf()

	# print basic stats
	print(f'Average Difference: {np.mean(np.abs(diffs)):.4f}')
	print(f'Mean Squared Error: {np.mean(np.square(diffs)):.4f}')
	print(f'Std. Dev. of Differences: {np.std(diffs):.4f}')
	print(f'Range of Differences: {np.min(diffs):.4f} to {np.max(diffs):.4f}')

	# scatter plot of predicted vs. results
	plt.scatter(preds, labels)
	plt.plot(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
	plt.xlabel('Predicted Rating')
	plt.ylabel('Actual Rating')
	plt.title('Scatter Distribution')
	plt.savefig(f'media/{model}/scatter.png', dpi=200)
	plt.clf()

	# histogram of predicted vs. results MSE
	plt.hist([np.around(preds), np.around(labels)], label = ['Predicted', 'Actual'])
	plt.legend(loc = 'upper right')
	plt.ylim(0, 650)
	plt.xlabel('Rating')
	plt.ylabel('Number of Predictions/Labels')
	plt.title('MSE Distribution')
	plt.savefig(f'media/{model}/histogram.png', dpi=200)
	plt.clf()

if __name__ == '__main__':
	main()