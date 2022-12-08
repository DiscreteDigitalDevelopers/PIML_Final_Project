import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

def main():
	with open('results.pkl', 'rb') as fp:
		results = pkl.load(fp)

	preds = results[0]
	labels = results[1]
	adiffs = results[2]
	diffs = preds - labels

	bins = np.linspace(0, 16, 100)

	hist = np.append(0, np.histogram(np.square(diffs), bins = bins)[0])
	plt.hist(np.square(diffs), bins, label = 'Squared Error')
	plt.plot(bins, np.cumsum(hist)/2.8, label = 'Cum. Sum')
	pval = (np.cumsum(hist) >= 1900)*1900/2.8
	plt.plot(bins, pval, label = 'p=0.05')
	plt.legend(loc='upper right')
	plt.yscale('log')
	plt.show()

if __name__ == '__main__':
	main()