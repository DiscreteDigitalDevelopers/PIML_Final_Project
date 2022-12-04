import pandas as pd


# configuration
DATA_PATH = 'data.tsv'


if __name__ == '__main__':

	# read tsv file
	df = pd.read_csv(DATA_PATH, sep = '\t')

	