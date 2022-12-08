from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import pickle as pkl
import os

class IMDB_Dataset(Dataset):
	def __init__(self, data_split, window_size, tokenizer, load_from_file = False):
		super().__init__()
		self.window_size = window_size
		self.tokenizer = tokenizer

		path_stub = f'datasets/imdb/{data_split}'

		# get filepath - ../ is implicit from main file
		self.filepath = f'data/processed/data_{data_split}.pkl'

		# ensure file exists
		if not os.path.exists(self.filepath):
			raise IOError('No such file found')

		# load pkl file
		with open(self.filepath, 'rb') as fp:
			p = pkl.load(fp)

		# get keys
		keys = p.keys()
		self.n = len(keys)

		# if load, only load, otherwise, create and save files
		if load_from_file:
			with open(f'{path_stub}_ids.pkl', 'rb') as fp:
				self.ids = pkl.load(fp)

			with open(f'{path_stub}_masks.pkl', 'rb') as fp:
				self.masks = pkl.load(fp)

			with open(f'{path_stub}_token_type_ids.pkl', 'rb') as fp:
				self.token_type_ids = pkl.load(fp)

			with open(f'{path_stub}_ratings.pkl', 'rb') as fp:
				self.ratings = pkl.load(fp)

		else:
			# create arrays
			self.ids = np.zeros((self.n, self.window_size))
			self.masks = np.zeros((self.n, self.window_size))
			self.token_type_ids = np.zeros((self.n, self.window_size))
			self.ratings = np.zeros(self.n)

			# process each rating
			for i, (key, (rating, text)) in enumerate(tqdm(p.items(), desc = 'encoding')):
				inputs = self.tokenizer.encode_plus(
					text,
					None,
					add_special_tokens=True,
					max_length=self.window_size,
					padding="max_length",
					return_token_type_ids=True,
					truncation=True,
				)
				self.ids[i] = inputs['input_ids']
				self.masks[i] = inputs['attention_mask']
				self.token_type_ids[i] = inputs["token_type_ids"]
				self.ratings[i] = rating

			# save to pkl files
			with open(f'{path_stub}_ids.pkl', 'wb') as fp:
				pkl.dump(self.ids, fp)

			with open(f'{path_stub}_masks.pkl', 'wb') as fp:
				pkl.dump(self.masks, fp)

			with open(f'{path_stub}_token_type_ids.pkl', 'wb') as fp:
				pkl.dump(self.token_type_ids, fp)

			with open(f'{path_stub}_ratings.pkl', 'wb') as fp:
				pkl.dump(self.ratings, fp)

	def __len__(self):
		return self.n

	def __getitem__(self, idx):
		return self.ids[idx], self.masks[idx], self.token_type_ids[idx], self.ratings[idx]