import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from transformers import BertTokenizer

from datasets.imdb_datasets import IMDB_Dataset
from models.bert import BERT_Model


# constants
batch_size = 16
learning_rate = 1e-6
max_epochs = 100
window_size = 256
load_from_file = True
progress_bar = True
train_model = False
load_model = True


def main():
	device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
	print(f"using device: {device}")

	# create tokenizer
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	# generate/load datasets
	train_dataset = IMDB_Dataset(
		'train',
		window_size,
		tokenizer,
		load_from_file = load_from_file
	)

	dev_dataset = IMDB_Dataset(
		'dev',
		window_size,
		tokenizer,
		load_from_file = load_from_file
	)

	test_dataset = IMDB_Dataset(
		'test',
		window_size,
		tokenizer,
		load_from_file = load_from_file
	)

	# create dataloaders
	train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)
	dev_loader = DataLoader(dev_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)
	test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 2)

	# create model and send to gpu
	model = BERT_Model(128, p = 0.5)
	model.to(device)

	# initialize optimizer
	optimizer = optim.Adam(model.parameters(), lr = learning_rate)

	# initialize loss function
	loss_func = nn.MSELoss()

	if load_model:
		model.load_state_dict(torch.load('checkpoint3.pth')['state_dict'])
		print('loaded model')

	if train_model:
		# TRAINING/VALIDATION LOOP
		# prev_loss is used to store validation losses -> training is stopped
		# once validation loss is above a 5-epoch rolling mean
		prev_loss = []

		# iterate for specified number of epochs
		for epoch in range(max_epochs):
			model.train()
			sum_loss = 0
			for batch_idx, (ids, masks, token_type_ids, targets) in enumerate(tqdm(train_loader, disable = not progress_bar, desc = f'Epoch {epoch:02d}')):
				# send tensors to device
				ids, masks, token_type_ids, targets = ids.to(device).long(), masks.to(device).long(), token_type_ids.to(device).long(), targets.to(device)

				# zero out gradients
				optimizer.zero_grad()

				# forward pass
				preds = model(ids, masks, token_type_ids).squeeze().double()

				# calculate loss
				loss = loss_func(preds, targets)
				sum_loss += loss.item()

				# backward pass
				loss.backward()

				# step optimizer
				optimizer.step()

			print(f'\tTrain loss =      {sum_loss/(batch_idx+1)/batch_size:.6f}')

			# validation loop
			model.eval()
			valid_loss = 0
			with torch.no_grad():
				for batch_idx, (ids, masks, token_type_ids, targets) in enumerate(dev_loader):
					# send tensors to device
					ids, masks, token_type_ids, targets = ids.to(device).long(), masks.to(device).long(), token_type_ids.to(device).long(), targets.to(device)

					# forward pass
					preds = model(ids, masks, token_type_ids).squeeze().double()

					# calculate loss
					loss = loss_func(preds, targets)
					valid_loss += loss.item()

			# append current loss to prev_loss list
			prev_loss.append(valid_loss/(batch_idx+1)/batch_size)

			print(f'\tValidation loss = {valid_loss/(batch_idx+1)/batch_size:.6f}')

			# # if valid_loss exceedes the 5-epoch rolling sum, break from training
			if valid_loss/(batch_idx+1)/batch_size > np.mean(prev_loss[-5:]):
				# break
				continue

			checkpoint = {
				'epoch': epoch,
				'state_dict': model.state_dict(),
				'optimizer': optimizer.state_dict(),
			}
			torch.save(checkpoint, 'checkpoint3.pth')

	# TESTING
	p = []
	l = []
	test_loss = 0
	model.eval()
	with torch.no_grad():
		for batch_idx, (ids, masks, token_type_ids, targets) in enumerate(tqdm(test_loader, disable = not progress_bar, desc = f'Testing')):
			# send tensors to device
			ids, masks, token_type_ids, targets = ids.to(device).long(), masks.to(device).long(), token_type_ids.to(device).long(), targets.to(device)

			# forward pass
			preds = model(ids, masks, token_type_ids).squeeze().double()
			p.append(preds.cpu().numpy())
			l.append(targets.cpu().numpy())

			# calculate loss
			loss = loss_func(preds, targets)
			test_loss += loss.item()

	print(f'\tTest loss = {test_loss/(batch_idx+1)/batch_size:.6f}')

	# collate results
	p = np.array(p).flatten()
	l = np.array(l).flatten()
	diff = np.abs(p - l)

	results = np.array([p, l, diff])

	with open('results.pkl', 'wb') as fp:
		pkl.dump(results, fp)

	# print basic stats
	print(f'\tAverage Difference: {np.mean(diff):.6f}')
	print(f'\tStd. Dev. of Differences: {np.std(diff):.6f}')

	# scatter plot of predicted vs. results
	plt.scatter(p, l)
	plt.plot(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
	plt.xlabel('predicted')
	plt.ylabel('actual')
	plt.show()
	plt.clf()

	# histogram of predicted vs. results
	plt.hist([np.around(p), np.around(l)], label = ['predicted', 'actual'])
	plt.legend(loc = 'upper right')
	plt.ylim(0, 900)
	plt.show()
	plt.clf()

if __name__ == '__main__':
	main()