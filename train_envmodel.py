import os
import sys
import tqdm
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from src.envs import all_envs
from src.utils.logger import Logger
from src.data.loaders import RolloutSequenceDataset
from src.models.pytorch import EnvModel
from src.models import all_envmodels, all_models, get_config

class Trainer():
	def __init__(self, make_env, config):
		self.dataset_train = RolloutSequenceDataset(config, train=True)
		self.dataset_test = RolloutSequenceDataset(config, train=False)
		self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=config.nworkers)
		self.test_loader = torch.utils.data.DataLoader(self.dataset_test, batch_size=config.batch_size, shuffle=False, num_workers=config.nworkers)

	def train_loop(self, ep, envmodel, update=1):
		batch_losses = []
		envmodel.network.train()
		with tqdm.tqdm(total=len(self.dataset_train)) as pbar:
			pbar.set_description_str(f"Train Ep: {ep}, ")
			for i,(states, actions, next_states, rewards, dones) in enumerate(self.train_loader):
				loss = envmodel.network.optimize(states, actions, next_states, rewards, dones).item()
				if i%update == 0:
					pbar.set_postfix_str(f"Loss: {loss:.4f}")
					pbar.update(states.shape[0]*update)
				batch_losses.append(loss)
		return np.mean(batch_losses)

	def test_loop(self, ep, envmodel):
		batch_losses = []
		envmodel.network.eval()
		with torch.no_grad():
			for states, actions, next_states, rewards, dones in self.test_loader:
				loss = envmodel.network.get_loss(states, actions, next_states, rewards, dones).item()
				batch_losses.append(loss)
		return np.mean(batch_losses)

def train(make_env, config):
	trainer = Trainer(make_env, config)
	envmodel = EnvModel(config.state_size, config.action_size, config, load="", gpu=True)
	checkpoint = f"{config.env_name}"
	logger = Logger(trainer, envmodel.network, config)
	ep_train_losses = []
	ep_test_losses = []
	for ep in range(config.epochs):
		train_loss = trainer.train_loop(ep, envmodel)
		test_loss = trainer.test_loop(ep, envmodel)
		ep_train_losses.append(train_loss)
		ep_test_losses.append(test_loss)
		envmodel.network.schedule(test_loss)
		if ep_test_losses[-1] <= np.min(ep_test_losses): envmodel.network.save_model(checkpoint)
		logger.log(f"Step: {ep:7d}, Reward: {ep_test_losses[-1]:9.3f} [{ep_train_losses[-1]:8.3f}], Avg: {np.mean(ep_test_losses, axis=0):9.3f} ({1.0:.3f})", envmodel.network.get_stats())

def parse_args(envs, models, envmodels):
	parser = argparse.ArgumentParser(description="MDRNN Trainer")
	parser.add_argument("env_name", type=str, choices=envs, help="Name of the environment to use. Allowed values are:\n"+', '.join(envs), metavar="env_name")
	parser.add_argument("envmodel", type=str, default=None, choices=envmodels, help="Which model to use as the dynamics. Allowed values are:\n"+', '.join(envmodels), metavar="envmodels")
	parser.add_argument("--model", type=str, default=None, choices=models, help="Which RL algorithm to use as the agent. Allowed values are:\n"+', '.join(models), metavar="model")
	parser.add_argument("--nworkers", type=int, default=0, help="Number of workers to use to load dataloader")
	parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train the envmodel")
	parser.add_argument("--seq_len", type=int, default=20, help="Length of sequence to train RNN")
	parser.add_argument("--batch_size", type=int, default=32, help="Size of batch to train RNN")
	parser.add_argument("--train_prop", type=float, default=0.9, help="Proportion of trajectories to use for training")
	return parser.parse_args()

if __name__ == "__main__":
	args = parse_args(all_envs, list(all_models.values())[0].keys(), all_envmodels)
	make_env, _, config = get_config(args.env_name, args.model)
	config.update(**args.__dict__)
	train(make_env=make_env, config=config)
		