import random
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm

@dataclass
class PreprocessDatasetCollator:
    """
    This data collator is used to preprocess and perform minibatching on the dataset ahead of time.
    By doing it before training, we spend a little extra time upfront, but it speeds up training from ~6it/s to ~90it/s.
    """

    return_tensors: str = "pt"
    max_len: int = 10  # subsets of the episode we use for training
    state_dim: int = 17  # size of state space
    act_dim: int = 6  # size of action space
    max_ep_len: int = 1000  # max episode length in the dataset
    scale: float = 1.0  # normalization of rewards/returns, we don't use this for backgammon
    #state_mean: np.array = None  # to store state means, we don't use this for backgammon
    #state_std: np.array = None  # to store state stds, we don't use this for backgammon
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0  # to store the number of trajectories in the dataset
    avg_traj_len: int = 0  # to store the average trajectory length in the dataset

    def __init__(self, dataset) -> None:
        self.act_dim = len(dataset[0]["actions"][0])
        self.state_dim = len(dataset[0]["observations"][0])
        self.dataset = dataset
        # calculate dataset stats for normalization of states
        #states = []
        traj_lens = []
        for obs in tqdm(dataset["observations"]):
            #states.extend(obs)
            traj_lens.append(len(obs))
        self.avg_traj_len = np.mean(traj_lens)
        self.n_traj = len(traj_lens)
        #states = np.vstack(states)
        #self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        traj_lens = np.array(traj_lens)
        self.p_sample = traj_lens / sum(traj_lens)

    @staticmethod
    def _discount_cumsum(x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_size = len(features)
        # this is a bit of a hack to be able to sample of a non-uniform distribution
        batch_inds = np.random.choice(
            np.arange(self.n_traj),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )
        # a batch of dataset features
        s, a, r, d, rtg, timesteps, mask, labels = [], [], [], [], [], [], [], []

        for ind in batch_inds:
            # for feature in features:
            feature = self.dataset[int(ind)]
            si = random.randint(0, len(feature["rewards"]) - 1)

            # get sequences from dataset
            s.append(np.array(feature["observations"][si: si + self.max_len]).reshape(1, -1, self.state_dim))
            a.append(np.array(feature["actions"][si: si + self.max_len]).reshape(1, -1, self.act_dim))
            r.append(np.array(feature["rewards"][si: si + self.max_len]).reshape(1, -1, 1))

            d.append(np.array(feature["dones"][si: si + self.max_len]).reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                self._discount_cumsum(np.array(feature["rewards"][si:]), gamma=1.0)[
                : s[-1].shape[1]  # TODO check the +1 removed here
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                print("if true")
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            #s[-1] = (s[-1] - self.state_mean) / self.state_std
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, self.max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))

            # labels are just the actions
            labels.append(a[-1].copy())

            # replace the last action with all zeros (respecting padding)
            a[-1][:, -1, :] = np.zeros((1, self.act_dim))

        s = torch.from_numpy(np.concatenate(s, axis=0)).float().cuda()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float().cuda()  # we don't want to convert actions to floats because we do that later on in the model
        r = torch.from_numpy(np.concatenate(r, axis=0)).float().cuda()
        d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float().cuda()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long().cuda()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float().cuda()
        labels = torch.from_numpy(np.concatenate(labels, axis=0)).float().cuda()

        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
            "labels": labels,
        }



class UnwrapCollator:
    """The dataset is already batched, but HuggingFace needs us to provide a batch size >= 1,
    so just provide this and a batch size of 1."""
    return_tensors: str = "pt"

    def __call__(self, features):
        # in this case, features should only ever be a single index because we've already created and collated the dataset ahead of time
        return features[0]


class DecisionTransformerPreprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, batch_size, max_len):
        """
        Dataset should follow a format that looks something like: https://huggingface.co/datasets/edbeeching/decision_transformer_gym_replay
        Basically, it should have an 'actions', 'observations, 'rewards', and 'dones' key.
        batch_size: the size of each preprocessed batch
        max_len: the maximum length of each sequence

        """
        self.batch_size = batch_size
        self.max_len = max_len

        self.states = []
        self.actions = []
        self.rewards = []
        self.rtgs = []
        self.timesteps = []
        self.attention_mask = []
        self.labels = []

        self._fill_dataset(dataset)

        self.state_dim = self.states[0].shape[-1]
        self.act_dim = self.actions[0].shape[-1]

    def _fill_dataset(self, dataset):
        preprocessor_collator = PreprocessDatasetCollator(dataset)
        preprocessor_collator.batch_size = self.batch_size
        preprocessor_collator.max_len = self.max_len

        # we will preprocess and batch the dataset with a dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, collate_fn=preprocessor_collator, shuffle=True)

        # each sequence in a batch is only a random subset of the whole trajectory
        # so we'll loop through the dataloader a couple of times to randomly sample enough to approximate the whole dataset
        dataset_iterations = int(np.ceil(preprocessor_collator.avg_traj_len / self.max_len))
        total_batches = dataset_iterations * len(dataloader)

        with tqdm(total=total_batches, desc='Preprocessing dataset', unit='batch') as pbar:
            for _ in range(dataset_iterations):
                for batch in dataloader:
                    self.states.append(batch["states"])
                    self.actions.append(batch["actions"])
                    self.rewards.append(batch["rewards"])
                    self.rtgs.append(batch["returns_to_go"])
                    self.timesteps.append(batch["timesteps"])
                    self.attention_mask.append(batch["attention_mask"])
                    self.labels.append(batch["labels"])
                    pbar.update(1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            "states": self.states[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "returns_to_go": self.rtgs[idx],
            "timesteps": self.timesteps[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }