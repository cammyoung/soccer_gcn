from __future__ import division
from __future__ import print_function
import util
import models

import time
import numpy as np
import pickle

import torch
import torch.nn.functional as F
import torch.optim as optim

pkl = "data.pkl"

with open(pkl, "rb") as f:
    block_adj, block_feat, block_pool, y, train_len, total_len = pickle.load(f)

n_feat = block_feat.size()[1]
n_class = y.max().item() + 1
n_hid = 10
learning_rate = 0.01
weight_decay = 5e-4
dropout=0.5
n_epochs = 100
train_idx = torch.LongTensor(range(train_len))
test_idx = torch.LongTensor(range(train_len, total_len))


model = models.GCN(nfeat=n_feat,
                    nhid=n_hid,
                    nclass=n_class,
                    dropout=dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate, weight_decay=weight_decay)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(block_feat, block_adj, block_pool)
    loss_train = F.nll_loss(output[train_idx], y[train_idx])
    acc_train = util.accuracy(output[train_idx], y[train_idx])
    loss_train.backward()
    optimizer.step()

    loss_val = F.nll_loss(output[test_idx], y[test_idx])
    acc_val = util.accuracy(output[test_idx], y[test_idx])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


# def test():
#     model.eval()
#     output = model(features, adj)
#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(n_epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# # Testing
# test()