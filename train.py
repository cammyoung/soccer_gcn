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
    train_block_adj, train_block_feat, train_block_pool, y_train, test_block_adj, test_block_feat, test_block_pool, y_test = pickle.load(f)

n_feat = train_block_feat.size()[1]
n_hid = 10
learning_rate = 0.01
weight_decay = 5e-4
dropout=False
n_epochs = 100

model = models.GCN(nfeat=n_feat,
                    nhid=n_hid,
                    nclass=3,
                    dropout=dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate, weight_decay=weight_decay)

def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    train_output = model(train_block_feat, train_block_adj, train_block_pool)
#     val_output = model(test_block_feat, test_block_adj, test_block_pool)
    loss_train = F.nll_loss(train_output, y_train)
    acc_train = util.accuracy(train_output, y_train)
    loss_train.backward()
    optimizer.step()

#     loss_val = F.nll_loss(val_output, y_test)
#     acc_val = accuracy(val_output, y_test)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
#           'loss_val: {:.4f}'.format(loss_val.item()),
#           'acc_val: {:.4f}'.format(acc_val.item()),
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