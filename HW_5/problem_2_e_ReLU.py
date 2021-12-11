import numpy as np
import matplotlib.pyplot as plt
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable

act_funcs = {'TANH':nn.Tanh(), 
             'RELU':nn.ReLU()}

class Net(nn.Module):
    

    def __init__(self, config, act='TANH'):
        
        super(Net, self).__init__()
        
        layers_list = []

        for i in range(len(config)-2):

            in_dim = config[i]
            out_dim = config[i+1]

            layers_list.append(nn.Linear(in_features=in_dim, 
                                            out_features=out_dim))
            layers_list.append(act_funcs[act.upper()])

            if act.upper() == 'TANH':
                nn.init.xavier_normal_(layers_list[-2].weight)
            if  act.upper() == 'RELU':
                nn.init.kaiming_uniform_(layers_list[-2].weight)

        layers_list.append(nn.Linear(in_features=config[-2],
                                        out_features=config[-1]))

        self.net = nn.ModuleList(layers_list)
        
    def forward(self, x):
        
        for layer in self.net:
            x = layer(x)
        
        return x

class dataset(Dataset):
    
    def __init__(self, x, y):
        super(dataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index, :], self.y[index]
    
    def __len__(self,):
        return self.x.shape[0]



'''
Load in data
'''

train_data = np.genfromtxt('bank-note/train.csv', delimiter=',')
test_data = np.genfromtxt('bank-note/test.csv', delimiter=',')

x_train = train_data[:,:-1]
y_train = train_data[:,-1].flatten()
y_train[np.where(y_train == 0)] = -1

x_test = test_data[:,:-1]
y_test = test_data[:,-1].flatten()
y_test[np.where(y_test == 0)] = -1

train_dataset = dataset(x_train, y_train)
test_dataset = dataset(x_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=len(train_dataset),
        shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset),
        shuffle=False)


configs = []

for depth in [3,5,9]:
    for width in [5,10,25,50,100]:

        configs.append([x_train.shape[1]] + depth*[width] + [1])

models = [Net(config, act='relu') for config in configs]

'''
run the NN
'''
class run:

    def __init__(self, model):
        self.model = model

    def __call__(self, train_loader, test_loader):
        print(self.model)
        epochs = 20

        optimizer = optim.Adam(self.model.parameters())
        
        test_error = []
        train_error = []
        idxs = np.arange(train_loader.dataset.x.shape[0])

        for ie in tqdm.tqdm(range(epochs+1)):

            Xtr, ytr = next(iter(train_loader))
            Xte, yte = next(iter(test_loader))
            Xtr, ytr, Xte, yte = Xtr.float(), \
                                 ytr.float(), \
                                 Xte.float(),\
                                 yte.float() 
            np.random.shuffle(idxs)

            for i in idxs:
                
                pred = self.model(Xtr[i])
                loss = torch.sum(torch.square(pred-ytr[i]) * 0.5)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            pred = self.model(Xtr)
            loss = torch.sum(torch.square(pred-ytr) * 0.5)
            pred = np.sign(pred.detach().numpy())
            pred[np.where(pred==0)] = -1
            test_pred = self.model(Xte)
            test_pred = np.sign(test_pred.detach().numpy())
            test_pred[np.where(test_pred==0)] = -1
            with torch.no_grad():
                tre = np.count_nonzero(ytr-pred.flatten()) / ytr.shape[0]
                tee = np.count_nonzero(yte-test_pred.flatten()) / yte.shape[0]

        self.train_error = tre
        self.test_error = tee


NNs = []
for model in models:
    NNs.append(run(model))
    NNs[-1](train_loader, test_loader)
    print(f'Train Error %: {NNs[-1].train_error*100}')
    print(f'Test Error %: {NNs[-1].test_error*100}')

