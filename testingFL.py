import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
import random
import numpy as np
import ast
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt


class MCLR(nn.Module):
    def __init__(self):
        super(MCLR, self).__init__()
        # Create a linear transformation to the incoming data
        # Input dimension: 784 (28 x 28), Output dimension: 10 (10 classes)
        self.fc1 = nn.Linear(784, 10)

    # Define how the model is going to be run, from input to output
    def forward(self, x):
        # Flattens input by reshaping it into a one-dimensional tensor.
        x = torch.flatten(x, 1)
        # Apply linear transformation
        x = self.fc1(x)
        # Apply a softmax followed by a logarithm
        output = F.log_softmax(x, dim=1)
        return output


class UserAVG():
    def __init__(self, client_id, model, learning_rate, batch_size):

        self.X_train, self.y_train, self.X_test, self.y_test, self.train_samples, self.test_samples = get_data(
            client_id)
        self.train_data = [(x, y) for x, y in zip(self.X_train, self.y_train)]
        self.test_data = [(x, y) for x, y in zip(self.X_test, self.y_test)]
        self.trainloader = DataLoader(self.train_data, self.train_samples)
        self.testloader = DataLoader(self.test_data, self.test_samples)

        self.loss = nn.NLLLoss()

        self.model = copy.deepcopy(model)

        self.id = client_id

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)

    def set_parameters(self, model):
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            old_param.data = new_param.data.clone()

    def train(self, epochs):
        loss = 0
        self.model.train()
        for epoch in range(1, epochs + 1):
            self.model.train()
            for batch_idx, (X, y) in enumerate(self.trainloader):
                self.optimizer.zero_grad()
                print(len(X))
                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()
        return loss.data

    def test(self):
        self.model.eval()
        test_acc = 0
        for x, y in self.testloader:
            output = self.model(x)
            test_acc += (torch.sum(torch.argmax(output, dim=1) == y) * 1. / y.shape[0]).item()
            print(str(self.id) + ", Accuracy of client ", self.id, " is: ", test_acc)
        return test_acc


# THIS IS THE STUFF THAT WOULD ACTUALLY BE IN SERVER

def send_parameters(server_model, users):
    for user in users:
        # ok so; Challenge:
        # finding a way to get server model into string data so it can be converted into bytes
        # so far, the model has parameters (2 of them) that make up the model and
        # there is a way to convert parameter into binary data
        # process:
        # param -> torch tensor obj -> numpy array -> list -> string -> bytes
        # reconstructing:
        # string -> list -> np arr -> tensor -> param
        # now the problem,
        # we dont exactly know that the model has 2 parameters or their size
        # stuff like this is hard to re create in the client without the code to create the model
        # once the size and shape of the model is present within the client, it shouldnt be a problem
        # because then we can leech of the old size and shape as it wont change
        # idk what to do :(

        param_list = []
        for param in server_model.parameters():
            # from param to tensor to np arr
            npa = param.detach().numpy()
            # from np arr to list to str
            string_list = str(npa.tolist())
            # one step opt (bit cluttered)
            # string_list = str(param.detach().numpy().tolist())
            param_list.append(string_list)

            reconstructed_list = ast.literal_eval(string_list)
            reconstructed_npa = np.asarray(reconstructed_list)
            reconstructed_tensor = torch.Tensor(reconstructed_npa)
            reconstructed_param = torch.nn.parameter.Parameter(reconstructed_tensor, requires_grad=True)
            print(reconstructed_param)

        # here instead of sending server model, the goal is to send the list of strings
        # and have the client be able to create a model from this
        user.set_parameters(server_model)


def aggregate_parameters(server_model, users, total_train_samples):
    # Clear global model before aggregation
    for param in server_model.parameters():
        param.data = torch.zeros_like(param.data)

    for user in users:
        for server_param, user_param in zip(server_model.parameters(), user.model.parameters()):
            server_param.data = server_param.data + user_param.data.clone() * user.train_samples / total_train_samples
    return server_model


def evaluate(users):
    total_accurancy = 0
    for user in users:
        total_accurancy += user.test()
    return total_accurancy / len(users)


def select_subset(users: list) -> list:
    rand = random.randint(1, len(users))
    return random.sample(users, rand)


def get_data(id=""):
    train_path = os.path.join("FLdata", "train", "mnist_train_client" + str(id) + ".json")
    test_path = os.path.join("FLdata", "test", "mnist_test_client" + str(id) + ".json")
    train_data = {}
    test_data = {}

    with open(os.path.join(train_path), "r") as f_train:
        train = json.load(f_train)
        train_data.update(train['user_data'])
    with open(os.path.join(test_path), "r") as f_test:
        test = json.load(f_test)
        test_data.update(test['user_data'])

    X_train, y_train, X_test, y_test = train_data['0']['x'], train_data['0']['y'], test_data['0']['x'], test_data['0'][
        'y']
    X_train = torch.Tensor(X_train).view(-1, 1, 28, 28).type(torch.float32)
    y_train = torch.Tensor(y_train).type(torch.int64)
    X_test = torch.Tensor(X_test).view(-1, 1, 28, 28).type(torch.float32)
    y_test = torch.Tensor(y_test).type(torch.int64)
    train_samples, test_samples = len(y_train), len(y_test)
    return X_train, y_train, X_test, y_test, train_samples, test_samples


num_user = 5
users = []
server_model = MCLR()
batch_size = 20
learning_rate = 0.01
num_global_iterations = 1

# creating the five clients here
# this is the part where instead, we will have the clients
# created as separate programs that connect to server
for i in range(1, num_user + 1):
    user = UserAVG(i, server_model, learning_rate, batch_size)
    users.append(user)
    # in particular this send_parameters function is of interest
    # will have to be translated to compress and send data to each client
    # and then client decodes and reads this data
    send_parameters(server_model, users)

# clients created now we can run

loss = []
acc = []

for glob_iter in range(num_global_iterations):
    # here we have the send parameters again. final program will have altered send_parameters ofcors
    send_parameters(server_model, users)

    # each user evaluates the average acc of global model
    avg_acc = evaluate(users)
    acc.append(avg_acc)
    print("Global Round:", glob_iter + 1, "Average accuracy across all clients : ", avg_acc)

    avgLoss = 0
    for user in users:
        avgLoss += user.train(1)

    loss.append(avgLoss)
    chosen_ones = select_subset(users)
    chosen_train_sample_amount = 0
    for u in chosen_ones:
        chosen_train_sample_amount += u.train_samples
    aggregate_parameters(server_model, chosen_ones, chosen_train_sample_amount)


plt.figure(1, figsize=(5, 5))
plt.plot(acc, label="FedAvg", linewidth=1)
plt.ylim([0.5, 0.99])
plt.legend(loc='upper right', prop={'size': 12}, ncol=2)
plt.ylabel('Testing Acc')
plt.xlabel('Global rounds')
plt.show()

"""
sum notes:

If the randomly selected subset of clients is capable of being empty, it just
fucks up everything so be careful with that. 
Right now, it just selects a minimum of 1. 
i.e. 1 to 5 clients randomly chosen for aggregation.

It is hard to send the information of the server model:
Will have to find a way to define the size, shape of model and how each client can reproduce this
luckily the actual data within the model seems like it shouldnt be an issue to send from server to client
"""
