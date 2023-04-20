import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import copy
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
        # instead of just having access to the users, we will send the data here
        user.set_parameters(server_model)


def aggregate_parameters(server_model, users, total_train_samples):
    # Clear global model before aggregation
    for param in server_model.parameters():
        param.data = torch.zeros_like(param.data)

    # i am confused here
    for user in users:
        for server_param, user_param in zip(server_model.parameters(), user.model.parameters()):
            server_param.data = server_param.data + user_param.data.clone() * user.train_samples / total_train_samples
    return server_model


def evaluate(users):
    total_accurancy = 0
    for user in users:
        total_accurancy += user.test()
    return total_accurancy / len(users)


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
num_global_iterations = 100

# creating the five clients here
# this is the part where instead, we will have the clients
# created as separate programs that connect to server
total_train_samples = 0
for i in range(1, num_user + 1):
    user = UserAVG(i, server_model, learning_rate, batch_size)
    users.append(user)
    total_train_samples += user.train_samples
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

    # the aggregate_parameters function does something
    # ngl i don't know what happens here but i feel like our algorithm should be different
    # im not sure if this is already using a subset of the users because we need to select a subset at random
    # this might just be using all of them to aggregate
    aggregate_parameters(server_model, users, total_train_samples)

"""
sum notes:
This is pretty much ripped straight from week 6 tutorial with a few changes to make it run
I understand most things apart from what aggregate is doing

** It is currently using Multinomial Logistic Regression as the classification method
It is specified that:
 To obtain the local model,
clients can use both Gradient Descent (GD) or Mini-Batch Gradient Descent (Mini-Batch GD)
as the optimization methods.
This shouldnt be to hard to change to

Also at the end we have to randomly select a subset of clients to perform the aggregation on
I believe that currently it just uses all the clients
This should be pretty ez

"""