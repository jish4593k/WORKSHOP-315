import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from tkinter import Tk, Label, Button, StringVar, Entry

# Convert the numpy arrays to PyTorch tensors
train_inputs = Variable(torch.Tensor([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]]))
train_outputs = Variable(torch.Tensor([[0, 1, 0, 1]]).T)

# Define the neural network class
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layer = nn.Linear(3, 1)

    def forward(self, x):
        return torch.sigmoid(self.layer(x))

# Instantiate the neural network and define the loss and optimizer
model = NeuralNetwork()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training the neural network
for iteration in range(10000):
    # Forward pass
    output = model(train_inputs)
    
    # Compute loss
    loss = criterion(output, train_outputs)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Testing neural network
test_input = Variable(torch.Tensor([1, 1, 0]))
predicted_output = model(test_input).item()

print("Predicted output for testing input [1, 1, 0]: ", predicted_output)

# GUI setup for testing the neural network
class NeuralNetworkGUI:
    def __init__(self, master):
        self.master = master
        master.title("Neural Network GUI")

        self.label = Label(master, text="Enter input for testing:")
        self.label.pack()

        self.entry = Entry(master)
        self.entry.pack()

        self.result_var = StringVar()
        self.result_label = Label(master, textvariable=self.result_var)
        self.result_label.pack()

        self.test_button = Button(master, text="Test", command=self.test_network)
        self.test_button.pack()

    def test_network(self):
        input_str = self.entry.get()
        input_list = [float(x) for x in input_str.split(',')]
        test_input = Variable(torch.Tensor(input_list))
        predicted_output = model(test_input).item()
        self.result_var.set(f"Predicted Output: {predicted_output:.2f}")


# Create GUI window
root = Tk()
neural_network_gui = NeuralNetworkGUI(root)
root.mainloop()
