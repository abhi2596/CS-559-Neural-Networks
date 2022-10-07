import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.random.rand(300)
v = np.random.uniform(low=-0.1, high=0.1, size=300)
d = np.sin(20*x)+3*x+v

plt.scatter(x, d)
plt.xlabel("x_i")
plt.ylabel("d_i")
plt.title("Data Distribution")
plt.savefig("Q3.jpg")
plt.show()


class NeuralNetwork:
    def __init__(self, lr):
        np.random.seed(42)
        self.w_h = np.random.randn(24)
        self.w_o = np.random.randn(24)
        self.b_h = np.random.randn(24)
        self.b_o = np.random.randn(1)
        self.lr = lr
        self.dwo, self.dwh, self.dbh, self.dbo = np.zeros(24), np.zeros(24), np.zeros(24), np.zeros(1)

    def forward(self, x):
        a = self.w_h * x + self.b_h
        h = np.tanh(self.w_h * x + self.b_h)
        o = np.sum(self.w_o * h) + self.b_o
        return o, h, a

    def backward(self, o, h, a, x, y):
        self.dwo += (o-y) * h
        self.dwh += (o-y) * self.w_o * (1 - (np.tanh(a))**2) * x
        self.dbo += (o-y)
        self.dbh += (o-y) * self.w_o * (1 - (np.tanh(a))**2)

    def zero_grad(self):
        self.dwo, self.dwh, self.dbh, self.dbo = np.zeros(24), np.zeros(24), np.zeros(24), np.zeros(1)

    def weight_updates(self):
        self.w_o = self.w_o - self.lr * self.dwo
        self.w_h = self.w_h - self.lr * self.dwh
        self.b_o = self.b_o - self.lr * self.dbo
        self.b_h = self.b_h - self.lr * self.dbh

    def update_lr(self):
        self.lr /= 10


def square_loss(x, y):
    return 1/2 * ((x-y)**2)


def train_loop(model, data, target, loss_fn):
    o, h, a = model.forward(data)
    model.backward(o, h, a, data, target)
    loss = loss_fn(o, target)[0]
    model.weight_updates()
    model.zero_grad()
    return loss


epochs = 10000
model = NeuralNetwork(lr=0.01)
mean_squared_error = []
for epoch in range(0, epochs):
    loss_per_epoch = 0
    for i in range(0, 300):
        loss = train_loop(model, x[i], d[i], square_loss)
        loss_per_epoch += loss
    loss_per_epoch /= 300
    if epoch != 0 and loss_previous_epoch < loss_per_epoch:
        model.update_lr()
    loss_previous_epoch = loss_per_epoch
    mean_squared_error.append(loss_per_epoch)
    loss_previous_epoch = loss_per_epoch
print("Loss after 10000 epochs is ", loss_per_epoch)
plt.plot(range(0, epochs), mean_squared_error)
plt.xlabel("epochs")
plt.ylabel("mean_squared_error")
plt.title("Mean squared error vs Epochs")
plt.savefig("Q4.jpg")
plt.show()

fig, ax = plt.subplots()
blue = ax.scatter(x, d, color="blue")
for i in x:
    o = model.forward(i)[0][0]
    red = ax.scatter(i, o, color="red")
blue.set_label("true")
red.set_label("predicted")
ax.set_xlabel("x")
ax.set_ylabel("f(x,w),d")
ax.set_title("Curve Fitting")
ax.legend()
plt.savefig("Q5.jpg")
plt.show()


