# cost function of logistic regression
## Vectorized Implementation of Optimization Using Gradient Descent
# Define Cost function
def cost(t, h, l=l, X=X, y=y, m=m):
    cost = np.transpose(-y) @ np.log(h) - np.transpose(1 - y) @ np.log(1 - h) + (l / 2) * np.transpose(t[1:]) @ t[1:]
    cost = (1 / m) * cost
    return cost


# Define first derivative of cost function
def cost_dev(j, t, X=X, y=y, m=m):
    dev = X[:, j] @ (1 / (1 + np.exp(-X @ theta)) - y)
    dev = (1 / m) * dev
    return dev


# Define iterations
cost_list = []
theta_temp = np.zeros(theta.shape)
theta_list = []
for i in range(1000000):

    for j in range(len(theta)):
        if j == 0:
            theta_temp[j] = theta[j] - a * cost_dev(j, theta)
        else:
            theta_temp[j] = theta[j] * (1 - (a * lmbd) / m) - a * cost_dev(j, theta)

    theta = theta_temp
    hypo = 1 / (1 + np.exp(-X @ theta))

    theta_list.append(list(theta))
    cost_val = cost(theta, hypo)
    cost_list.append(cost_val)