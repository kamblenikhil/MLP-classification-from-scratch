# Assignment 4 - Machine Learning

## Part 1: K-Nearest Neighbors Classification

### What is K-Nearest Neighbors (KNN)
K-Nearest Neighbors evaluates the labels of a certain number of data points surrounding a target data point in order to forecast which class the data point belongs to. <br>

When a KNN algorithm is run, it passes through three basic stages:

1. Set K to the number of neighbors you want.
2. The distance between a provided/test example and the dataset examples is calculated.
3. The calculated distances are sorted.
4. Obtaining the top K entries' labels.
5. A prediction about the test example is returned.

![image](https://media.github.iu.edu/user/18330/files/64f1de5a-ad92-4003-8599-43c3bcaf66a5)

Here, we test the given data by calculating Manhattan and Euclidean distance as per the metrics are passed to the knn function using the fitted classifier model. 

```python3
    # find distance: if l1 ->manhattan, else -> euclidean
    if self.metric == "l1":
      distance = manhattan_distance(self._X[j], X[i])
    elif self.metric == "l2":
      distance = euclidean_distance(self._X[j], X[i])
    
    d.append([distance, self._y[j]])
```

<b>Euclidean Distance</b>
```python3
    eu_distance = np.sqrt(((x1-x2)**2).sum())
```

<b>Manhattan Distance</b>
```python3
    man_distance = np.abs(x1 - x2).sum()
```

Once we get all the distances, we sort the array and consider n_neighbors (3, 5, 7, 11) to then weigh the class considering the weights as uniform and distance. After weighing the class, we then append the maximum value from the given dictionary to our final prediction list.

```python3

# sorting the distance array
d.sort()

# consider only n neighbors
for x in range(0,self.n_neighbors):
    a.append(d[x])

# weighing the class
for dist, clust in a:
    # considering weights as uniform and distance 
    if self.weights == "uniform":
        if clust not in temp:
            temp[clust] = 1
        else:
            temp[clust] = temp[clust] + 1
    # the below code was suggested by stephen
    elif self.weights == "distance":
        if clust not in temp:
            temp[clust] = float(1/dist)
        else:
            temp[clust] = temp[clust] + float(1/dist)

# appending the max from the given temp dictionary
prediction.append(max(temp,key=temp.get))
```

<hr>

## Part 2: Multilayer Perceptron Classification

### What is Multi-Layered Perceptron NN
Between the input and output layers of a Multi-Layered Perceptron NN, there can be an n-number of hidden layers. These hidden layers can have an n-number of neurons, with the first hidden layer receiving information from the input layer and processing it using the activation function before passing it on to the successive hidden levels until it reaches the output layer. A non-linear activation function is used by every neuron in a buried layer. Backpropagation is a supervised learning technique used by MLP during training.

![image](https://media.github.iu.edu/user/18330/files/bfca21cd-c913-4848-9aa4-6b5982887db8)

### Forward Propagation
We’ll compute the unactivated values of the nodes in the first hidden layer by applying W[1] and b[1] to our input layer. We’ll call the output of this operation Z[1]:<br>

![image](https://media.github.iu.edu/user/18330/files/deefc939-3aeb-4b40-ba30-00c3226bac61)

### Backward propagation

1. Initialize weights 
2. for each Xi,<br>
  i. pass Xi forward through the network<br>
  ii. calculate loss<br>
  iii. compute derivative and update weights.<br>
3. Repeat step 2 until convergence.

### Activation Function

<b>1. Identity Activation Function</b>

```python3
    if derivative == False:
        return x
    return np.ones(shape=x.shape, dtype=np.float64)
```

<b>2. Sigmoid Activation Function</b>

```python3
    act_func = 1.0 / (1.0 + np.exp(-x))

    if derivative == False:
        return act_func
    return act_func * (1.0 - act_func)
```

<b>3. Tanh Activation Function</b>

```python3
    act_func = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

    if derivative == False:
        return act_func
    return 1-(act_func)**2
```

<b>4. ReLU Activation Function</b>

```python3
    if derivative == False:
        return 1.0 * (x > 0)
    return x * (x > 0)
```

<b>5. Cross Entropy Function</b>

```python3
    return -np.sum(y * np.log(p + 1e-9)) / len(y)
```

<b>6. One Hot Encoding</b>

```python3
    one_hot = np.zeros(shape=(len(y), len(set(y))), dtype=np.int64)
    yT = np.array([y]).T

    for idx, row in enumerate(yT):
        one_hot[idx, row[0]] = 1
    return one_hot
```
