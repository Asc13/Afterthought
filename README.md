# Afterthought

Tutorial for a simple Aftertought utilization 
### 1. Load a model
```python
model = Wrapper(model = 'InceptionV3')
```

### 2. Create a parameterization
```python
parameterization = Parameterization.image_fft(size = 512)
```

### 3. Create a objective
```python
layer = 'mixed5'
channel = 16

objective = Objective.channel(model = model, layer = layer, channels = channel)
```

### 4. Run the activation maximization
```python
images = run(objective = objective, parameterization = parameterization,
             learning_rate = 0.1, steps = 1000, verbose = False)

plot_all(images)
```

# Tutorials

For a better understanding of the content of the toolkit, Afterthought has a section with tutorials on the various components.

[Activation maximization](https://github.com/Asc13/Afterthought/blob/main/Tutorials/activation_alternatives.ipynb)

[Parameterization](https://github.com/Asc13/Afterthought/blob/main/Tutorials/parameterization.ipynb)

[Objectives](https://github.com/Asc13/Afterthought/blob/main/Tutorials/objectives.ipynb)

[Regularization](https://github.com/Asc13/Afterthought/blob/main/Tutorials/regularization.ipynb)

[Transformations](https://github.com/Asc13/Afterthought/blob/main/Tutorials/transformations.ipynb)

[Activation alternatives](https://github.com/Asc13/Afterthought/blob/main/Tutorials/activation_alternatives.ipynb)
