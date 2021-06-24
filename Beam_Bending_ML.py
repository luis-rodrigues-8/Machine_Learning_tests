import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt


# Define the functions that build and train the model

def create_model(my_learning_rate, my_feature_layer):
    # Simple tf.keras. Will try out more complex models in the future.
    model = tf.keras.models.Sequential()

    # Add the layer containing the feature columns to the model.
    model.add(my_feature_layer)

    # Describe the topography of the model by calling the tf.keras.layers.Dense()

    # Define the first hidden layer with 20 nodes.
    model.add(tf.keras.layers.Dense(units=20,
                                    activation='relu',
                                    name='Hidden1'))

    # Define the second hidden layer with 12 nodes.
    model.add(tf.keras.layers.Dense(units=12,
                                    activation='relu',
                                    name='Hidden2'))

    # Define the output layer.
    model.add(tf.keras.layers.Dense(units=1,
                                    name='Output'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    return model


def train_model(model, dataset, epochs, label_name,
                batch_size=None):
    
    # Split the dataset into features and label.
    # Didn't fully understand this next line yet (copy pasted it) and it causes some problems. Will try to find out a better way to do it.
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True)

    epochs = history.epoch

    # Gather a snapshot of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)
    mse = hist["mean_squared_error"]

    return epochs, mse


# Define the plotting function for the loss curves

def plot_the_loss_curve(epochs, mae_training):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
    plt.legend()

    merged_mae_lists = mae_training[1:]
    highest_loss = max(merged_mae_lists)
    lowest_loss = min(merged_mae_lists)
    delta = highest_loss - lowest_loss
    print(delta)

    top_of_y_axis = highest_loss + (delta * 0.05)
    bottom_of_y_axis = lowest_loss - (delta * 0.05)

    plt.ylim([bottom_of_y_axis, top_of_y_axis])
    plt.show()

    
# Define the analytical solution

def ana_sol(x, q):
    y = np.zeros(len(x))
    for i in range(0, len(x) - 1):
        y[i] = -x[i]**5 + 2*x[i]**4   # Will put here the equations for the deflection of the beam 

    return y


# Create fictional points

x_min = -25
x_max = 25
n = 1000  # number of points
x_vec = np.linspace(x_min, x_max, n)
y_vec = ana_sol(x_vec, 10 ** 8)

# Add noise
noise_factor = 0.1
y_vec = y_vec + noise_factor * (np.random.random(len(y_vec)) - 0.5) * (np.max(y_vec) - np.min(y_vec))

points = np.zeros((len(x_vec), 2))

for i in range(len(x_vec)):
    points[i, 0] = x_vec[i]
    points[i, 1] = y_vec[i]

# Create a Pandas Data Frame with the points. Still need to find out if this is necessary in this case.
df = pd.DataFrame(data=points, columns=['x_vec', 'y_vec'])
df = df.reindex(np.random.permutation(df.index))  # shuffle the points

# Append x_vec to feature columns. 
feature_columns = []
x_col = tf.feature_column.numeric_column("x_vec")
feature_columns.append(x_col)

# Convert the list of feature columns into a layer
my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# Tune the hyperparameters of the model
learning_rate = 0.2
epochs = 300
batch_size = 100

# Specify the label
label_name = "y_vec"

# Establish the model's topography.
my_model = create_model(learning_rate, my_feature_layer)

# Train the model on the x_vec
epochs, mse = train_model(my_model, df, epochs,
                          label_name, batch_size)
plot_the_loss_curve(epochs, mse)

# Plot a sample of the fictional points and the ML solution curve
sample_df = df.sample(int(n/5))
plt.scatter(sample_df['x_vec'], sample_df['y_vec'])

df['y_vec'] = np.zeros(len(y_vec))  # To make sure my_model.predict isn't using the y_vec values in any way.
# print(df)
# print({name: np.sort(np.array(value)) for name, value in df.items()})

# Had some problems here, because my_model.predict only takes a 'dict' as argument and not a np.array
# Probably related to my problems in the function train_model.
plt.plot(x_vec, my_model.predict({name: np.sort(np.array(value)) for name, value in df.items()}), c='r')
plt.show()

# It now gives a warning because the 'dict' has two inputs, but I can't seem to make it work right yet.
