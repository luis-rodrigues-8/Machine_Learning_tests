import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

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

# Define the functions that will train and build the model

def train_model(model, dataset, epochs, label_name,
                batch_size=None):
    # Split the dataset into features and label.
    # Didn't fully understand this next line yet (copy pasted it). Will try to find out a better way to do it.
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True)

    epochs = history.epoch

    # Gather a snapshot of the model's mean squared error at each epoch.
    hist = pd.DataFrame(history.history)
    mse = hist["mean_squared_error"]

    return epochs, mse


def create_model(my_learning_rate, my_feature_layer):
    # Simple tf.keras. Will try out more complex models in the future.
    model = tf.keras.models.Sequential()

    # Add the layer containing the feature columns to the model.
    model.add(my_feature_layer)

    model.add(tf.keras.layers.Dense(units=256,
                                    activation='relu',
                                    name='Hidden1'))

    # tf.keras.activations.tanh is another possible activation function

    model.add(tf.keras.layers.Dense(units=256,
                                    activation='relu',
                                    name='Hidden2'))

    model.add(tf.keras.layers.Dense(units=1,
                                    name='Output'))

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    return model

# Define the analytical solution
# Simple beam (pinned support and roller support on each end) with length l, uniform load q
# moment of inertia of its section I and Young's modulus E.


def ana_sol(x, q, l, E, I):
    y = np.zeros(len(x))
    c_1 = -(q * l**3) / (24*E*I)
    y = (1/(E*I)) * (((q*l)/12) * x**3 - (q/24) * x**4) + c_1 * x

    return y


# Create fictional points

x_min = 0
x_max = 1
n = 5000  # number of points
x_vec = np.linspace(x_min, x_max, n)

l = x_max - x_min
q = 10 * 10**4
E = 210 * 10**9
I = 2 * 10**(-5)
y_vec = ana_sol(x_vec, q, l, E, I) * 10**3  # [mm]

# Add noise
noise_factor = 0.05
y_vec = y_vec + noise_factor * (np.random.random(len(y_vec)) - 0.5) * (np.max(y_vec) - np.min(y_vec))

points = np.zeros((len(x_vec), 2))

for i in range(len(x_vec)):
    points[i, 0] = x_vec[i]
    points[i, 1] = y_vec[i]

# Create a Pandas Data Frame with the points
df = pd.DataFrame(data=points, columns=['x_vec', 'y_vec'])
df = df.reindex(np.random.permutation(df.index))  # shuffle the points

# Append x_vec to feature columns. Still need to find out if this is completely necessary.
feature_columns = []
x_col = tf.feature_column.numeric_column("x_vec")
feature_columns.append(x_col)

# Convert the list of feature columns into a layer
my_feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


# Tune the hyperparameters of the model
learning_rate = 0.01
epochs = 80
batch_size = int(n/2)

label_name = "y_vec"

my_model = create_model(learning_rate, my_feature_layer)

epochs, mse = train_model(my_model, df, epochs,
                          label_name, batch_size)
plot_the_loss_curve(epochs, mse)

# Plot a sample of the fictional points and the ML solution curve
sample_df = df.sample(int(n/5))
plt.scatter(sample_df['x_vec'], sample_df['y_vec'])

df['y_vec'] = np.zeros(len(y_vec))

# Had some problems here, because my_model.predict only takes a 'dict' as argument and not a np.array
# Probably related to my problems in the function train_model.
y_sol = my_model.predict({name: np.sort(np.array(value)) for name, value in df.items()})
plt.plot(x_vec, y_sol, c='r')
plt.show()

# It now gives a warning because the 'dict' has two inputs, but I can't seem to make it work right yet.


# We will now estimate the Young's modulus of the beam using the ML model.
# This is done by calculating the average of estimated Es for each point (x_vec[i], y_sol[i]).
# We consider that the load q and the moment of inercia I are known.

sum_E = 0
for i in range(n):
    sum_E = sum_E + (((q * l) / 12) * x_vec[i] ** 3 - (q / 24) * x_vec[i] ** 4 - (q * l**3 * x_vec[i]) / 24) / (y_sol[i] * 10**(-3) * I)

avg_E = sum_E/n
print("Estimated E: ", avg_E)
print("True E: ", E)
print("Relative error: ", abs(avg_E - E)/E)
print('\n')

# Now the same for the moment of inertia, considering E known.

sum_I = 0
for i in range(n):
    sum_I = sum_I + (((q * l) / 12) * x_vec[i] ** 3 - (q / 24) * x_vec[i] ** 4 - (q * l**3 * x_vec[i]) / 24) / (y_sol[i] * 10**(-3) * E)

avg_I = sum_I/n
print("Estimated Iz: ", avg_I)
print("True Iz: ", I)
print("Relative error: ", abs(avg_I - I)/I)
