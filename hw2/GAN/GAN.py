import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist

learning_rate = tf.Variable(0.001, trainable=False)
tf.keras.backend.set_value(learning_rate, 0.00025)

(X_train, _), (_, _) = mnist.load_data()
X_train = X_train.astype(np.float32) / 255.0
X_train = X_train.reshape(-1, 28 * 28)

def generator_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=100),
        tf.keras.layers.Dense(28 * 28, activation='tanh'),
        tf.keras.layers.Reshape((28, 28))
    ])
    return model

def discriminator_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

generator = generator_model()
discriminator = discriminator_model()
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

discriminator.trainable = False
gan_input = tf.keras.layers.Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

def plot_generated_images(generator, epoch, examples=10, dim=(1, 10), figsize=(10, 1)):
    noise = np.random.normal(0, 1, (examples, 100))
    generated_images = generator.predict(noise)
    
    plt.figure(figsize=figsize)
    for i in range(examples):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], cmap='gray_r')
        plt.axis('off')
    plt.suptitle(f'Epoch: {epoch}')
    plt.savefig(f'simple_gen_im/{epoch}.png')
    plt.clf()

epochs = 1000
batch_size = 128
d_losses, g_losses = [], []

for epoch in range(epochs):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_images = X_train[idx]
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_images = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
    d_losses.append(0.5 * (d_loss_real + d_loss_fake))

    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))
    g_losses.append(g_loss)

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, D Loss: {d_losses[-1]}, G Loss: {g_loss}')
        plot_generated_images(generator, epoch)

plt.figure(figsize=(10,10))
plt.plot(d_losses, label='D Loss')
plt.plot(g_losses, label='G Loss')
plt.legend()
plt.savefig('loss_function_plot.png')

generate_and_display_images = lambda num_images=25: plot_generated_images(generator, 'Final', num_images, (5, 5), (10, 10))
generate_and_display_images()
