import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from scipy.stats import entropy
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input

# Load and preprocess the MNIST dataset
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

def preprocess_images(images):
    images = tf.convert_to_tensor(images)  # Convert NumPy array to TensorFlow tensor
    images = tf.image.grayscale_to_rgb(images)  # Convert grayscale to RGB
    images = tf.image.resize(images, (299, 299))
    images = tf.keras.applications.inception_v3.preprocess_input(images)
    return images

# Function to calculate FID
def calculate_FID(real_images, generated_images):
    inception = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    real_features = inception.predict(preprocess_images(real_images))
    generated_features = inception.predict(preprocess_images(generated_images))
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
    diff = np.sum((mu_real - mu_gen)**2) + np.trace(sigma_real + sigma_gen - 2 * sqrtm(np.dot(sigma_real, sigma_gen)))
    return np.sqrt(diff)

# Function to calculate IS
def calculate_IS(generated_images, n_split=10):
    scores = []
    for i in range(n_split):
        part = generated_images[i * (generated_images.shape[0] // n_split):(i + 1) * (generated_images.shape[0] // n_split), :, :, :]
        p_yx = np.mean(part, axis=0)
        p_y = np.mean(p_yx, axis=0)
        kl_d = np.mean([entropy(p_yx[i], p_y) for i in range(p_yx.shape[0])])
        scores.append(np.exp(kl_d))
    return np.mean(scores), np.std(scores)

# Generator model
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model

# Discriminator model
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Create the models
generator = build_generator()
discriminator = build_discriminator()

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Training loop
EPOCHS = 50
BATCH_SIZE = 256
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Training the GAN
def train(dataset, epochs):
    for epoch in range(epochs):
        for batch in dataset:
            train_step(batch)

        generated_images = generator(tf.random.normal([num_examples_to_generate, noise_dim]), training=False)

        inception_score_mean, inception_score_std = calculate_IS(generated_images)
        fid_score = calculate_FID(train_images[:num_examples_to_generate], generated_images)

        print(f"Epoch {epoch + 1}/{epochs} - IS: {inception_score_mean} ± {inception_score_std}, FID: {fid_score}")

# Create a dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(train_images.shape[0]).batch(BATCH_SIZE)

# Train the GAN
train(train_dataset, EPOCHS)

# Generate images using the trained generator
def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

# Generate images
generate_and_save_images(generator, EPOCHS, seed)
