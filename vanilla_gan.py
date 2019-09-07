import tensorflow as tf
import tensorflow_datasets as tfds

class DataFeed:
    def __init__(self, dataset_name='mnist'):
        self.dataset = tfds.load(name="mnist", split=tfds.Split.TRAIN)

    def get(self):
        pass

    def get_batch(self):
        pass

class VanillaGAN:
    latent_dim = 100
    image_shape = (28, 28, 1)
    image_dim = 784

    def __init__(self):
        self.latent_vector = tf.placeholder(dtype=tf.float32, shape=(None, self.latent_dim))
        self.generated_images = self._build_generator()

        self.true_images =  tf.placeholder(dtype=tf.float32, shape=(None, self.image_dim))

        self.true_images_switch = tf.placeholder(dtype=tf.float32, shape=(None,))

        self.discriminator_input = self.true_images_switch * self.true_images +\
            (1. - self.true_images_switch) * self.generated_images

        self._build_discriminator()

        self._build_loss()

    def _build_generator(self):
        return 0

    def _build_discriminator(self):
        pass

    def _build_loss(self):
        pass

    def train(self):
        df = DataFeed()
        
def main():
    # gan = VanillaGAN()
    # gan.train()

    df = DataFeed()

if __name__ == '__main__':
    main()