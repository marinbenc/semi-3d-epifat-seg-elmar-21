from io import BytesIO

import scipy.misc
import tensorflow as tf
import PIL
from PIL import Image


class Logger(object):

    def __init__(self, log_dir):
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def image_summary(self, tag, image, step):
        s = BytesIO()
        Image.fromarray(image).save(s, format="png")

        # Create an Image object
        img_sum = tf.summary.image(
            encoded_image_string=s.getvalue(),
            height=image.shape[0],
            width=image.shape[1],
        )

        # Create and write Summary
        summary = tf.summary(value=[tf.summary.Value(tag=tag, image=img_sum)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def image_list_summary(self, tag, images, step):
        if len(images) == 0:
            return
        img_summaries = []
        for i, img in enumerate(images):
            s = BytesIO()
            Image.fromarray(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.summary.image(
                encoded_image_string=s.getvalue(),
                height=img.shape[0],
                width=img.shape[1],
            )

            # Create a Summary value
            img_summaries.append(
                tf.summary.value(tag="{}/{}".format(tag, i), image=img_sum)
            )

        summary = tf.summary(value=img_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()