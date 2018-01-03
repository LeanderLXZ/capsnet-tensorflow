import time
import utils
import os
import math
from PIL import Image
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from config import config


class Test(object):

    def __init__(self, cfg):

        # Config
        self.cfg = cfg

        # Get checkpoint path
        self.checkpoint_path = './checkpoints/{}/model.ckpt-{}'.format(self.cfg.TEST_VERSION, self.cfg.TEST_CKP_IDX)

        # Get log path, append information if the directory exist.
        self.test_log_path = self.cfg.TEST_LOG_PATH
        i_append_info = 0
        while os.path.isdir(self.test_log_path):
            i_append_info += 1
            self.test_log_path = self.cfg.TEST_LOG_PATH + '({})'.format(i_append_info)

        # Path for saving images
        self.test_image_path = os.path.join(self.test_log_path, 'images')

        # Check directory of paths
        utils.check_dir([self.test_log_path])
        if self.cfg.TEST_WITH_RECONSTRUCTION:
            if self.cfg.TEST_SAVE_IMAGE_STEP is not None:
                utils.check_dir([self.test_image_path])

        # Save config
        utils.save_config_log(self.test_log_path, self.cfg)

        # Load data
        utils.thick_line()
        print('Loading data...')
        utils.thin_line()
        x_test = utils.load_data_from_pickle('./data/source_data/mnist/test_image.p')
        y_test = utils.load_data_from_pickle('./data/source_data/mnist/test_label.p')

        x_test = np.divide(x_test, 255.)
        self.x_test = x_test.reshape([-1, 28, 28, 1])
        assert self.x_test.shape == (10000, 28, 28, 1), self.x_test.shape
        self.y_test = y_test
        assert self.y_test.shape == (10000, 10), self.y_test.shape

        # Calculate number of batches
        self.n_batch_test = len(self.y_test) // self.cfg.TEST_BATCH_SIZE

    def _get_tensors(self, loaded_graph):
        """
        Get inputs, labels, cost, and accuracy tensor from <loaded_graph>
        """
        with loaded_graph.as_default():

            utils.thin_line()
            print('Loading graph and tensors...')

            inputs_ = loaded_graph.get_tensor_by_name("inputs:0")
            labels_ = loaded_graph.get_tensor_by_name("labels:0")
            cost_ = loaded_graph.get_tensor_by_name("cost:0")
            accuracy_ = loaded_graph.get_tensor_by_name("accuracy:0")

            if self.cfg.TEST_WITH_RECONSTRUCTION:
                train_cost_ = loaded_graph.get_tensor_by_name("train_cost:0")
                rec_cost_ = loaded_graph.get_tensor_by_name("rec_images:0")
                rec_images_ = loaded_graph.get_tensor_by_name("rec_images:0")
                return inputs_, labels_, cost_, accuracy_, train_cost_, rec_cost_, rec_images_
            else:
                return inputs_, labels_, cost_, accuracy_

    def _save_images(self, sess, rec_images, inputs, labels,
                     x_batch, y_batch, batch_counter):
        """
        Save reconstruction images.
        """
        rec_images = sess.run(rec_images, feed_dict={inputs: x_batch, labels: y_batch})

        # Get maximum size for square grid of images
        save_col_size = math.floor(np.sqrt(rec_images.shape[0] * 2))
        save_row_size = save_col_size // 2

        # Scale to 0-255
        rec_images = np.divide(((rec_images - rec_images.min()) * 255),
                               (rec_images.max() - rec_images.min()))
        real_images = np.divide(((x_batch - x_batch.min()) * 255),
                                (x_batch.max() - x_batch.min()))

        # Put images in a square arrangement
        rec_images_in_square = np.reshape(rec_images[: save_row_size*save_col_size],
                                          (save_row_size, save_col_size, rec_images.shape[1],
                                           rec_images.shape[2], rec_images.shape[3])).astype(np.uint8)
        real_images_in_square = np.reshape(real_images[: save_row_size*save_col_size],
                                           (save_row_size, save_col_size, real_images.shape[1],
                                            real_images.shape[2], real_images.shape[3])).astype(np.uint8)

        if self.cfg.DATABASE_NAME == 'mnist':
            mode = 'L'
            rec_images_in_square = np.squeeze(rec_images_in_square, 4)
            real_images_in_square = np.squeeze(real_images_in_square, 4)
        else:
            mode = 'RGB'

        # Combine images to grid image
        gap = 2
        new_im = Image.new(mode, ((rec_images.shape[1]++gap) * save_row_size * 2 - gap,
                                  (rec_images.shape[2]+gap) * save_col_size - gap), 'white')
        for row_i in range(save_row_size*2):
            for col_i in range(save_col_size):
                if (row_i+1) % 2 == 0:
                    if mode == 'L':
                        image = rec_images_in_square[(row_i+1)//2-1, col_i, :, :]
                    else:
                        image = rec_images_in_square[(row_i+1)//2 - 1, col_i, :, :, :]
                else:
                    if mode == 'L':
                        image = real_images_in_square[int((row_i+1)//2), col_i, :, :]
                    else:
                        image = real_images_in_square[int((row_i+1)//2), col_i, :, :, :]
                im = Image.fromarray(image, mode)
                new_im.paste(im, (col_i*(rec_images.shape[2]+gap),
                                  row_i*(rec_images.shape[1]+gap)))

        save_image_path = os.path.join(self.test_image_path, 'batch_{}.jpg'.format(batch_counter))
        new_im.save(save_image_path)

    def _eval_on_batches(self, sess, inputs, labels, cost, accuracy,
                         train_cost, rec_cost, rec_images,  x, y, n_batch):
        """
        Calculate losses and accuracies of full train set.
        """
        cost_all = []
        acc_all = []
        train_cost_all = []
        rec_cost_all = []
        batch_counter = 0
        _batch_generator = utils.get_batches(x, y, self.cfg.TEST_BATCH_SIZE)

        if self.cfg.TEST_WITH_RECONSTRUCTION:
            for _ in tqdm(range(n_batch), total=n_batch, ncols=100, unit=' batches'):
                batch_counter += 1
                x_batch, y_batch = next(_batch_generator)
                cost_i, train_cost_i, rec_cost_i, acc_i = \
                    sess.run([cost, train_cost, rec_cost, accuracy],
                             feed_dict={inputs: x_batch, labels: y_batch})
                cost_all.append(cost_i)
                train_cost_all.append(train_cost_i)
                rec_cost_all.append(rec_cost_i)
                acc_all.append(acc_i)

                # Save reconstruct images
                if self.cfg.TEST_SAVE_IMAGE_STEP is not None:
                    if batch_counter % self.cfg.TEST_SAVE_IMAGE_STEP == 0:
                        self._save_images(sess, rec_images, inputs, labels,
                                          x_batch, y_batch, batch_counter)

            train_cost = sum(train_cost_all) / len(train_cost_all)
            rec_cost = sum(rec_cost_all) / len(rec_cost_all)

        else:
            for _ in tqdm(range(n_batch), total=n_batch, ncols=100, unit=' batches'):
                x_batch, y_batch = next(_batch_generator)
                cost_i, acc_i = \
                    sess.run([cost, accuracy],
                             feed_dict={inputs: x_batch, labels: y_batch})
                cost_all.append(cost_i)
                acc_all.append(acc_i)
            train_cost, rec_cost = None, None

        cost = sum(cost_all) / len(cost_all)
        accuracy = sum(acc_all) / len(acc_all)

        return cost, train_cost, rec_cost, accuracy

    def test(self):
        """
        Test model
        """
        start_time = time.time()
        loaded_graph = tf.Graph()

        with tf.Session(graph=loaded_graph) as sess:

            # Load saved model
            loader = tf.train.import_meta_graph(self.checkpoint_path + '.meta')
            loader.restore(sess, self.checkpoint_path)

            # Get Tensors from loaded model
            if self.cfg.TEST_WITH_RECONSTRUCTION:
                inputs, labels, cost, accuracy, \
                    train_cost, rec_cost, rec_images = self._get_tensors(loaded_graph)
            else:
                inputs, labels, cost, accuracy = self._get_tensors(loaded_graph)
                train_cost, rec_cost, rec_images = None, None, None

            utils.thick_line()
            print('Testing on test set...')

            utils.thin_line()
            print('Calculating loss and accuracy of test set...')

            cost_test, train_cost_test, rec_cost_test, acc_test = \
                self._eval_on_batches(sess, inputs, labels, cost, accuracy,
                                      train_cost, rec_cost, rec_images,
                                      self.x_test, self.y_test, self.n_batch_test)

            # Print losses and accuracy
            utils.thin_line()
            print('Test_Loss: {:.4f}\n'.format(cost_test),
                  'Test_Accuracy: {:.2f}%'.format(acc_test * 100))
            if self.cfg.TEST_WITH_RECONSTRUCTION:
                utils.thin_line()
                print('Test_Train_Loss: {:.4f}\n'.format(train_cost_test),
                      'Test_Reconstruction_Loss: {:.4f}'.format(rec_cost_test))

            # Save test log
            utils.save_test_log(self.test_log_path, cost_test, acc_test, train_cost_test,
                                rec_cost_test, self.cfg.TEST_WITH_RECONSTRUCTION)

            utils.thin_line()
            print('Testing finished! Using time: {:.2f}'.format(time.time() - start_time))
            utils.thick_line()


if __name__ == '__main__':

    Test_ = Test(config)
    Test_.test()