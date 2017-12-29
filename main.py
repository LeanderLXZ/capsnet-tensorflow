import time
import utils
import tensorflow as tf
import numpy as np
from capsNet import CapsNet
from config import cfg


def get_batches(x, y):
    """
        Split features and labels into batches
        """
    for start in range(0, len(x)-cfg.BATCH_SIZE, cfg.BATCH_SIZE):
        end = start + cfg.BATCH_SIZE
        yield x[start:end], y[start:end]


def display_status(sess, x_valid, y_valid, cost, accuracy, inputs, labels,
                   x_batch, y_batch, epoch_i, batch_counter, start_time):

    valid_batch_idx = np.random.choice(range(len(x_valid)), cfg.BATCH_SIZE).tolist()
    x_valid_batch = x_valid[valid_batch_idx]
    y_valid_batch = y_valid[valid_batch_idx]

    cost_train, acc_train = \
        sess.run([cost, accuracy], feed_dict={inputs: x_batch, labels: y_batch})
    cost_valid, acc_valid = \
        sess.run([cost, accuracy], feed_dict={inputs: x_valid_batch, labels: y_valid_batch})

    print('Epoch: {}/{} |'.format(epoch_i + 1, cfg.EPOCHS),
          'Batch: {} |'.format(batch_counter),
          'Time: {:.2f}s |'.format(time.time() - start_time),
          'Train_Loss: {:.4f} |'.format(cost_train),
          'Train_Accuracy: {:.2f}% |'.format(acc_train * 100),
          'Valid_Loss: {:.4f} |'.format(cost_valid),
          'Valid_Accuracy: {:.2f}% |'.format(acc_valid * 100))


def add_summaries(sess, x_valid, y_valid, train_writer, valid_writer,
                  merged, inputs, labels, x_batch, y_batch, batch_counter):

    valid_batch_idx = np.random.choice(range(len(x_valid)), cfg.BATCH_SIZE).tolist()
    x_valid_batch = x_valid[valid_batch_idx]
    y_valid_batch = y_valid[valid_batch_idx]

    summary_train = sess.run(merged, feed_dict={inputs: x_batch, labels: y_batch})
    train_writer.add_summary(summary_train, batch_counter)
    summary_valid = sess.run(merged, feed_dict={inputs: x_valid_batch, labels: y_valid_batch})
    valid_writer.add_summary(summary_valid, batch_counter)

    print('[Summaries Added!]')


def print_full_set_eval(sess, x_train, y_train, x_valid, y_valid, cost, accuracy,
                        inputs, labels, start_time, epoch_i, batch_counter):

    utils.thick_line()
    print('Calculating losses using full data set...')
    utils.thick_line()
    cost_train_all = []
    cost_valid_all = []
    acc_train_all = []
    acc_valid_all = []

    if cfg.EVAL_WITH_FULL_TRAIN_SET:
        for train_batch_x, train_batch_y in get_batches(x_train, y_train):
            cost_train_i, acc_train_i = \
                sess.run([cost, accuracy], feed_dict={inputs: train_batch_x, labels: train_batch_y})
            cost_train_all.append(cost_train_i)
            acc_train_all.append(acc_train_i)

    for valid_batch_x, valid_batch_y in get_batches(x_valid, y_valid):
        cost_valid_i, acc_valid_i = \
            sess.run([cost, accuracy], feed_dict={inputs: valid_batch_x, labels: valid_batch_y})
        cost_valid_all.append(cost_valid_i)
        acc_valid_all.append(acc_valid_i)

    cost_train = sum(cost_train_all) / len(cost_train_all)
    cost_valid = sum(cost_valid_all) / len(cost_valid_all)
    acc_train = sum(acc_train_all) / len(acc_train_all)
    acc_valid = sum(acc_valid_all) / len(acc_valid_all)

    print('Epoch: {}/{} |'.format(epoch_i + 1, cfg.EPOCHS),
          'Batch: {} |'.format(batch_counter),
          'Time: {:.2f}s |'.format(time.time() - start_time))
    utils.thin_line()
    if cfg.EVAL_WITH_FULL_TRAIN_SET:
        print('Full_Set_Train_Loss: {:.4f}\n'.format(cost_train),
              'Full_Set_Train_Accuracy: {:.2f}%'.format(acc_train * 100))
    print('Full_Set_Valid_Loss: {:.4f}\n'.format(cost_valid),
          'Full_Set_Valid_Accuracy: {:.2f}%'.format(acc_valid*100))
    utils.thick_line()


def train(model):

    start_time = time.time()
    utils.thick_line()
    print('Loading data...')
    utils.thick_line()

    # Load data
    # import numpy as np
    # x_train = np.random.normal(0.5, 0.5, [80, 28, 28, 1])
    # y_train = np.ones([80, 10], dtype=np.int)
    # x_valid = np.random.normal(0.5, 0.5, [20, 28, 28, 1])
    # y_valid = np.ones([20, 10], dtype=np.int)

    x_train = utils.load_data_from_pickle('./data/source_data/mnist/train_image.p')
    x_train = x_train.reshape([-1, 28, 28, 1])
    x_valid = x_train[55000:60000]
    assert x_valid.shape == (5000, 28, 28, 1), x_valid.shape
    x_train = x_train[:55000]
    assert x_train.shape == (55000, 28, 28, 1), x_train.shape
    y_train = utils.load_data_from_pickle('./data/source_data/mnist/train_label.p')
    y_valid = y_train[55000:60000]
    assert y_valid.shape == (5000, 10), y_valid.shape
    y_train = y_train[:55000]
    assert y_train.shape == (55000, 10), y_train.shape

    # Build graph
    utils.thick_line()
    print('Building graph...')
    train_graph, inputs, labels, cost, optimizer, accuracy = \
        model.build_graph(image_size=x_train.shape[1:], num_class=y_train.shape[1])

    train_log_path = cfg.LOG_PATH + '/train'
    valid_log_path = cfg.LOG_PATH + '/valid'
    utils.check_dir([cfg.LOG_PATH, train_log_path, valid_log_path])

    with tf.Session(graph=train_graph) as sess:

        utils.thick_line()
        print('Training...')

        # Merge all the summaries and create writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(train_log_path, sess.graph)
        valid_writer = tf.summary.FileWriter(valid_log_path)

        sess.run(tf.global_variables_initializer())
        batch_counter = 0

        for epoch_i in range(cfg.EPOCHS):

            utils.thick_line()
            print('Start epoch: {}/{}'.format(epoch_i + 1, cfg.EPOCHS))
            utils.thin_line()

            for x_batch, y_batch in get_batches(x_train, y_train):

                batch_counter += 1

                # Training optimizer
                sess.run(optimizer, feed_dict={inputs: x_batch, labels: y_batch})

                if batch_counter % cfg.DISPLAY_STEP == 0:
                    display_status(sess, x_valid, y_valid, cost, accuracy, inputs, labels,
                                   x_batch, y_batch, epoch_i, batch_counter, start_time)

                if batch_counter % cfg.SUMMARY_STEP == 0:
                    add_summaries(sess, x_valid, y_valid, train_writer, valid_writer,
                                  merged, inputs, labels, x_batch, y_batch, batch_counter)

                if batch_counter % cfg.FULL_SET_EVAL_STEP == 0:
                    print_full_set_eval(sess, x_train, y_train, x_valid, y_valid, cost, accuracy,
                                        inputs, labels, start_time, epoch_i, batch_counter)

            utils.thin_line()
            print('Epoch done! Using time: {.2f}'.format(time.time() - start_time))

    utils.thick_line()
    print('Done! Total time: {.2f}'.format(time.time() - start_time))
    utils.thick_line()


if __name__ == '__main__':

    CapsNet_ = CapsNet()
    train(CapsNet_)
