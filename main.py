import time
import utils
import tensorflow as tf
from capsNet import CapsNet
from config import cfg


def _get_batches(x, y):
    """
        Split features and labels into batches
        """
    for start in range(0, len(x), cfg.BATCH_SIZE):
        end = min(start + cfg.BATCH_SIZE, len(x))
        yield x[start:end], y[start:end]


def _print_status(sess, x_train, y_train, x_valid, y_valid,
                  merged, cost, accuracy, inputs, labels,
                  start_time, epoch_i, batch_counter,
                  train_writer, valid_writer):

    cost_train_all = []
    cost_valid_all = []
    acc_train_all = []
    acc_valid_all = []

    for train_batch_x, train_batch_y in _get_batches(x_train, y_train):
        summary_train_i, cost_train_i, acc_train_i = \
            sess.run([merged, cost, accuracy],
                     feed_dict={inputs: train_batch_x, labels: train_batch_y})
        train_writer.add_summary(summary_train_i, batch_counter)
        cost_train_all.append(cost_train_i)
        acc_train_all.append(acc_train_i)

    for valid_batch_x, valid_batch_y in _get_batches(x_valid, y_valid):
        summary_valid_i, cost_valid_i, acc_valid_i = \
            sess.run([merged, cost, accuracy],
                     feed_dict={inputs: valid_batch_x, labels: valid_batch_y})
        valid_writer.add_summary(summary_valid_i, batch_counter)
        cost_valid_all.append(cost_valid_i)
        acc_valid_all.append(acc_valid_i)

    cost_train = sum(cost_train_all) / len(cost_train_all)
    cost_valid = sum(cost_valid_all) / len(cost_valid_all)
    acc_train = sum(acc_train_all) / len(acc_train_all)
    acc_valid = sum(acc_valid_all) / len(acc_valid_all)

    total_time = time.time() - start_time

    print('Epoch: {}/{} |'.format(epoch_i + 1, cfg.EPOCHS),
          'Batch: {} |'.format(batch_counter),
          'Time: {:.2f}s |'.format(total_time),
          'Train_Loss: {:.4f} |'.format(cost_train),
          'Valid_Loss: {:.4f}'.format(cost_valid),
          'Train_Accuracy: {:.2f}% |'.format(acc_train*100),
          'Valid_Accuracy: {:.2f}%'.format(acc_valid*100))


def train(model):

    # Load data
    x_train = None
    y_train = None
    x_valid = None
    y_valid = None

    # Build graph
    train_graph, inputs, labels, cost, optimizer, accuracy = \
        model.build_graph(image_size=x_train.shape[1:])

    train_log_path = cfg.LOG_PATH + '/train'
    valid_log_path = cfg.LOG_PATH + '/valid'
    utils.check_dir([cfg.LOG_PATH, train_log_path, valid_log_path])

    with tf.Session(graph=train_graph) as sess:

        # Merge all the summaries and create writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(train_log_path, sess.graph)
        valid_writer = tf.summary.FileWriter(valid_log_path)

        sess.run(tf.global_variables_initializer())

        start_time = time.time()
        batch_counter = 0

        for epoch_i in range(cfg.EPOCHS):

            for x_batch, y_batch in _get_batches(x_train, y_train):

                batch_counter += 1

                # Training optimizer
                sess.run(optimizer, feed_dict={inputs: x_batch, labels: y_batch})

                if batch_counter % cfg.DISPLAY_STEP == 0:

                    _print_status(sess, x_train, y_train, x_valid, y_valid,
                                  merged, cost, accuracy, inputs, labels,
                                  start_time, epoch_i, batch_counter,
                                  train_writer, valid_writer)


if __name__ == '__main__':

    train(CapsNet)
