import os
import glob
import numpy as np
import tensorflow as tf

# https://www.tensorflow.org/guide/datasets#applying_arbitrary_python_logic_with_tfpy_func
def load_numpy_arrays(array_path, label, frame_range):
    try:
        array = np.load(array_path.decode()).astype(np.float32)
    except ValueError:
        print(array_path)

    n_frames = np.random.randint(frame_range[0], frame_range[1]+1)

    if len(array) < n_frames:
        container = np.zeros((n_frames , 65), dtype=np.float32)
        container[0:len(array)] = array
    else:
        start_idx = np.random.randint(0, len(array)-n_frames+1)
        container = array[start_idx:start_idx+n_frames]

    container = np.expand_dims(container, 1)

    return container, label

def generate_voxc1_ds(voxc1_dir, frame_range=(200, 400), is_train=False, return_labels=False):
    """

    :param voxc1_dir: feature's root eg) 'voxceleb1/feats/fbank64'
    :param frame_range: number of frames eg) 300
    :return: tf.data.Dataset of voxceleb1
    """

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    # voxc1_dir should be feature's root
    feat_files = sorted(glob.glob(voxc1_dir + '/**/*.npy', recursive=True))
    all_labels = sorted(os.listdir(voxc1_dir))
    label2index = {label:i for i, label in enumerate(all_labels)}
    # n_labels = len(label2index)

    def parse_label(file_n):
        # for window compatibility
        file_n = file_n.replace("\\", '/')
        label = file_n.split("/")[-2]
        return label2index[label]

    label_list = list(map(parse_label, feat_files))

    path_ds = tf.data.Dataset.from_tensor_slices((feat_files, label_list))

    voxc1_ds = path_ds.map(
        lambda array_path, label: tuple(tf.numpy_function(load_numpy_arrays,
                                                   [array_path, label, frame_range],
                                                   [tf.float32, tf.int32])),
        num_parallel_calls=AUTOTUNE
    )

    if is_train:
        voxc1_ds = voxc1_ds.shuffle(buffer_size=len(feat_files))
        voxc1_ds = voxc1_ds.repeat()
        voxc1_ds = voxc1_ds.prefetch(buffer_size=AUTOTUNE)

    if return_labels:
        ret = (voxc1_ds, label_list)
    else:
        ret = voxc1_ds

    return ret


def voxc1_to_ds(voxc1_dir, batch_size, frame_range):
    voxc1_train_dir = os.path.join(voxc1_dir, "dev/train/")
    voxc1_val_dir = os.path.join(voxc1_dir, "dev/val/")
    voxc1_test_dir = os.path.join(voxc1_dir, "dev/test/")

    train_ds = generate_voxc1_ds(voxc1_train_dir, frame_range, is_train=True)
    train_ds = train_ds.padded_batch(batch_size, padded_shapes=([None, 1, 65], []))

    val_ds = generate_voxc1_ds(voxc1_val_dir, (300, 300))
    val_ds = val_ds.batch(batch_size)

    test_ds = generate_voxc1_ds(voxc1_test_dir, (300, 300))
    test_ds = test_ds.batch(batch_size)

    return train_ds, val_ds, test_ds

def voxc1_to_gends(train_x, train_y, val_x, val_y, batch_size):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def train_generator():
        for x, y in zip(train_x, train_y):
            yield x, y

    def val_generator():
        for x, y in zip(val_x, val_y):
            yield x, y

    train_ds = tf.data.Dataset.from_generator(train_generator,
                                              output_types=(tf.float32, tf.int32),
                                              output_shapes=((500, 1, 65), ()))
    train_ds = train_ds.shuffle(buffer_size=134000)
    train_ds = train_ds.repeat()
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    train_ds = train_ds.batch(batch_size)

    val_ds = tf.data.Dataset.from_generator(val_generator,
                                            output_types=(tf.float32, tf.int32),
                                            output_shapes=((500, 1, 65), ()))
    val_ds = val_ds.batch(batch_size)

    return train_ds, val_ds



def measure_ds_speed(ds, n_samples, batch_size):
    import time
    steps_per_epoch=tf.ceil(len(n_samples)/batch_size).numpy()

    def timeit(ds, batches=2*steps_per_epoch+1):
      overall_start = time.time()
      # Fetch a single batch to prime the pipeline (fill the shuffle buffer),
      # before starting the timer
      it = iter(ds.take(batches+1))
      next(it)

      start = time.time()
      for i, (images) in enumerate(it):
        if i%10 == 0:
          print('.',end='')
      print()
      end = time.time()

      duration = end-start
      print("{} batches: {} s".format(batches, duration))
      print("{:0.5f} Images/s".format(batch_size*batches/duration))
      print("Total time: {}s".format(end-overall_start))

    timeit(ds)

