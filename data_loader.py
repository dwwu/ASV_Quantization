import os
import glob
import numpy as np
import tensorflow as tf

# https://www.tensorflow.org/guide/datasets#applying_arbitrary_python_logic_with_tfpy_func
def load_numpy_arrays(array_path, label, n_frames):
    try:
        array = np.load(array_path.decode()).astype(np.float32)
    except ValueError:
        print(array_path)
    if len(array) < n_frames:
        container = np.zeros((n_frames , 65), dtype=np.float32)
        container[0:len(array)] = array
    else:
        container = array[:n_frames]

    container = np.expand_dims(container, 1)

    return container, label

def generate_voxc1_ds(voxc1_dir, n_frames, is_train=False, return_labels=False):
    """

    :param voxc1_dir: feature's root eg) 'voxceleb1/feats/fbank64'
    :param n_frames: number of frames eg) 300
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
                                                   [array_path, label, n_frames],
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

