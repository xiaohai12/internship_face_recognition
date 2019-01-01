import io
import os, cv2, time
import pandas as pd
from PIL import Image
import tensorflow as tf
from object_detection.utils import dataset_util
from collections import namedtuple


def csv_to_list(ori_csv_file, image_dir):
    count, im_list = 0, []
    with open(ori_csv_file) as f:
        lines = f.readlines()
        s_time = time.time()
        length = len(lines) - 1
        for line in lines[1:]:
            fpath, x, y, w, h= line.strip().split(',')
	    
            [x, y, w, h] = map(float, [x, y, w, h])
            xmin, ymin, xmax, ymax = map(str, [x, y, x + w, y + h])

            filename = fpath[1:-1] + '.jpg'
            person = filename.split('/')[0]

            im_path = os.path.join(image_dir, filename)

            height, width, _ = cv2.imread(im_path).shape

            value = (filename, str(width), str(height),
                     'face', xmin, ymin, xmax, ymax, person)
            im_list.append(value)
            count += 1
            print '%d/%d, time: %f' % (count, length, time.time() - s_time)
    return im_list


def class_text_to_int(row_label):
    if row_label == 'face':
        return 1
    else:
        None


def load_names(txt_names):
    vggface2names = []
    with open(txt_names) as f:
        for line in f.readlines():
            vggface2names.append(line.strip())
    return vggface2names


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, names):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    persons_text = []
    persons = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
        persons_text.append(row['person'].encode('utf8'))
        persons.append(names.index(row['person']) + 1)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/person/text': dataset_util.bytes_list_feature(persons_text),
        'image/object/person/label': dataset_util.int64_list_feature(persons),
    }))
    return tf_example


if __name__ == '__main__':

    # transform to csv file.
    root_path = '/home/leishengzhao_sx/internship/vggface2/'

    # train dataset
    ori_csv_file = os.path.join(root_path, 'bbox/loose_bb_train.csv')
    image_dir = os.path.join(root_path, 'images')
    im_list = csv_to_list(ori_csv_file, image_dir)

    # test dataset
    ori_csv_file = os.path.join(root_path, 'bbox/loose_bb_test.csv')
    image_dir = os.path.join(root_path, 'images')
    im_list += csv_to_list(ori_csv_file, image_dir)

    # save csv file
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax', 'person']
    df = pd.DataFrame(im_list, columns=column_name)
    csv_file = 'train_test.csv'
    df.to_csv(csv_file, index=None)

    # transfrom csv file to tensorflow record
    names = load_names('/home/leishengzhao_sx/internship/vggface2/vggface2names.txt')
    output_path = 'train_test.record'
    writer = tf.python_io.TFRecordWriter(output_path)
    examples = pd.read_csv(csv_file)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, image_dir, names)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
