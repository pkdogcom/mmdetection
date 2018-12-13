import argparse
import json
import os

import numpy as np
from PIL import Image


class MyEncoder(json.JSONEncoder):

    # to solve the problem that TypeError(repr(o) + " is not JSON serializable"
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert WIDER Face annotations to mmdetection format')
    parser.add_argument(
        '--rootdir', help="root directory for the dataset", required=True, type=str)
    parser.add_argument(
        '--outdir', help="output dir for json files", type=str)
    return parser.parse_args()


def parse_wider_gt(ann_file):
    """
    :param ann_file: the path of annotations file
    :return: a dict like [im-name] = [[x,y,w,h], ...]
    """
    wider_annot_dict = {}
    f = open(ann_file)
    while True:
        bboxes = []
        bboxes_ignore = []
        filename = f.readline().rstrip()
        if not filename:
            break
        face_num = int(f.readline().rstrip())
        for i in range(face_num):
            annot = f.readline().rstrip().split() # x, y, w, h, ..., other annotations, ...
            x, y, w, h = float(annot[0]), float(annot[1]), float(annot[2]), float(annot[3])
            if w >= 10 or h >= 10:  # Ignore super small image (<10 pixels) as common practice
                bboxes.append([x, y, x + w, y + h])
            else:
                bboxes_ignore.append([x, y, x+w, y + h])
        wider_annot_dict[filename] = (np.reshape(np.array(bboxes), [-1, 4]), np.reshape(np.array(bboxes_ignore), [-1, 4]))
    return wider_annot_dict


def convert_wider_annots(root_dir, out_dir):
    """Convert from WIDER FDDB-style format to COCO bounding box"""

    subsets = ['train', 'val']

    for subset in subsets:
        json_name = 'instances_WIDER_{}.json'.format(subset)

        print('Starting %s' % subset)
        ann_dict = {}
        images = []
        ann_file = os.path.join(root_dir, 'wider_face_split', 'wider_face_{}_bbx_gt.txt'.format(subset))

        wider_annot_dict = parse_wider_gt(ann_file)  # [im-file] = [[x,y,w,h], ...]

        for filename in wider_annot_dict.keys():
            if len(images) % 500 == 0:
                print("Processed %s images" % (len(images)))

            image = {}
            im = Image.open(os.path.join(root_dir, 'WIDER_{}'.format(subset), 'images', filename))
            image['width'] = im.width
            image['height'] = im.height
            image['filename'] = filename
            bboxes, bboxes_ignore = wider_annot_dict[filename]
            image['ann'] = {'bboxes': bboxes, 'labels': np.ones([len(bboxes)], dtype=np.int),
                            'bboxes_ignore': bboxes_ignore,
                            'labels_ignore': np.ones([len(bboxes_ignore)], dtype=np.int)}
            images.append(image)

        print("Num images: %s" % len(images))
        with open(os.path.join(out_dir, json_name), 'w', encoding='utf8') as outfile:
            outfile.write(json.dumps(images, cls=MyEncoder, indent=4))
            outfile.close()


if __name__ == '__main__':
    args = parse_args()
    if args.outdir is None:
        args.outdir = os.path.join(args.rootdir, 'annotations')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    convert_wider_annots(args.rootdir, args.outdir)
