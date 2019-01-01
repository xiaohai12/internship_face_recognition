import tensorflow as tf
import numpy as np
import cv2
import time



def init(detModel, detGpuRate):
    PATH_TO_CKPT = detModel
    net = tf.Graph()
    with net.as_default():
        od_graph_def = tf.GraphDef()
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = detGpuRate
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            sess = tf.Session(graph=net, config=config)
    return sess, net


def sortedBboxesConfs(bboxes, dConfs):
    if len(bboxes) == 0:
        return bboxes, dConfs

    bd = zip(bboxes, dConfs)
    bd = sorted(bd, key=lambda x: x[0][0])
    [nb, nd] = zip(*bd)

    return nb, nd

def sortedRecogConfs(bboxes, dConfs,Face):
    if len(bboxes) == 0:
        return bboxes, dConfs

    bd = zip(bboxes, dConfs,Face)
    bd = sorted(bd, key=lambda x: x[0][0])
    [nb, nd,nf] = zip(*bd)

    return nb, nd,nf

def filterBboxesConfs(shape, imgsBboxes, imgsConfs, single=False, thresh=0.5):
    [w, h] = shape
    if single:
        bboxes, confs = [], []
        for y in range(len(imgsBboxes)):
            if imgsConfs[y] >= thresh:
                [x1, y1, x2, y2] = list(imgsBboxes[y])
                x1, y1, x2, y2 = int(w * x1), int(h * y1), int(w * x2), int(h * y2)
                bboxes.append([y1, x1, y2, x2])
                confs.append(imgsConfs[y])
        return sortedBboxesConfs(bboxes, confs)
    else:
        retImgsBboxes, retImgsConfs = [], []
        for x in range(len(imgsBboxes)):
            bboxes, confs = [], []
            for y in range(len(imgsBboxes[x])):
                if imgsConfs[x][y] >= thresh:
                    [x1, y1, x2, y2] = list(imgsBboxes[x][y])
                    x1, y1, x2, y2 = int(w * x1), int(h * y1), int(w * x2), int(h * y2)
                    bboxes.append([y1, x1, y2, x2])
                    confs.append(imgsConfs[x][y])
            bboxes, confs = sortedBboxesConfs(bboxes, confs)
            retImgsBboxes.append(bboxes)
            retImgsConfs.append(confs)
        return retImgsBboxes, retImgsConfs

def filterRecogConfs(shape, imgsBboxes, imgsConfs,imageFace,single=False, thresh=0.5):
    [w, h] = shape
    if single:
        bboxes, confs,face = [], [],[]
        for y in range(len(imgsBboxes)):
            if imgsConfs[y] >= thresh:
                [x1, y1, x2, y2] = list(imgsBboxes[y])
                x1, y1, x2, y2 = int(w * x1), int(h * y1), int(w * x2), int(h * y2)
                bboxes.append([y1, x1, y2, x2])
                confs.append(imgsConfs[y])
                face.append(imageFace[y])
        if confs==[]:
            y=0
            [x1, y1, x2, y2] = list(imgsBboxes[y])
            x1, y1, x2, y2 = int(w * x1), int(h * y1), int(w * x2), int(h * y2)
            bboxes.append([y1, x1, y2, x2])
            confs.append(imgsConfs[y])
            face.append(imageFace[y])
        return sortedRecogConfs(bboxes, confs,face)
    else:
        retImgsBboxes, retImgsConfs = [], []
        for x in range(len(imgsBboxes)):
            bboxes, confs = [], []
            for y in range(len(imgsBboxes[x])):
                if imgsConfs[x][y] >= thresh:
                    [x1, y1, x2, y2] = list(imgsBboxes[x][y])
                    x1, y1, x2, y2 = int(w * x1), int(h * y1), int(w * x2), int(h * y2)
                    bboxes.append([y1, x1, y2, x2])
                    confs.append(imgsConfs[x][y])
            bboxes, confs = sortedBboxesConfs(bboxes, confs)
            retImgsBboxes.append(bboxes)
            retImgsConfs.append(confs)
        return retImgsBboxes, retImgsConfs

def detect(im, sess, net):  # the default nms value is 0.3
    image_np = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = net.get_tensor_by_name('image_tensor:0')
    bboxes = net.get_tensor_by_name('detection_boxes:0')

    dConfs = net.get_tensor_by_name('detection_scores:0')
    classes = net.get_tensor_by_name('detection_classes:0')
    num_detections = net.get_tensor_by_name('num_detections:0')
    recog_classes = net.get_tensor_by_name('recognition_classes:0')
    recog_Confs = net.get_tensor_by_name('recognition_scores:0')
    recog_boxes = net.get_tensor_by_name('recognition_boxes:0')
    (bboxes, dConfs, classes,recog_boxes,recog_Confs,recog_classes,num_detections) = \
        sess.run([bboxes, dConfs, classes,recog_boxes,recog_Confs,recog_classes,num_detections], feed_dict={image_tensor: image_np_expanded})


    w, h, _ = im.shape

    #Bboxes, confs = filterBboxesConfs([w, h], bboxes[0], dConfs[0], True)
    Bboxes, recog_Confs,recog_classes = filterRecogConfs([w, h], recog_boxes[0], recog_Confs[0],recog_classes[0],True)
    Bboxes = [[int(1 * b) for b in boxes] for boxes in Bboxes]



    # return Bboxes, dConfs,recog_classes,recog_Confs
    return Bboxes, recog_Confs,recog_classes

def detectBatch(ims, sess, net):
    for i in range(0, len(ims)):
        if i == 0:
            ims_np = cv2.cvtColor(ims[0], cv2.COLOR_BGR2RGB)
            ims_np_expanded = np.expand_dims(ims_np, axis=0)
        else:
            convert = cv2.cvtColor(ims[i], cv2.COLOR_BGR2RGB)
            convert = np.expand_dims(convert, axis=0)
            ims_np_expanded = np.concatenate((ims_np_expanded, convert), axis=0)
    image_tensor = net.get_tensor_by_name('image_tensor:0')
    imgsBboxes = net.get_tensor_by_name('detection_boxes:0')
    imgsConfs = net.get_tensor_by_name('detection_scores:0')
    classes = net.get_tensor_by_name('detection_classes:0')

    num_detections = net.get_tensor_by_name('num_detections:0')


    (imgsBboxes, imgsConfs, classes, num_detections) = sess.run([imgsBboxes, imgsConfs, classes, num_detections],
                                                                feed_dict={image_tensor: ims_np_expanded})
    w, h, _ = ims[0].shape

    imgsBboxes, imgsConfs = filterBboxesConfs([w, h], imgsBboxes, imgsConfs)
    imgsBboxes = [[[int(1 * b) for b in bbox] for bbox in bboxes] for bboxes in imgsBboxes]

    return imgsBboxes, imgsConfs

if __name__ == '__main__':

    gpu_id = 2
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
	
    # detModel= 'DetectionModels/saved_model/saved_model.pb'
    detModel= 'output_model/frozen_inference_graph.pb'
    detGpuRate = 0.1
    sess, net = init(detModel, detGpuRate)

    root_dir = 'images'
    file_paths = os.listdir(root_dir)
    output_path = 'model_output'
    font = cv2.FONT_HERSHEY_COMPLEX_SMALL


    f = open('person_label.txt', 'r')
    result = {}
    for line in f.readlines():
        line = line.strip()
        if not len(line):
            continue
        result[line.split(':')[1]] = line.split(':')[0]
    f.close()

    record = []
    for x in range(len(file_paths)):
        path = root_dir + '/'+file_paths[x]
        img_paths = os.listdir(path)
        label = int(result[file_paths[x]])
        counts = 0
        folder_path = os.path.join(output_path,file_paths[x])
        folder = os.path.exists(folder_path)
        predict ={}
        predict[label]= []
        if not folder:
            os.makedirs(folder_path)
        for y in range(len(img_paths)):
            print img_paths[y]
            start = time.time()
            img_path = os.path.join(path, img_paths[y])
            im = cv2.imread(img_path)
            bboxes,confs,face = detect(im,sess,net)
            end = time.time()
            for i in range(len(bboxes)):
                [x1, y1, x2, y2] = bboxes[i]
                cv2.rectangle(im, (x1, y1), (x2, y2), (231, 255, 0), 2)
                print face[i]
                predict[label].append(face[i])
                if face[i]==label:
                    counts +=1
                    break
                #cv2.putText(im, "object:"+str(confs[i]), (x1, y1), font, 0.8, (0, 0, 255))
                # cv2.putText(im, str(int(recog_classes[i])), (x1, y1), font, 1, (0, 0, 255))
                # cv2.putText(im, str(round(recog_confs[i],2)), (x1+25, y1), font, 1, (38,67,72))
            print str(end-start) +'   ' +  str(counts) +'/' + str(y)

            cv2.imwrite(os.path.join(folder_path,str(y) + '.jpg'), im)
        print counts/float(len(img_paths))
        print predict
        record.append(counts/float(len(img_paths)))
    print record
