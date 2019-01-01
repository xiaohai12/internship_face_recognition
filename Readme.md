# object_detection 各版本编译使用说明文档
## object_detectionv2:
- This is the version with ssh-mobilenetv1 network structure.
- anchor size:(0.05,0.1,0.2,0.4,0.8,0.95)
- modified files from the original version:
    - model_builder.py
    - ssh_mobilenet_v1_feature_extractor.py
    - feature_map_generators.py
    - ssh2_mobilenet_v1.config
    - multiple_grid_anchor_generator.py
- How to directly set it?
    - Download the directory object_detectionv2, and then rename it to object_detection, you should put this directory under tensorflow/models/research/, execute this command just at research/ directory: protoc object_detection/protos/*.proto --python_out=. , then this version of object_detection environment has been installed. 
- How to train it?
    - To train this version, you need the train and the validation data, and the pbtxt file for classes to detection. Here, they are face_label.pbtxt, train.record, val.record, also, you need the configuration file to set the training details, the file is named ssh2_mobilenet_v1.config. 
    You can run this command to directly train (before this, you need to change your own path):
    python /usr/lib/python2.7/site-packages/tensorflow/models/research/object_detection/train.py --logtostderr --pipeline_config_path=ssh2_mobilenet_v1.config  --train_dir=model_output(you can choose your own saved model path)
