PWD=$(cd -P -- "$(dirname -- "$0")" && pwd -P)
mkdir -p ${OUTPUT_DIR}/snapshot
#apt-get install protobuf-compiler
#protoc object_detection/protos/*.proto --python_out=.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

python object_detection/train.py --logtostderr --pipeline_config_path=ssh14_mobilenet_v1.config  --train_dir=${OUTPUT_DIR}/snapshot
#python object_detection/export_inference_graph.py --pipeline_config_path ssh14_mobilenet_v1.config --trained_checkpoint_prefix ${DATA_DIR}/TrainedModels/model.ckpt-9812 --output_directory ${OUTPUT_DIR}/snapshot
#python test_image.py --detModel_path TrainedModels/frozen_inference_graph.pb --input_dir images --output_directory ${OUTPUT_DIR}/snapshot
