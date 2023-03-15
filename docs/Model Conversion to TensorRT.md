# Model Conversion to TensorRT (.plan)

Run from the model repository directory:

```
sudo docker run --rm --gpus=all -v${PWD}:/models nvcr.io/nvidia/tensorrt:22.12-py3 \
trtexec \
--workspace=10000 \
--onnx=/models/yolox_x.onnx \
--saveEngine=/models/triton/yolox_x32/1/model.plan
```

```
sudo docker run --rm -it --gpus=1 -v${PWD}:/models nvcr.io/nvidia/tensorrt:22.12-py3 \
trtexec \
--workspace=10000 \
--onnx=/models/yolox_x.onnx \
--fp16 \
--saveEngine=/models/yolox_x16/1/model.plan

trtexec \
--workspace=10000 \
--onnx=/models/yoloxv8l.onnx \
--fp16 \
--saveEngine=/models/yoloxv8l16/1/model.plan

```

```
perf_analyzer -m yolox_x16 --concurrency-range 2:2 --collect-metrics --verbose-csv -f perf16.csv
```


[TRT] CUDA lazy loading is not enabled. Enabling it can significantly reduce device memory usage. See `CUDA_MODULE_LOADING` in https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars





----

sudo docker run --rm --gpus=all -v${PWD}:/models nvcr.io/nvidia/tensorrt:22.12-py3 \
trtexec \
--workspace=10000 \
--onnx=/models/panet_ctw.onnx \
--saveEngine=/models/triton/panet_ctw/1/model.plan


trtexec \
--workspace=10000 \
--onnx=/models/abinet.onnx \
--saveEngine=/models/abinet.plan
