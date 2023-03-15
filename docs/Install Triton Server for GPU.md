# Install Triton Server for GPU

References:

- [NVIDIA Driver Downloads](https://www.nvidia.com/Download/Find.aspx)
- https://devcloud.co.za/how-to-use-arm-based-gpu-ec2-instances-as-ecs-container-instances

## Install NVIDIA Drivers

### 1. Download the driver

For aarch64:

```
wget https://us.download.nvidia.com/tesla/525.60.13/NVIDIA-Linux-aarch64-525.60.13.run
```

(Debian package: /tesla/525.60.13/nvidia-driver-local-repo-ubuntu2204-525.60.13_1.0-1_arm64.deb)
   
                                   
For x86_64:

```
wget https://us.download.nvidia.com/tesla/525.60.13/NVIDIA-Linux-x86_64-525.60.13.run
```

### 2. Install gcc, make and headers

```
sudo apt update -y
sudo apt install gcc make linux-headers-$(uname -r) -y
```

### 3. Run the executable

```
chmod +x NVIDIA-Linux-aarch64-525.60.13.run
sudo sh ./NVIDIA-Linux-aarch64-525.60.13.run  --disable-nouveau --silent
```

For x86_64:

```
chmod +x NVIDIA-Linux-x86_64-525.60.13.run
sudo sh ./NVIDIA-Linux-x86_64-525.60.13.run  --disable-nouveau --silent
sudo apt install libvulkan1 -y
```

### 4. (Optional) Test if GPU is detected

```
nvidia-smi
```

## Install Docker and NVIDIA container runtime

### 1. Download and install Docker

```
curl https://get.docker.com | sh   && sudo systemctl --now enable docker
```

### 2. Setup NVIDIA repository information

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
         sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
         sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

### 3. Install NVIDIA container runtime

sudo apt update
sudo apt install -y nvidia-docker2 nvidia-container-runtime
sudo systemctl restart docker

### 4. Confirm if the NVIDIA runtime is used by Docker

sudo docker info --format '{{json .Runtimes.nvidia}}'

(Should get {"path":"nvidia-container-runtime"}.)

## Start Triton Server

https://github.com/triton-inference-server/server/blob/main/docs/getting_started/quickstart.md


git clone -b r22.12 https://github.com/triton-inference-server/server.git
cd server/docs/examples/
./fetch_models.sh

sudo docker run -it --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/home/ubuntu/server/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:22.12-py3 tritonserver --model-repository=/models --model-control-mode explicit --load-model yolox_x32

### Verify Triton Is Running Correctly

In another terminal window, run:

```
curl -v localhost:8000/v2/health/ready
...
< HTTP/1.1 200 OK
< Content-Length: 0
< Content-Type: text/plain
```

## Inference Request

```
sudo docker run -it --gpus=all --net=host nvcr.io/nvidia/tritonserver:22.12-py3-sdk
/workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg
```

## Install AWS CLI (optional)

```
sudo apt update
sudo apt install awscli
```



------------------------------------------------------------



sudo docker run -it --gpus=1 --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/home/ubuntu/ocr/models:/models nvcr.io/nvidia/tritonserver:22.12-py3 tritonserver --model-repository=/models


----

sudo docker run -it --gpus=all --rm -p8000:8000 -p8001:8001 -p8002:8002 -v/home/ubuntu/server/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:22.12-py3 tritonserver --model-repository=/models --model-control-mode explicit --load-model yolox_x32
