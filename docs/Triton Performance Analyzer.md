# Triton Performance Analyzer

- https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/perf_analyzer.html
- https://github.com/triton-inference-server/server/blob/main/docs/user_guide/perf_analyzer.md


sudo docker run -it --gpus=1 --net=host nvcr.io/nvidia/tritonserver:22.12-py3-sdk

Inside triton client docker container:

perf_analyzer -m yolox_x32 --concurrency-range 2:2 --collect-metrics --verbose-csv -f perf32.csv






sudo docker run -it --gpus=0 --net=host nvcr.io/nvidia/tritonserver:22.12-py3-sdk

perf_analyzer -m yolox_x16 --concurrency-range 2:2 --collect-metrics --verbose-csv -f perf16.csv

----

## yolox_x32 on g4dn.xlarge

```
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Latency limit: 0 msec
  Concurrency limit: 2 concurrent requests
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 2
  Client: 
    Request count: 313
    Throughput: 17.3865 infer/sec
    Avg latency: 114692 usec (standard deviation 7452 usec)
    p50 latency: 114086 usec
    p90 latency: 116097 usec
    p95 latency: 116518 usec
    p99 latency: 117596 usec
    Avg HTTP time: 114683 usec (send/recv 4632 usec + response wait 110051 usec)
  Server: 
    Inference count: 313
    Execution count: 313
    Successful request count: 313
    Avg request latency: 106073 usec (overhead 44 usec + queue 32 usec + compute input 47733 usec + compute infer 57356 usec + compute output 907 usec)

  Server Prometheus Metrics: 
    Avg GPU Utilization:
      GPU-689d67d8-e3dc-813d-7f1b-1efb383758fd : 88.8889%
    Avg GPU Power Usage:
      GPU-689d67d8-e3dc-813d-7f1b-1efb383758fd : 65.5048 watts
    Max GPU Memory Usage:
      GPU-689d67d8-e3dc-813d-7f1b-1efb383758fd : 1887436800 bytes
    Total GPU Memory:
      GPU-689d67d8-e3dc-813d-7f1b-1efb383758fd : 16106127360 bytes
Inferences/Second vs. Client Average Batch Latency
Concurrency: 2, throughput: 17.3865 infer/sec, latency 114692 usec
```

## yolox_x16 on g4dn.xlarge

perf_analyzer -m yolox_x16 --concurrency-range 2:2 --collect-metrics --verbose-csv -f perf16.csv
*** Measurement Settings ***
 Batch size: 1
 Service Kind: Triton
 Using "time_windows" mode for stabilization
 Measurement window: 5000 msec
 Latency limit: 0 msec
 Concurrency limit: 2 concurrent requests
 Using synchronous calls for inference
 Stabilizing using average latency
Request concurrency: 2
 Client:
  Request count: 1169
  Throughput: 64.9294 infer/sec
  Avg latency: 30761 usec (standard deviation 2098 usec)
  p50 latency: 30615 usec
  p90 latency: 31113 usec
  p95 latency: 31352 usec
  p99 latency: 32117 usec
  Avg HTTP time: 30753 usec (send/recv 4492 usec + response wait 26261 usec)
 Server:
  Inference count: 1170
  Execution count: 1170
  Successful request count: 1170
  Avg request latency: 22335 usec (overhead 35 usec + queue 28 usec + compute input 6010 usec + compute infer 15334 usec + compute output 926 usec)
 Server Prometheus Metrics:
  Avg GPU Utilization:
   GPU-689d67d8-e3dc-813d-7f1b-1efb383758fd : 89.4737%
  Avg GPU Power Usage:
   GPU-689d67d8-e3dc-813d-7f1b-1efb383758fd : 65.5147 watts
  Max GPU Memory Usage:
   GPU-689d67d8-e3dc-813d-7f1b-1efb383758fd : 1002438656 bytes
  Total GPU Memory:
   GPU-689d67d8-e3dc-813d-7f1b-1efb383758fd : 16106127360 bytes
Inferences/Second vs. Client Average Batch Latency
Concurrency: 2, throughput: 64.9294 infer/sec, latency 30761 usec

## yolox_x32 on g5g.xlarge

perf_analyzer -m yolox_x32 --concurrency-range 2:2 --collect-metrics --verbose-csv -f perf32.csv
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Latency limit: 0 msec
  Concurrency limit: 2 concurrent requests
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 2
  Client: 
    Request count: 316
    Throughput: 17.5533 infer/sec
    Avg latency: 114175 usec (standard deviation 1150 usec)
    p50 latency: 114087 usec
    p90 latency: 115749 usec
    p95 latency: 116202 usec
    p99 latency: 117055 usec
    Avg HTTP time: 114167 usec (send/recv 3608 usec + response wait 110559 usec)
  Server: 
    Inference count: 316
    Execution count: 316
    Successful request count: 316
    Avg request latency: 107595 usec (overhead 31 usec + queue 29 usec + compute input 49851 usec + compute infer 57053 usec + compute output 630 usec)

  Server Prometheus Metrics: 
    Avg GPU Utilization:
      GPU-eaa31c7a-8c73-c0c8-a331-9e8776e6b357 : 100%
    Avg GPU Power Usage:
      GPU-eaa31c7a-8c73-c0c8-a331-9e8776e6b357 : 70.1517 watts
    Max GPU Memory Usage:
      GPU-eaa31c7a-8c73-c0c8-a331-9e8776e6b357 : 1935671296 bytes
    Total GPU Memory:
      GPU-eaa31c7a-8c73-c0c8-a331-9e8776e6b357 : 16106127360 bytes
Inferences/Second vs. Client Average Batch Latency
Concurrency: 2, throughput: 17.5533 infer/sec, latency 114175 usec

## yolox_x16 on g5g.xlarge

perf_analyzer -m yolox_x16 --concurrency-range 2:2 --collect-metrics --verbose-csv -f perf16.csv
*** Measurement Settings ***
  Batch size: 1
  Service Kind: Triton
  Using "time_windows" mode for stabilization
  Measurement window: 5000 msec
  Latency limit: 0 msec
  Concurrency limit: 2 concurrent requests
  Using synchronous calls for inference
  Stabilizing using average latency

Request concurrency: 2
  Client: 
    Request count: 657
    Throughput: 36.4917 infer/sec
    Avg latency: 54721 usec (standard deviation 4201 usec)
    p50 latency: 54446 usec
    p90 latency: 55264 usec
    p95 latency: 55655 usec
    p99 latency: 57088 usec
    Avg HTTP time: 54713 usec (send/recv 3609 usec + response wait 51104 usec)
  Server: 
    Inference count: 657
    Execution count: 657
    Successful request count: 657
    Avg request latency: 48200 usec (overhead 31 usec + queue 32 usec + compute input 20183 usec + compute infer 27329 usec + compute output 624 usec)

  Server Prometheus Metrics: 
    Avg GPU Utilization:
      GPU-eaa31c7a-8c73-c0c8-a331-9e8776e6b357 : 94.7368%
    Avg GPU Power Usage:
      GPU-eaa31c7a-8c73-c0c8-a331-9e8776e6b357 : 68.6006 watts
    Max GPU Memory Usage:
      GPU-eaa31c7a-8c73-c0c8-a331-9e8776e6b357 : 1107296256 bytes
    Total GPU Memory:
      GPU-eaa31c7a-8c73-c0c8-a331-9e8776e6b357 : 16106127360 bytes
Inferences/Second vs. Client Average Batch Latency
Concurrency: 2, throughput: 36.4917 infer/sec, latency 54721 usec




----

g5g.xlarge	$0.126 per Hour
g4dn.xlarge	$0.1578 per Hour


g5g.xlarge	$0.42
g4ad.xlarge	$0.37853
g4dn.xlarge	$0.526

https://aws.amazon.com/ec2/pricing/on-demand/

g5g.xlarge	 $0.42	    4	8 GiB	EBS Only	    Up to 10 Gigabit
g5g.2xlarge	 $0.556	    8	16 GiB	EBS Only	    Up to 10 Gigabit

g4ad.xlarge	 $0.37853	4	16 GiB	150 GB NVMe SSD	Up to 10 Gigabit
g4ad.2xlarge $0.54117	8	32 GiB	300 GB NVMe SSD	Up to 10 Gigabit

g4dn.xlarge	 $0.526	    4	16 GiB	125 GB NVMe SSD	Up to 25 Gigabit
g4dn.2xlarge $0.752	    8	32 GiB	225 GB NVMe SSD	Up to 25 Gigabit


-------------

Get model links from ultralytics/README.md.

```
```


### YOLO-v8-x 32F

  Server Prometheus Metrics: 
    Avg GPU Utilization:
      GPU-b171660f-ffab-27bf-aa6a-6c1d0eaa8f49 : 100%
    Avg GPU Power Usage:
      GPU-b171660f-ffab-27bf-aa6a-6c1d0eaa8f49 : 69.922 watts
    Max GPU Memory Usage:
      GPU-b171660f-ffab-27bf-aa6a-6c1d0eaa8f49 : 1595932672 bytes
    Total GPU Memory:
      GPU-b171660f-ffab-27bf-aa6a-6c1d0eaa8f49 : 16106127360 bytes
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 18.273 infer/sec, latency 54676 usec
Concurrency: 3, throughput: 20.2164 infer/sec, latency 147825 usec
Concurrency: 5, throughput: 19.9395 infer/sec, latency 249570 usec
Concurrency: 7, throughput: 19.7196 infer/sec, latency 353126 usec
Concurrency: 9, throughput: 19.4802 infer/sec, latency 459181 usec

-------------

### YOLO-v8-x 16F

  Server Prometheus Metrics: 
    Avg GPU Utilization:
      GPU-b171660f-ffab-27bf-aa6a-6c1d0eaa8f49 : 100%
    Avg GPU Power Usage:
      GPU-b171660f-ffab-27bf-aa6a-6c1d0eaa8f49 : 69.3742 watts
    Max GPU Memory Usage:
      GPU-b171660f-ffab-27bf-aa6a-6c1d0eaa8f49 : 960495616 bytes
    Total GPU Memory:
      GPU-b171660f-ffab-27bf-aa6a-6c1d0eaa8f49 : 16106127360 bytes
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 48.3113 infer/sec, latency 20684 usec
Concurrency: 3, throughput: 73.5299 infer/sec, latency 40774 usec
Concurrency: 5, throughput: 73.0415 infer/sec, latency 68368 usec
Concurrency: 7, throughput: 72.49 infer/sec, latency 96426 usec
Concurrency: 9, throughput: 71.9883 infer/sec, latency 124879 usec

-------------

### YOLO-v8-l 16F

  Server Prometheus Metrics: 
    Avg GPU Utilization:
      GPU-b171660f-ffab-27bf-aa6a-6c1d0eaa8f49 : 100%
    Avg GPU Power Usage:
      GPU-b171660f-ffab-27bf-aa6a-6c1d0eaa8f49 : 69.3005 watts
    Max GPU Memory Usage:
      GPU-b171660f-ffab-27bf-aa6a-6c1d0eaa8f49 : 899678208 bytes
    Total GPU Memory:
      GPU-b171660f-ffab-27bf-aa6a-6c1d0eaa8f49 : 16106127360 bytes
Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 58.1023 infer/sec, latency 17201 usec
Concurrency: 3, throughput: 110.192 infer/sec, latency 27203 usec
Concurrency: 5, throughput: 108.857 infer/sec, latency 45877 usec
Concurrency: 7, throughput: 107.537 infer/sec, latency 65027 usec
Concurrency: 9, throughput: 106.552 infer/sec, latency 84385 usec
