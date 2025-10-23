docker build -t sherloc .
docker run -it --rm --gpus all --net host --shm-size=1g -v  $(pwd)/../..:/code -v /data3:/data3 sherloc
