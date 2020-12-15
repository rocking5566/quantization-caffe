DATA?="${HOME}/Data"
DOCKER_FILE=Dockerfile
DOCKER=nvidia-docker
DOCKER_REPOSITORY=toolchain/bmcompress
IMAGE_NAME=${DOCKER_REPOSITORY}:dev_gpu
REPO_ROOT?=$(shell git rev-parse --show-toplevel)

X11_DISPLAY=--env DISPLAY=$(DISPLAY) \
	--env="QT_X11_NO_MITSHM=1" \
	-v /tmp/.X11-unix:/tmp/.X11-unix:ro

MOUNT_DIR=-v $(REPO_ROOT):/workspace \
	-v $(DATA):/data \
	-v /home/rocking/work/cvitek/onnx2caffe:/onnx2caffe

MOUNT_WEBCAM=--device /dev/video0:/dev/video0

build:
	$(DOCKER) build -t $(IMAGE_NAME) -f $(DOCKER_FILE) .

bash:
	$(DOCKER) run -it -w /workspace --privileged --rm $(MOUNT_DIR) $(MOUNT_WEBCAM) --net=host $(IMAGE_NAME) bash

x11:
	xhost +
	$(DOCKER) run -it -w /workspace --privileged --rm $(MOUNT_DIR) $(X11_DISPLAY) $(MOUNT_WEBCAM) --net=host $(IMAGE_NAME) bash
	xhost -
