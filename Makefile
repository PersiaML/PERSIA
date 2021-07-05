CUDIR := `pwd`
TARGET_DIR := `realpath build`
LOCAL_DIR := ../persia-system-python
REMOTE_USER_DIR := /home/aip-persia-team/wyl/

format:
	python3 -m black persia --config pyproject.toml

lint: 
	pytype persia --config setup.cfg

sync_repo: # sync to aiplatform-persia-dev machine
	rsync -r -v --exclude-from rsync_exclude.txt $(CUDIR) pdev2:$(REMOTE_USER_DIR)

build_dev_image:
	docker build -f docker/Dockerfile.dev.template docker -t persia-dev:master

build_dev_image_proxy:
	docker build -f docker/Dockerfile.dev.template docker \
	--build-arg http_proxy=$$http_proxy --build-arg https_proxy=$$https_proxy \
	--network=host -t persia-dev:master

build_pip:
	docker run -it --rm -v $(CUDIR):/workspace/ persia-dev:master bash -c "python3 setup.py bdist_wheel --dist-dir=docker/config/"

build_pip_proxy:
	docker run -it --rm \
	-e HTTP_PROXY=$$http_proxy -e HTTPS_PROXY=$$https_proxy \
	--network=host -v $(CUDIR):/workspace/ persia-dev:master \
	bash -c "python3 setup.py bdist_wheel --dist-dir=docker/config/"

build_runtime_image:
	docker build -f docker/Dockerfile.gpu.template docker -t persia-runtime:v1

build_runtime_image_proxy:
	docker build -f docker/Dockerfile.gpu.template docker \
	--build-arg http_proxy=$$http_proxy --build-arg https_proxy=$$https_proxy \
	--network=host -t persia-runtime:v1


build_image: build_dev_image build_pip build_runtime_image

build_persia_server:
	docker run -it --rm -v $(CUDIR):/workspace/ -v $(TARGET_DIR):/build/ persia-dev:master bash -c \
	"cd /workspace/persia-embedding-real && cargo build --release --package persia-embedding-sharded-server --target-dir /build && \
	mv /build/release/persia-embedding-sharded-middleware /build/ && \
	mv /build/release/persia-embedding-sharded-server /build/ && rm -rf /build/release"

build_persia_server_proxy:
	docker run -it --rm -v $(CUDIR):/workspace/ \
	-e HTTP_PROXY=$$http_proxy -e HTTPS_PROXY=$$https_proxy \
	--network=host -v $(TARGET_DIR):/build/ persia-dev:master bash -c \
	"cd /workspace/persia-embedding-real && cargo build --release --package persia-embedding-sharded-server --target-dir /build && \
	mv /build/release/persia-embedding-sharded-middleware /build/ && \
	mv /build/release/persia-embedding-sharded-server /build/ && rm -rf /build/release"


build_persia_cpu_client:
	docker run -it --rm -v $(CUDIR):/workspace/  -v $(TARGET_DIR):/build/ \
	persia-dev:master bash -c "cd /workspace/persia-embedding-real && \
	cargo build --release --package persia-embedding-py-client-sharded-server --target-dir /build && \
	cd /build/release && mv libpersia_embedding_py_client_sharded_server.so /build/persia_embedding_py_client_sharded_server.so && \
	rm -rf /build/release"

build_persia_client:
	docker run -it --rm -v $(CUDIR):/workspace/  -v $(TARGET_DIR):/build/ \
	persia-dev:master bash -c "cd /workspace/persia-embedding-real && \
	cargo build --release --package persia-embedding-py-client-sharded-server --features cuda --target-dir /build && \
	cd /build/release && mv libpersia_embedding_py_client_sharded_server.so /build/persia_embedding_py_client_sharded_server.so && \
	rm -rf /build/release"

build_persia_cpu_client_proxy:
	docker run -it --rm -v $(CUDIR):/workspace/ \
	-e HTTP_PROXY=$$http_proxy -e HTTPS_PROXY=$$https_proxy \
	--network=host -v $(TARGET_DIR):/build/ \
	persia-dev:master bash -c "cd /workspace/persia-embedding-real && \
	cargo build --release --package persia-embedding-py-client-sharded-server --target-dir /build && \
	cd /build/release && mv libpersia_embedding_py_client_sharded_server.so /build/persia_embedding_py_client_sharded_server.so && \
	rm -rf /build/release"

build_persia_client_proxy:
	docker run -it --rm -v $(CUDIR):/workspace/ \
	-e HTTP_PROXY=$$http_proxy -e HTTPS_PROXY=$$https_proxy \
	--network=host -v $(TARGET_DIR):/build/ \
	persia-dev:master bash -c "cd /workspace/persia-embedding-real && \
	cargo build --release --package persia-embedding-py-client-sharded-server --features cuda --target-dir /build && \
	cd /build/release && mv libpersia_embedding_py_client_sharded_server.so /build/persia_embedding_py_client_sharded_server.so && \
	rm -rf /build/release"


build_persia: build_persia_server build_persia_client 

build_persia_proxy: build_persia_server_proxy build_persia_client_proxy

build_all: build_image build_persia

run_dien_example:
	cd examples/DIEN && make run

run_dlrm_example:
	cd examples/DLRM && make run