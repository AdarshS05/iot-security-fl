FROM python:3.9-slim
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
	git \
	wget \
	make \
	gcc \
	pkg-config \
	clang \
	git \
	ca-certificates \
&& rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/radareorg/radare2 /tmp/radare2 && \
	/tmp/radare2/sys/install.sh && rm -rf /tmp/radare2
	
WORKDIR /app

RUN pip install --no-cache-dir r2pipe

RUN r2 -v

CMD ["bin/bash"]
