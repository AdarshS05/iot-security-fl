FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
	wget \
	make \
	gcc \
	clang \
	git \
&& rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/radareorg/radare2 /tmp/radare2 && \
	/tmp/radare2/sys/install.sh && rm -rf /tmp/radare2
	
WORKDIR /app

RUN pip install --no-cache-dir r2pipe

CMD ["bash"]
