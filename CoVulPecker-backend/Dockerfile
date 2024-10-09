FROM ubuntu:latest


RUN apt-get update && \
    apt-get -y upgrade


RUN apt-get install -y gcc openjdk-21-jdk wget python-is-python3 git curl gnupg bash zip unzip sudo graphviz nano python3-pip

RUN wget https://github.com/joernio/joern/releases/latest/download/joern-install.sh &&\
    chmod +x ./joern-install.sh && \
    printf 'Y\n/bin/joern\ny\n/usr/local/bin\n\n' | sudo ./joern-install.sh --interactive && \
    rm -rf joern-install.sh && \ 
    rm -rf joern-cli.zip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD [ "/bin/bash" ]