FROM golang:1.15

RUN apt-get update
RUN apt-get -y install bzip2 git

RUN mkdir -p /yap/src \
    && cd /yap/src \
    && git clone --filter=tree:0 https://github.com/Yu-val-weiss/yap.git

WORKDIR /yap/src/yap

RUN bunzip2 data/*.bz2

ENV GOPATH=/yap

RUN go get . \
    && go build .

EXPOSE 8000

ENTRYPOINT ["./yap", "api"]



