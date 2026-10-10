#!/usr/bin/env sh
set -eu

IP="${1:-}"
if [ -z "$IP" ]; then
  echo "用法：$0 <服务器局域网IP>"
  echo "示例：$0 192.168.1.23"
  exit 2
fi

mkdir -p nginx/certs
# 使用 Docker 生成证书，避免宿主机必须安装 openssl。
docker run --rm \
  -v "$(pwd)/nginx/certs:/certs" \
  -e CERT_IP="$IP" \
  alpine:3.20 sh -c '
    apk add --no-cache openssl >/dev/null &&
    openssl req -x509 -nodes -newkey rsa:2048 -days 825 \
      -keyout /certs/server.key -out /certs/server.crt \
      -subj "/CN=$CERT_IP" \
      -addext "subjectAltName=IP:$CERT_IP" &&
    chmod 600 /certs/server.key
  '

echo "证书已生成：nginx/certs/server.crt"
