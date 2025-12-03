#!/bin/bash
IMAGE_NAME="speedjobs-model"
VERSION="1.0.0"

CPU_PLATFORM=amd64

echo "🔨 Docker 이미지 빌드 중..."
echo "이미지: ${IMAGE_NAME}:${VERSION}"
echo "플랫폼: linux/${CPU_PLATFORM}"
echo ""

# Docker 이미지 빌드
docker build \
  --tag ${IMAGE_NAME}:${VERSION} \
  --file Dockerfile \
  --platform linux/${CPU_PLATFORM} \
  ${IS_CACHE} .

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 빌드 완료: ${IMAGE_NAME}:${VERSION}"
    echo ""
    echo "빌드된 이미지 확인:"
    docker images | grep ${IMAGE_NAME}
else
    echo ""
    echo "❌ 빌드 실패"
    exit 1
fi