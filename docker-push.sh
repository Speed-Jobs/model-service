#!/bin/bash

IMAGE_NAME="speedjobs-model"
VERSION="1.0.0"

DOCKER_REGISTRY="amdp-registry.skala-ai.com/skala25a"
DOCKER_REGISTRY_USER="robot\$skala25a"
DOCKER_REGISTRY_PASSWORD="1qB9cyusbNComZPHAdjNIFWinf52xaBJ"

echo "🔐 Docker 레지스트리 로그인 중..."
echo "레지스트리: ${DOCKER_REGISTRY}"
echo ""

# 1. Docker 레지스트리에 로그인
echo ${DOCKER_REGISTRY_PASSWORD} | docker login ${DOCKER_REGISTRY} \
	-u ${DOCKER_REGISTRY_USER}  --password-stdin \
   	|| { echo "❌ Docker 로그인 실패"; exit 1; }

echo "✅ 로그인 성공"
echo ""

# 2. harbor 로 push 하기 위해 tag 추가
echo "🏷️  태그 추가 중..."
docker tag ${IMAGE_NAME}:${VERSION} ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}

if [ $? -eq 0 ]; then
    echo "✅ 태그 추가 완료: ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"
else
    echo "❌ 태그 추가 실패"
    exit 1
fi
echo ""

# 3. Docker 이미지 푸시
echo "📤 이미지 푸시 중..."
docker push ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 푸시 완료: ${DOCKER_REGISTRY}/${IMAGE_NAME}:${VERSION}"
else
    echo ""
    echo "❌ 푸시 실패"
    exit 1
fi