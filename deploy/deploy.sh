#!/bin/bash
# Deploy F1 Race Predictor to AWS ECS Fargate
# Prerequisites: AWS CLI configured, Docker installed
#
# Usage: ./deploy/deploy.sh <aws-account-id> <region>
# Example: ./deploy/deploy.sh 123456789012 us-west-2

set -e

ACCOUNT_ID=${1:?"Usage: $0 <aws-account-id> <region>"}
REGION=${2:-"us-west-2"}
APP_NAME="f1-race-predictor"
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${APP_NAME}"

echo "=== Step 1: Create ECR repository ==="
aws ecr create-repository \
    --repository-name ${APP_NAME} \
    --region ${REGION} 2>/dev/null || echo "ECR repo already exists"

echo "=== Step 2: Build Docker image ==="
docker build -t ${APP_NAME} .

echo "=== Step 3: Push to ECR ==="
aws ecr get-login-password --region ${REGION} | \
    docker login --username AWS --password-stdin ${ECR_REPO}
docker tag ${APP_NAME}:latest ${ECR_REPO}:latest
docker push ${ECR_REPO}:latest

echo "=== Step 4: Deploy CloudFormation stack ==="
aws cloudformation deploy \
    --template-file deploy/cloudformation.yaml \
    --stack-name ${APP_NAME} \
    --parameter-overrides \
        ImageUri=${ECR_REPO}:latest \
    --capabilities CAPABILITY_IAM \
    --region ${REGION}

echo "=== Done ==="
ALB_URL=$(aws cloudformation describe-stacks \
    --stack-name ${APP_NAME} \
    --region ${REGION} \
    --query 'Stacks[0].Outputs[?OutputKey==`AppUrl`].OutputValue' \
    --output text)
echo "App URL: ${ALB_URL}"
