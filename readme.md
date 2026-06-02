# LAMBDA_MAIN – AWS Lambda Microservices (CodeBuild Deployment)

This repository contains three independent AWS Lambda microservices.  
Each service is containerized using Docker, deployed by **AWS CodeBuild**, and pushed to **AWS ECR** automatically.  
Every Lambda has its own folder, dependencies, buildspec, and Dockerfile.

---

## 📁 Project Structure

```
LAMBDA_MAIN/
│
├── serverless.yml # Serverless config (root)
│
├── lambda_mcq/
│ ├── src/ # MCQ generation code
│ ├── handler.py # Lambda entrypoint
│ ├── Dockerfile # Container build
│ ├── buildspec.yml # CodeBuild build spec for this lambda
│ └── requirements.txt # Python deps
│
├── lambda_resumeparser/
│ ├── src/ # Resume parsing code (parser, utils, prompts)
│ ├── handler.py # Lambda entrypoint
│ ├── Dockerfile # Container build
│ ├── buildspec.yml # CodeBuild build spec for this lambda
│ └── requirements.txt # Python deps
│
└── lambda_roadmap/
├── src/ # Roadmap generator (LangChain + Pinecone)
├── handler.py # Lambda entrypoint
├── Dockerfile # Container build
├── buildspec.yml # CodeBuild build spec for this lambda
└── requirements.txt # Python deps

---

## 🧠 Overview of Each Lambda

### 1️⃣ **lambda_mcq**
- Generates 8–10 MCQs using LangChain + OpenAI  
- Returns clean JSON output  
- Designed to be integrated with Roadmap/Quiz systems  

### 2️⃣ **lambda_resumeparser**
- Parses PDF/DOCX resumes  
- Extracts:
  - Basic details
  - Education
  - Skills
  - Experience
  - Social Links  
- Stores embeddings in Pinecone  
- Supports both **S3 file input** and **Base64 uploads**  

### 3️⃣ **lambda_roadmap**
- Generates personalized AI learning roadmaps  
- Uses Pinecone for semantic context retrieval  
- Uses LangChain + OpenAI to produce structured JSON modules/topics  

---

## 🚀 Deployment — CodeBuild + Docker + ECR + Lambda

Each Lambda folder has its own **buildspec.yml** which:

✔ Builds the Docker image  
✔ Logs in to ECR  
✔ Pushes the image to ECR  
✔ Updates the Lambda function automatically  

### Example buildspec snippet included in each folder:
```yaml
version: 0.2
phases:
  pre_build:
    commands:
      - aws --version
      - $(aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com)
      - REPOSITORY_URI=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$IMAGE_REPO_NAME
  build:
    commands:
      - docker build -t $IMAGE_REPO_NAME .
      - docker tag $IMAGE_REPO_NAME:latest $REPOSITORY_URI:latest
  post_build:
    commands:
      - docker push $REPOSITORY_URI:latest
      - aws lambda update-function-code --function-name $LAMBDA_NAME --image-uri $REPOSITORY_URI:latest
artifacts:
  files: []
```

---

## 🔧 How To Deploy

### 1️⃣ Create an ECR repository per lambda
```
aws ecr create-repository --repository-name lambda_mcq
aws ecr create-repository --repository-name lambda_resumeparser
aws ecr create-repository --repository-name lambda_roadmap
```

### 2️⃣ Create Lambda functions (first time only)
```
aws lambda create-function \
  --function-name resume-parser-lambda \
  --package-type Image \
  --code ImageUri=<ECR_URI> \
  --role arn:aws:iam::<ACCOUNT_ID>:role/<ROLE>
```

### 3️⃣ Run CodeBuild (or CodePipeline)
- Automatically builds containers
- Pushes to ECR
- Updates Lambda image

No manual Docker commands needed — CodeBuild handles everything.

---

## 🔐 Environment Variables

Set in Lambda Console:

| Key | Purpose |
|-----|---------|
| `OPENAI_API_KEY` | OpenAI model access |
| `PINECONE_API_KEY` | Pinecone vector DB |
| `PINECONE_ENVIRONMENT` | Pinecone project region |
| `OUTPUT_BUCKET` | For saving parsed resume output (optional) |

---

## 🧪 Testing

### Resume Parser Input (Base64)
```json
{
  "filename": "resume.pdf",
  "content_base64": "<BASE64_DATA>"
}
```

### MCQ Generator Input
```json
{
  "module": "Neural Networks"
}
```

### Roadmap Generator Input
```json
{
  "query": "AI in Medical Imaging"
}
```

---

## ✔️ Summary

This repository provides:

- A **modular**, **scalable**, **production-ready** Lambda architecture  
- Fully automated deployment using **CodeBuild**  
- Clean separation of services:
  - MCQ Generation  
  - Resume Parsing  
  - Roadmap Generation  
- Uses Docker, OpenAI, LangChain, and Pinecone  

Each service is independently deployable, easy to maintain, and future-proof.
