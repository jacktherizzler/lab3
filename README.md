# Lab 3: Model Deployment for Inference Using FastAPI, Docker, and Docker Hub

This repository deploys the best-performing model from Lab 2 as a FastAPI inference API,
containerizes it with Docker, and prepares it for publishing to Docker Hub.

## Project Structure

```text
lab3/
├── app/
│   ├── __init__.py
│   └── main.py
├── artifacts/
│   ├── trained_model.pkl
│   └── results.json
├── .dockerignore
├── .gitignore
├── Dockerfile
├── deployment_record.md
├── requirements.txt
└── README.md
```

## Task 1: Select the Best Model from Lab 2

Review the Lab 2 GitHub Actions runs and choose the model with:

- lowest `MSE`
- highest `R2 Score`

Download these artifacts from that run into `artifacts/`:

- `trained_model.pkl`
- `results.json`

Record the details in `deployment_record.md`:

- commit hash
- workflow run ID
- MSE
- R2 Score

## Run the API Locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open:

- `http://127.0.0.1:8000/docs`

## Test the Prediction Endpoint

### POST request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4,
    "volatile_acidity": 0.7,
    "citric_acid": 0.0,
    "residual_sugar": 1.9,
    "chlorides": 0.076,
    "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

### GET request

```bash
curl "http://127.0.0.1:8000/predict?fixed_acidity=7.4&volatile_acidity=0.7&citric_acid=0.0&residual_sugar=1.9&chlorides=0.076&free_sulfur_dioxide=11.0&total_sulfur_dioxide=34.0&density=0.9978&pH=3.51&sulphates=0.56&alcohol=9.4"
```

Expected response format:

```json
{
  "name": "Your Name",
  "roll_no": "Your Roll Number",
  "wine_quality": 5
}
```

## Build and Run with Docker

```bash
docker build -t wine-quality-api:v1 .
docker run -p 8000:8000 wine-quality-api:v1
```

## Tag and Push to Docker Hub

Use the required Docker Hub naming format:

- username: `<roll_no>_<name>`
- repository: `wine_predict_<roll_no>`

Example:

```bash
docker tag wine-quality-api:v1 <dockerhub_username>/wine_predict_<roll_no>:v1
docker push <dockerhub_username>/wine_predict_<roll_no>:v1
```

## Pull and Run from Docker Hub

```bash
docker pull <dockerhub_username>/wine_predict_<roll_no>:v1
docker run -p 8000:8000 <dockerhub_username>/wine_predict_<roll_no>:v1
```

## Deliverables

Submit:

1. Docker Hub repository link
2. Screenshot of downloaded Lab 2 artifacts
3. Screenshot of Docker image pushed to Docker Hub
4. Screenshot of pulled image running successfully
5. Screenshot of inference API response

