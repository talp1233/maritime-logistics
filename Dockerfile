FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate sample data and train ML models during build
RUN python -c "from src.data_collection.ground_truth import GroundTruthCollector; GroundTruthCollector().generate_sample_data(90)"
RUN python main.py --train-ml-gt

EXPOSE 8000

# Default: launch web dashboard
CMD ["python", "-m", "uvicorn", "src.web.app:app", "--host", "0.0.0.0", "--port", "8000"]
