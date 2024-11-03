gcloud builds submit --tag gcr.io/neural-network-app-440619/handwritten-digit-recognizer-app .
gcloud run deploy --image gcr.io/neural-network-app-440619/handwritten-digit-recognizer-app --platform managed --allow-unauthenticated
