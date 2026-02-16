from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
# from predictor import ComplaintPredictor
from app.predictor import ComplaintPredictor


app = FastAPI(title="Civic Complaint Classifier API")

predictor = ComplaintPredictor(
    model_path="models/best_model_civic.pt",
    mappings_path="models/category_mappings.json"
)


class ComplaintRequest(BaseModel):
    complaint_text: str


@app.get("/")
def root():
    return {"message": "Civic AI API running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/categories")
def categories():
    return {"categories": predictor.get_categories()}


@app.post("/classify")
def classify(request: ComplaintRequest):

    if not request.complaint_text.strip():
        raise HTTPException(status_code=400, detail="Empty complaint")

    result = predictor.predict(request.complaint_text)

    return result
