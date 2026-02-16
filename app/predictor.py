import torch
import yaml
import json
from transformers import AutoTokenizer
# from model import ComplaintClassifier
from app.model import ComplaintClassifier



class ComplaintPredictor:

    BASE_PRIORITY = {
        "Water Supply": 9,
        "Road Damage": 8,
        "Public Property Damage": 8,
        "Electricity Issue": 8,
        "Illegal Construction": 7,
        "Drainage Issue": 7,
        "Street Lights": 6,
        "Garbage Collection": 6,
        "Encroachment": 5,
        "Noise Pollution": 4,
        "Stray Animals": 4,
        "Tree Related": 3
    }
    def _keyword_override(self, text):
        text = text.lower()

        # Water Supply
        if any(word in text for word in [
            "paani nahi", "water not coming", "no water",
            "pipeline phat", "pipeline burst", "jal nahi"
        ]):
            return "Water Supply"

        # Road Damage
        if any(word in text for word in [
            "gadda", "khadda", "pothole", "road broken",
            "road crack", "bada hole"
        ]):
            return "Road Damage"

        # Electricity Issue
        if any(word in text for word in [
            "bijli nahi", "power cut", "light nahi",
            "electric pole", "sparking", "wire hanging"
        ]):
            return "Electricity Issue"

        return None


    def __init__(self, model_path, mappings_path, config_path="app/config.yaml"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Load category mappings
        with open(mappings_path, "r") as f:
            mappings = json.load(f)

        self.category_to_idx = mappings["category_to_idx"]
        self.idx_to_category = {int(k): v for k, v in mappings["idx_to_category"].items()}
        self.num_categories = len(self.category_to_idx)

        # Tokenizer (internet based)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model"]["name"])

        # Model
        self.model = ComplaintClassifier(
            model_name=self.config["model"]["name"],
            num_categories=self.num_categories,
            classifier_hidden_size=self.config["model"]["classifier_hidden_size"],
            dropout=self.config["model"]["hidden_dropout_prob"]
)


        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

    def _calculate_priority(self, category, confidence):

        base = self.BASE_PRIORITY.get(category, 5)

        boost = 0
        if confidence >= 0.85:
            boost = 2
        elif confidence >= 0.75:
            boost = 1

        final_priority = min(10, base + boost)
        return final_priority

    def predict(self, text):

        encoding = self.tokenizer(
            text,
            max_length=self.config["model"]["max_length"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1)
            confidence, pred = torch.max(probs, dim=-1)

        category = self.idx_to_category[pred.item()]
        confidence = confidence.item()
        override_category = self._keyword_override(text)
        if override_category:
            category = override_category


        priority = self._calculate_priority(category, confidence)

        return {
            "category": category,
            "confidence": confidence,
            "priority": priority
        }

    def get_categories(self):
        return list(self.category_to_idx.keys())
