import torch
import torch.nn as nn
from transformers import AutoModel


class ClassificationHead(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout=0.3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.layers(x)


class ComplaintClassifier(nn.Module):
    def __init__(
        self,
        model_name,
        num_categories,
        classifier_hidden_size=512,
        dropout=0.3
    ):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(model_name)
        hidden_size = self.transformer.config.hidden_size

        self.classification_head = ClassificationHead(
            input_size=hidden_size,
            hidden_size=classifier_hidden_size,
            num_classes=num_categories,
            dropout=dropout
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classification_head(cls_embedding)

        return logits
