from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    def __init__(self):
        self.model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def analyze_sentiment(self, text: str) -> Tuple[int, float]:
        """
        Analyze the sentiment of a given text.
        Returns a tuple of (sentiment_score, confidence)
        sentiment_score: 1-5 (1 being most negative, 5 being most positive)
        confidence: probability score for the prediction
        """
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                sentiment_score = torch.argmax(probabilities).item() + 1
                confidence = probabilities[0][sentiment_score - 1].item()

            return sentiment_score, confidence
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return 3, 0.0  # Return neutral sentiment with 0 confidence in case of error

    def analyze_comments(self, comments: List[str]) -> Dict[str, float]:
        """
        Analyze a list of comments and return sentiment statistics.
        Returns a dictionary with average sentiment score and percentage of positive/negative comments.
        """
        if not comments:
            return {
                "average_sentiment": 3.0,
                "positive_percentage": 0.0,
                "negative_percentage": 0.0
            }

        sentiments = []
        positive_count = 0
        negative_count = 0

        for comment in comments:
            sentiment_score, _ = self.analyze_sentiment(comment)
            sentiments.append(sentiment_score)
            if sentiment_score >= 4:
                positive_count += 1
            elif sentiment_score <= 2:
                negative_count += 1

        total_comments = len(comments)
        return {
            "average_sentiment": sum(sentiments) / total_comments,
            "positive_percentage": (positive_count / total_comments) * 100,
            "negative_percentage": (negative_count / total_comments) * 100
        }