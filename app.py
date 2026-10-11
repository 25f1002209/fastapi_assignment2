
from fastapi import FastAPI
from pydantic import BaseModel
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = SentimentIntensityAnalyzer()


class SentimentRequest(BaseModel):
    sentences: list[str]


def analyze_sentiment(sentence: str) -> str:
    scores = analyzer.polarity_scores(sentence)
    compound = scores["compound"]

    if compound >= 0.05:
        return "happy"
    elif compound <= -0.05:
        return "sad"
    else:
        return "neutral"


@app.post("/sentiment")
async def batch_sentiment(request: SentimentRequest):
    results = []

    for sentence in request.sentences:
        sentiment = analyze_sentiment(sentence)

        results.append({
            "sentence": sentence,
            "sentiment": sentiment
        })

    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
