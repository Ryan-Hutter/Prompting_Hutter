from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from transformers import pipeline
from itertools import groupby

# Initialize the FastAPI app
app = FastAPI()

# Initialize the NER pipeline
ner_pipeline = pipeline(
    "ner",
    model="dbmdz/bert-large-cased-finetuned-conll03-english",
    grouped_entities=True
)

# Initialize the QA pipeline
qa_pipeline = pipeline(
    "question-answering",
    model="deepset/roberta-base-squad2"
)

# Allowed domains for filtering
allowed_domains = [
    "clothing",
    "fashion",
    "shopping",
    "accessories",
    "sustainability",
    "shoes",
    "hats",
    "shirts",
    "dresses",
    "pants",
    "jeans",
    "skirts",
    "jackets",
    "coats",
    "t-shirts",
    "sweaters",
    "hoodies",
    "activewear",
    "formal wear",
    "casual wear",
    "sportswear",
    "outerwear",
    "swimwear",
    "underwear",
    "lingerie",
    "socks",
    "scarves",
    "gloves",
    "belts",
    "ties",
    "caps",
    "beanies",
    "boots",
    "sandals",
    "heels",
    "sneakers",
    "materials",
    "cotton",
    "polyester",
    "wool",
    "silk",
    "leather",
    "denim",
    "linen",
    "athleisure",
    "ethnic wear",
    "fashion trends",
    "custom clothing",
    "tailoring",
    "sustainable materials",
    "recycled clothing",
    "fashion brands",
    "streetwear"
]

# Pydantic models for structured response
class Entity(BaseModel):
    word: str
    entity_group: str
    score: float

class NERResponse(BaseModel):
    entities: List[Entity]

class QAResponse(BaseModel):
    question: str
    answer: str
    score: float

class CombinedRequest(BaseModel):
    text: str  # The input text prompt

class CombinedResponse(BaseModel):
    ner: NERResponse  # NER output
    qa: QAResponse  # QA output

# Function to check if the input text belongs to allowed domains
def is_text_in_allowed_domain(text: str, domains: List[str]) -> bool:
    for domain in domains:
        if domain in text.lower():
            return True
    return False

# Combined endpoint for NER and QA with domain filtering
@app.post("/process/", response_model=CombinedResponse)
async def process_request(request: CombinedRequest):
    """
    Process the input text for both NER and QA, returning both responses,
    only if the text matches the allowed domains.
    """
    input_text = request.text

    # Check if the input text belongs to the allowed domains
    if not is_text_in_allowed_domain(input_text, allowed_domains):
        raise HTTPException(
            status_code=400,
            detail=(
                "The input text does not match the allowed domains. "
                "Please provide a query related to clothing, fashion, or accessories."
            )
        )

    # Perform Named Entity Recognition (NER)
    ner_entities = ner_pipeline(input_text)

    # Process the NER entities into the required format
    formatted_entities = [
        {
            "word": entity["word"],
            "entity_group": entity["entity_group"],
            "score": float(entity["score"]),  # Convert numpy.float32 to Python float
        }
        for entity in ner_entities
    ]
    ner_response = {"entities": formatted_entities}

    # Perform Question Answering (QA)
    qa_result = qa_pipeline(question=input_text, context=input_text)
    qa_result["score"] = float(qa_result["score"])  # Convert numpy.float32 to Python float

    qa_response = {
        "question": input_text,
        "answer": qa_result["answer"],
        "score": qa_result["score"]
    }

    # Return both NER and QA responses
    return {"ner": ner_response, "qa": qa_response}


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint to confirm the server is running.
    """
    return {"message": "Welcome to the filtered NER and QA API!"}


# JSon response

# {
#   "entities": [
#     {
#       "word": "Nike",
#       "entity_group": "ORG",
#       "score": 0.995
#     },
#     {
#       "word": "running shoes",
#       "entity_group": "PRODUCT",
#       "score": 0.987
#     },
#     {
#       "word": "outdoor activities",
#       "entity_group": "ACTIVITY",
#       "score": 0.960
#     }
#   ]
# }
# {
#   "question": "Can you suggest comfortable Nike running shoes for outdoor activities?",
#   "answer": "Nike Air Zoom Pegasus or React Infinity Run are great options for outdoor running.",
#   "score": 0.978
# }
