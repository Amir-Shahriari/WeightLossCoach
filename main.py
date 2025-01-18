from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pydantic import BaseModel

# Load exercise data from the new dataset
current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "exercise_dataset.csv")

if not os.path.exists(dataset_path):
    raise FileNotFoundError("The file 'exercise_dataset.csv' does not exist.")

exercise_data = pd.read_csv(dataset_path)

# Ensure required columns exist
required_columns = {"Activity, Exercise or Sport (1 hour)", "130 lb", "155 lb", "180 lb", "205 lb"}
if not required_columns.issubset(exercise_data.columns):
    raise ValueError(f"The dataset must contain the following columns: {required_columns}")

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize sentence transformer for embedding and FAISS for similarity search
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
sample_embedding_dim = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(sample_embedding_dim)

# Preprocess and embed exercise data
def preprocess_and_embed_data(data):
    embeddings = []
    metadata = []
    for _, row in data.iterrows():
        text = row["Activity, Exercise or Sport (1 hour)"]
        embedding = embedding_model.encode(text)
        embeddings.append(embedding)
        metadata.append({
            "activity": text
        })
    embeddings = np.array(embeddings, dtype="float32")
    index.add(embeddings)
    return metadata

metadata = preprocess_and_embed_data(exercise_data)

# Pydantic model for user details
class UserDetails(BaseModel):
    weight: float
    height: float
    age: int
    gender: str
    activity_level: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Weight Loss Coach"}

@app.post("/calculate_and_recommend_weekly/")
def calculate_and_recommend_weekly(details: UserDetails):
    # Unpack details
    weight = details.weight
    height = details.height
    age = details.age
    gender = details.gender.lower()
    activity_level = details.activity_level.lower()

    try:
        # Calculate BMR
        if gender == "male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        elif gender == "female":
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        else:
            raise HTTPException(status_code=400, detail="Invalid gender. Use 'male' or 'female'.")

        # Adjust BMR with activity level
        activity_multipliers = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very active": 1.9,
        }
        if activity_level not in activity_multipliers:
            raise HTTPException(
                status_code=400,
                detail="Invalid activity level. Choose from 'sedentary', 'light', 'moderate', 'active', 'very active'."
            )
        tdee = bmr * activity_multipliers[activity_level]

        # Calculate weekly deficit
        daily_deficit = tdee - 500
        weekly_deficit = daily_deficit * 7

        # Calculate total calories burned per hour and sort activities by efficiency
        exercise_data['Calories per Hour'] = exercise_data[["130 lb", "155 lb", "180 lb", "205 lb"]].mean(axis=1)
        sorted_exercises = exercise_data.sort_values(by='Calories per Hour', ascending=False)

        # Generate tailored weekly exercise plan
        weekly_exercise_plan = []
        remaining_calories = weekly_deficit

        for _, row in sorted_exercises.iterrows():
            if remaining_calories <= 0:
                break

            activity = row["Activity, Exercise or Sport (1 hour)"]
            calories_per_hour = row['Calories per Hour']

            # Allocate dynamic time per activity
            max_weekly_duration = min(remaining_calories / calories_per_hour, 4)  # Max 4 hours per week per activity
            daily_duration = (max_weekly_duration / 7) * 60  # Convert to minutes

            weekly_exercise_plan.append({
                "activity": activity,
                "calories_burned_per_hour": round(calories_per_hour, 2),
                "weekly_duration_hours": round(max_weekly_duration, 2),
                "daily_duration_minutes": round(daily_duration, 1),
            })

            remaining_calories -= max_weekly_duration * calories_per_hour

        return {
            "message": "Weekly exercise plan created successfully.",
            "daily_calorie_intake": round(daily_deficit, 2),
            "weekly_exercise_plan": weekly_exercise_plan
        }

    except Exception as e:
        return {"error": str(e)}

# Run the app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
