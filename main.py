from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load calorie knowledge data
current_dir = os.path.dirname(os.path.abspath(__file__))
calorie_file_path = os.path.join(current_dir, "calorie_knowledge.csv")
exercise_file_path = os.path.join(current_dir, "exercises_calories.csv")

if not os.path.exists(calorie_file_path):
    raise FileNotFoundError("The file 'calorie_knowledge.csv' does not exist.")
if not os.path.exists(exercise_file_path):
    raise FileNotFoundError("The file 'exercises_calories.csv' does not exist.")

calorie_data = pd.read_csv(calorie_file_path)
exercise_data = pd.read_csv(exercise_file_path)

# Check if required columns exist in the datasets
required_columns_calories = {"Topic", "Details"}
required_columns_exercises = {"Exercise/Activity", "Calories Burned (per hour)"}

if not required_columns_calories.issubset(calorie_data.columns):
    raise ValueError(f"The calorie_knowledge.csv file must contain the following columns: {required_columns_calories}")
if not required_columns_exercises.issubset(exercise_data.columns):
    raise ValueError(f"The exercises_calories.csv file must contain the following columns: {required_columns_exercises}")

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity, modify as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Hugging Face sentence transformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize FAISS Index with the correct dimensionality
sample_embedding_dim = embedding_model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(sample_embedding_dim)

# Add calorie knowledge data to the FAISS index
def preprocess_and_embed_data(data):
    embeddings = []
    metadata = []
    for i, row in data.iterrows():
        text = f"{row['Topic']} {row['Details']}"
        embedding = embedding_model.encode(text)
        embeddings.append(embedding)
        metadata.append(row.to_dict())
    embeddings = np.array(embeddings, dtype="float32")
    
    # Add embeddings to the FAISS index
    index.add(embeddings)
    return metadata

metadata = preprocess_and_embed_data(calorie_data)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Weight Loss Coach"}

from pydantic import BaseModel

class UserDetails(BaseModel):
    weight: float
    height: float
    age: int
    gender: str
    activity_level: str

@app.post("/calculate_and_recommend_weekly/")
def calculate_and_recommend_weekly(details: UserDetails):
    # Unpack details into variables
    weight = details.weight
    height = details.height
    age = details.age
    gender = details.gender
    activity_level = details.activity_level

    print("Received details:", details.dict())  # Log incoming data
    
    try:
        # Calculate BMR using the Mifflin-St Jeor Equation
        if gender.lower() == "male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        elif gender.lower() == "female":
            bmr = 10 * weight + 6.25 * height - 5 * age - 161
        else:
            raise HTTPException(status_code=400, detail="Invalid gender. Use 'male' or 'female'.")

        # Adjust BMR with activity multiplier
        activity_multipliers = {
            "sedentary": 1.2,
            "light": 1.375,
            "moderate": 1.55,
            "active": 1.725,
            "very active": 1.9,
        }
        if activity_level.lower() not in activity_multipliers:
            raise HTTPException(
                status_code=400,
                detail="Invalid activity level. Choose from 'sedentary', 'light', 'moderate', 'active', 'very active'."
            )
        tdee = bmr * activity_multipliers[activity_level.lower()]

        # Calculate weekly calorie deficit
        daily_deficit = tdee - 500
        weekly_deficit = daily_deficit * 7

        # Generate a weekly exercise plan
        weekly_exercise_plan = []
        remaining_calories = weekly_deficit

        for _, row in exercise_data.iterrows():
            if remaining_calories <= 0:
                break

            exercise = row["Exercise/Activity"]
            calories_per_hour = row["Calories Burned (per hour)"]

            # Allocate up to one hour per day for the exercise
            max_weekly_calories = calories_per_hour * 7  # Maximum calories burned in 7 hours
            calories_to_burn = min(remaining_calories, max_weekly_calories)
            duration_per_week = calories_to_burn / calories_per_hour  # Total hours for the week
            duration_per_day = min(duration_per_week / 7 * 60, 60)  # Max 60 minutes per day

            weekly_exercise_plan.append({
                "exercise": exercise,
                "calories_burned_per_hour": calories_per_hour,
                "weekly_duration_hours": round(duration_per_week, 2),
                "daily_duration_minutes": round(duration_per_day, 1)
            })

            # Reduce remaining calories
            remaining_calories -= calories_to_burn

        return {
            "message": "Weekly exercise plan created successfully.",
            "daily_calorie_intake": daily_deficit,
            "weekly_exercise_plan": weekly_exercise_plan
        }

    except Exception as e:
        return {"error": str(e)}



# Ensure the app runs dynamically on the correct port
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
