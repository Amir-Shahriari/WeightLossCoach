from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import numpy as np
from pydantic import BaseModel
import random

# Import huggingface transformers
from transformers import pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "exercise_dataset.csv")

if not os.path.exists(dataset_path):
    raise FileNotFoundError("The file 'exercise_dataset.csv' does not exist.")

exercise_data = pd.read_csv(dataset_path)

# Ensure required columns exist
required_columns = {"Activity, Exercise or Sport (1 hour)", "130 lb", "155 lb", "180 lb", "205 lb"}
if not required_columns.issubset(exercise_data.columns):
    raise ValueError(f"The dataset must contain the following columns: {required_columns}")

# ----------------------------------------------------------------
# Initialize Llama 2 text-generation pipeline
# Adjust the model name if you have a different version or a quantized model
# ----------------------------------------------------------------
MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

try:
    text_generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        # If you have a GPU, uncomment 'device=0' for faster generation:
        # device=0,
        # trust_remote_code=True,
        max_new_tokens=200,   # increased to allow more complete answers
        truncation=True,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        num_return_sequences=1
    )
except Exception as e:
    print(f"Error loading the Llama 2 model '{MODEL_NAME}': {e}")
    text_generator = None  # fallback if model fails to load

def generate_exercise_definition(exercise_name: str) -> str:
    """
    Generates a short definition or explanation of the exercise using a local HF Llama 2 model.
    If the model is unavailable, returns a default fallback.
    """
    if text_generator is None:
        return f"[No local Llama 2 model loaded. Cannot generate definition for: {exercise_name}]"

    # Prompt that strongly requests a complete, short conclusion
    prompt = (
        f"You are a knowledgeable fitness coach. A user wants a concise explanation "
        f"of how to do '{exercise_name}' safely and effectively. Provide 2-3 sentences "
        "covering the main steps, safety tips, and a final concluding remark. "
        "The response should end cleanly without trailing phrases."
    )

    try:
        result = text_generator(prompt)
        # result is a list of dicts like: [{'generated_text': '...'}]
        generation = result[0]["generated_text"]
        # If Llama 2 echoes the entire prompt, consider removing it:
        cleaned = generation.replace(prompt, "").strip()
        return cleaned
    except Exception as e:
        print("Error generating text:", e)
        return f"[No insights available for {exercise_name}. Error: {e}]"

# ---- Pydantic model for user input ---- #
class UserDetails(BaseModel):
    weight: float
    height: float
    age: int
    gender: str
    activity_level: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Weight Loss Coach"}

@app.post("/calculate_and_recommend_daily/")
def calculate_and_recommend_daily(details: UserDetails):
    """
    1) Calculates the user’s TDEE (using weight, height, age, gender, activity_level).
    2) Deducts 500 kcal/day => daily_calorie_intake.
    3) Picks up to 3 exercises from the user's activity_level bucket,
       each enough to burn 500 kcal/day on its own.
    4) Generates a short definition/description from the Llama 2 pipeline for each one.
    """
    try:
        # --- 1. Unpack user details ---
        weight = details.weight
        height = details.height
        age = details.age
        gender = details.gender.strip().lower()
        activity_level = details.activity_level.strip().lower()

        # Validate gender
        if gender not in ["male", "female"]:
            raise HTTPException(status_code=400, detail="Invalid gender. Use 'male' or 'female'.")

        # Activity multipliers
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

        # --- 2. Calculate BMR -> TDEE ---
        if gender == "male":
            bmr = 10 * weight + 6.25 * height - 5 * age + 5
        else:
            bmr = 10 * weight + 6.25 * height - 5 * age - 161

        tdee = bmr * activity_multipliers[activity_level]
        daily_calorie_intake = tdee - 500  # 500 kcal deficit

        # --- 3. Each exercise individually burns 500 kcal/day ---
        daily_exercise_burn = 500

        # --- 4. Prepare the exercise data ---
        exercise_data['Calories per Hour'] = exercise_data[["130 lb", "155 lb", "180 lb", "205 lb"]].mean(axis=1)
        sorted_exercises = exercise_data.sort_values(by='Calories per Hour', ascending=False).reset_index(drop=True)

        # Bucket slicing logic
        n = len(sorted_exercises)
        if n == 0:
            raise HTTPException(status_code=500, detail="No exercise data available.")

        bucket_size = n // 5 if n >= 5 else 1
        b1 = 0
        b2 = bucket_size
        b3 = bucket_size * 2
        b4 = bucket_size * 3
        b5 = bucket_size * 4

        slice_very_active = sorted_exercises.iloc[b1:b2]
        slice_active = sorted_exercises.iloc[b2:b3]
        slice_moderate = sorted_exercises.iloc[b3:b4]
        slice_light = sorted_exercises.iloc[b4:b5]
        slice_sedentary = sorted_exercises.iloc[b5:]

        bucket_map = {
            "very active": slice_very_active,
            "active": slice_active,
            "moderate": slice_moderate,
            "light": slice_light,
            "sedentary": slice_sedentary
        }

        chosen_slice = bucket_map.get(activity_level, sorted_exercises)
        if len(chosen_slice) == 0:
            chosen_slice = sorted_exercises  # fallback

        # --- 5. Pick up to 3 exercises randomly ---
        sample_size = min(3, len(chosen_slice))
        if sample_size == 0:
            return {
                "message": "Daily exercise plan created successfully (each option burns 500 kcal/day).",
                "recommended_daily_calorie_intake": round(daily_calorie_intake, 2),
                "daily_exercise_plan": []
            }

        chosen_exercises = chosen_slice.sample(n=sample_size, replace=False, random_state=random.randint(0,999999))

        # --- 6. Build the plan w/ Llama 2-based "insights" ---
        exercise_plan = []
        for _, row in chosen_exercises.iterrows():
            activity_name = row["Activity, Exercise or Sport (1 hour)"]
            cals_per_hour = row["Calories per Hour"]

            daily_hours = daily_exercise_burn / cals_per_hour
            daily_minutes = daily_hours * 60

            weekly_hours = daily_hours * 7
            weekly_minutes = daily_minutes * 7

            # Generate text from the Llama 2 pipeline
            definition_text = generate_exercise_definition(activity_name)

            exercise_plan.append({
                "activity": activity_name,
                "calories_burned_per_hour": round(cals_per_hour, 2),
                "daily_burn_allocation": daily_exercise_burn,  # always 500
                "daily_duration_hours": round(daily_hours, 2),
                "daily_duration_minutes": round(daily_minutes, 1),
                "weekly_duration_hours": round(weekly_hours, 2),
                "weekly_duration_minutes": round(weekly_minutes, 1),
                "insights": definition_text
            })

        return {
            "message": "Daily exercise plan created successfully (each exercise burns 500 kcal/day on its own).",
            "recommended_daily_calorie_intake": round(daily_calorie_intake, 2),
            "daily_exercise_plan": exercise_plan
        }

    except Exception as e:
        return {"error": str(e)}
