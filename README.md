demo of the app:
[Watch the video](https://www.dropbox.com/scl/fi/97v7twc83viks3wrej08q/DemoWeightLossCoach.mp4?rlkey=xxqep9to4y3nsjex148318lnz&st=edqjwzwl&dl=0)

# Weight Loss Coach

**Weight Loss Coach** is a FastAPI-based application designed to calculate personalized daily calorie recommendations and suggest tailored exercise routines for weight loss. The app leverages a Hugging Face model (Llama 2) to provide insightful descriptions of exercises.

## Features

- **Daily Calorie Recommendation**: Calculates your Total Daily Energy Expenditure (TDEE) and adjusts it for a 500-calorie deficit.
- **Personalized Exercise Plan**: Suggests exercises tailored to your activity level, each burning 500 kcal daily.
- **AI-Powered Insights**: Provides concise, AI-generated explanations for each exercise using the Llama 2 model.

---

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Amir-Shahriari/WeightLossCoach.git
   cd WeightLossCoach
   ```

2. **Install Dependencies**:
   Make sure you have Python 3.8+ installed. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Data**:
   Ensure the file `exercise_dataset.csv` is in the root directory of the project. This file should include columns for activities and calorie expenditure for various weights.

4. **Set Up the Hugging Face Model**:
   - The app uses the Hugging Face `meta-llama/Llama-2-7b-chat-hf` model for exercise insights.
   - If you have a GPU, configure the pipeline to use `device=0` for faster text generation.

---

## Usage

1. **Run the API**:
   Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

2. **Access the API**:
   Open your browser and go to `http://127.0.0.1:8000/docs` to explore the interactive Swagger UI for testing the endpoints.

3. **Endpoints**:

   - **`GET /`**: Welcome message.
   - **`POST /calculate_and_recommend_daily/`**: Calculates daily calorie recommendations and suggests a tailored exercise plan.

---

## Input Details

The `/calculate_and_recommend_daily/` endpoint expects a JSON object with the following fields:

- `weight` (float): User's weight in kilograms.
- `height` (float): User's height in centimeters.
- `age` (int): User's age in years.
- `gender` (string): User's gender (`"male"` or `"female"`).
- `activity_level` (string): User's activity level (`"sedentary"`, `"light"`, `"moderate"`, `"active"`, `"very active"`).

### Example Request:
```json
{
  "weight": 70,
  "height": 175,
  "age": 30,
  "gender": "male",
  "activity_level": "moderate"
}
```

### Example Response:
```json
{
  "message": "Daily exercise plan created successfully (each exercise burns 500 kcal/day on its own).",
  "recommended_daily_calorie_intake": 2000.0,
  "daily_exercise_plan": [
    {
      "id": 0,
      "activity": "Running",
      "calories_burned_per_hour": 600,
      "daily_burn_allocation": 500,
      "daily_duration_hours": 0.83,
      "daily_duration_minutes": 50,
      "weekly_duration_hours": 5.83,
      "weekly_duration_minutes": 350,
      "insights": "Running is a great cardio exercise to improve endurance..."
    }
  ]
}
```

---

## Technical Details

- **Framework**: FastAPI
- **Data Files**: 
  - `exercise_dataset.csv`: Contains exercise details and calorie data.
- **AI Model**: Hugging Face Llama 2 (`meta-llama/Llama-2-7b-chat-hf`)
- **Middleware**: CORS enabled for cross-origin resource sharing.

---

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any feature suggestions or improvements.

