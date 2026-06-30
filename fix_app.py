import sys

app_path = r"C:\Users\pedro\Documentos\NOVA IMS\2º semestre- eu\MLOps\projeto\_player_rating_extracted\mlops_player_rating\src\mlops_player_rating\serving\app.py"

with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

import re

# 1. Inject schema class
schema = '''class PlayerRecord(BaseModel):
    player_api_id: int
    date: str
    height: float
    weight: float
    preferred_foot: str
    attacking_work_rate: str
    defensive_work_rate: str
    crossing: float = None
    finishing: float = None
    heading_accuracy: float = None
    short_passing: float = None
    volleys: float = None
    dribbling: float = None
    curve: float = None
    free_kick_accuracy: float = None
    long_passing: float = None
    ball_control: float = None
    acceleration: float = None
    sprint_speed: float = None
    agility: float = None
    reactions: float = None
    balance: float = None
    shot_power: float = None
    jumping: float = None
    stamina: float = None
    strength: float = None
    long_shots: float = None
    aggression: float = None
    interceptions: float = None
    positioning: float = None
    vision: float = None
    penalties: float = None
    marking: float = None
    standing_tackle: float = None
    sliding_tackle: float = None
    gk_diving: float = None
    gk_handling: float = None
    gk_kicking: float = None
    gk_positioning: float = None
    gk_reflexes: float = None

class PredictionRequest(BaseModel):
    records: list[PlayerRecord]'''

content = re.sub(r'class PredictionRequest.*?records: list\[dict\[str, Any\]\]', schema, content, flags=re.DOTALL)

# 2. Add try-except to predict
content = content.replace("df = pd.DataFrame(request.records)", "try:\n        df = pd.DataFrame([r.model_dump() for r in request.records])")

# 3. Add except clause
content = content.replace("return {\"predictions\": predictions.tolist()}", "return {\"predictions\": predictions.tolist()}\n    except Exception as e:\n        from fastapi import HTTPException\n        raise HTTPException(status_code=400, detail=str(e))")

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)

print("FastAPI schema updated successfully.")
