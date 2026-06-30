import os

utils_path = r"C:\Users\pedro\Documentos\NOVA IMS\2º semestre- eu\MLOps\projeto\_player_rating_extracted\mlops_player_rating\src\mlops_player_rating\core\utils.py"
app_path = r"C:\Users\pedro\Documentos\NOVA IMS\2º semestre- eu\MLOps\projeto\_player_rating_extracted\mlops_player_rating\src\mlops_player_rating\serving\app.py"

with open(utils_path, "r", encoding="utf-8") as f:
    content = f.read()

# Remove the imputers
import re
content = re.sub(r'def fit_attribute_imputer.*?return medians', '', content, flags=re.DOTALL)
content = re.sub(r'def apply_attribute_imputer.*?return df', '', content, flags=re.DOTALL)

# Add multicollinearity fix to engineer_features
drop_skills = "\n    # Drop raw skill columns to prevent perfect multicollinearity and over-dimensionality.\n    df = df.drop(columns=[c for c in SKILL_COLUMNS if c in df.columns])\n\n    return df"
content = content.replace("return df", drop_skills, 1)

# Fix preprocess_for_inference
content = re.sub(r'    attribute_medians: dict\[str, float\],\n', '', content)
content = content.replace("df = apply_attribute_imputer(df, attribute_medians)\n    ", "")

with open(utils_path, "w", encoding="utf-8") as f:
    f.write(content)

with open(app_path, "r", encoding="utf-8") as f:
    app_content = f.read()

# Update app.py to not pass attribute_medians
app_content = re.sub(r'attribute_medians=models\["attribute_medians"\],\n        ', '', app_content)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(app_content)

