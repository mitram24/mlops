import sys

nodes_path = r"C:\Users\pedro\Documentos\NOVA IMS\2º semestre- eu\MLOps\projeto\_player_rating_extracted\mlops_player_rating\src\mlops_player_rating\pipelines\data_cleaning\nodes.py"
pipeline_path = r"C:\Users\pedro\Documentos\NOVA IMS\2º semestre- eu\MLOps\projeto\_player_rating_extracted\mlops_player_rating\src\mlops_player_rating\pipelines\data_cleaning\pipeline.py"

with open(nodes_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update signature
content = content.replace("-> tuple[pd.DataFrame, dict[str, Any]]:", "-> pd.DataFrame:")
content = content.replace("`(cleaned, attribute_imputer)`:", "`cleaned`:")

# 2. Remove imputer imports and logic
import re
content = re.sub(r'apply_attribute_imputer,\n    apply_value_semantics,\n    fit_attribute_imputer,', 'apply_value_semantics,', content)

imputer_logic = """    # Fit + apply the per-attribute median imputer, and keep it as a feature-store artefact.
    medians = fit_attribute_imputer(df)
    df = apply_attribute_imputer(df, medians)
    attribute_imputer = {"attribute_medians": medians}"""
content = content.replace(imputer_logic, "")

# 3. Remove ID column dropping logic
id_logic = """    # Identifier columns carry no signal and risk leakage - drop them.
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])"""
content = content.replace(id_logic, "")

# 4. Update return statement
content = content.replace("return df, attribute_imputer", "return df")

with open(nodes_path, "w", encoding="utf-8") as f:
    f.write(content)

# Update pipeline.py to remove the second output
with open(pipeline_path, "r", encoding="utf-8") as f:
    pipe_content = f.read()

pipe_content = pipe_content.replace('outputs=["cleaned", "attribute_imputer"],', 'outputs="cleaned",')

with open(pipeline_path, "w", encoding="utf-8") as f:
    f.write(pipe_content)

print("Data cleaning nodes updated successfully.")
