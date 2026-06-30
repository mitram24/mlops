import sys
import os
sys.path.insert(0, os.path.abspath('src'))
try:
    from mlops_player_rating.pipeline_registry import register_pipelines
    print("SUCCESS: ", register_pipelines().keys())
except Exception as e:
    import traceback
    traceback.print_exc()
