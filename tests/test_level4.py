import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from server.data_wrangler_environment import DataWranglerEnvironment
from models import DataWranglerAction

def test_level_4():
    os.environ["TASK_LEVEL"] = "4"
    env = DataWranglerEnvironment()
    env.reset()
    
    # 1. regex to extract number. 
    # Strings: "$500.5" -> pattern "^\$?([0-9.]+)"
    env.step(DataWranglerAction(action_type="extract_regex", target_column="amount", new_name="amount", regex_pattern=r"^\$?([0-9.]+)"))
    
    # 2. Cast amount to float
    env.step(DataWranglerAction(action_type="cast_type", target_column="amount", cast_to="float"))
    
    # 3. Group by customer_category and sum amount
    env.step(DataWranglerAction(action_type="group_by_aggregate", target_column="customer_category", agg_column="amount", agg_func="sum"))
    
    obs = env.step(DataWranglerAction(action_type="submit"))
    assert obs.reward > 0.8 # It should grade highly because the DFs will match exactly!
