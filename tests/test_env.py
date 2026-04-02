import pytest
from server.data_wrangler_environment import DataWranglerEnvironment
from models import DataWranglerAction

def test_environment_reset():
    env = DataWranglerEnvironment()
    obs = env.reset()
    assert obs.columns == ["User Name", "Unnamed: 0", "Age"]
    assert obs.row_count == 3
    assert not obs.is_done

def test_drop_action_scoring():
    env = DataWranglerEnvironment()
    env.reset()
    # It should penalize dropping User Name
    action = DataWranglerAction(action_type="drop_column", target_column="User Name")
    obs = env.step(action)
    assert "User Name" not in obs.columns
    assert "Warning" in obs.last_action_feedback or "Error" in obs.last_action_feedback
    
def test_successful_grading():
    import os
    os.environ["TASK_LEVEL"] = "1"
    env = DataWranglerEnvironment()
    env.reset()
    
    # 1. Drop Unnamed: 0
    env.step(DataWranglerAction(action_type="drop_column", target_column="Unnamed: 0"))
    
    # 2. Rename User Name
    env.step(DataWranglerAction(action_type="rename_column", target_column="User Name", new_name="user_name"))
    
    # 3. Rename Age
    env.step(DataWranglerAction(action_type="rename_column", target_column="Age", new_name="age"))
    
    # 4. Submit
    obs = env.step(DataWranglerAction(action_type="submit"))
    assert obs.is_done
    assert obs.reward > 0.8  # partial credit + efficiency
