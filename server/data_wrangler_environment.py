import os
import pandas as pd
from uuid import uuid4
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import DataWranglerAction, DataWranglerObservation
except (ImportError, ValueError, ModuleNotFoundError):
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from models import DataWranglerAction, DataWranglerObservation

class DataWranglerEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self.df = None
        self.target_df = None
        self.task_level = int(os.environ.get("TASK_LEVEL", "1"))
        self._initialize_task()

    def _initialize_task(self):
        self.df = pd.DataFrame()
        self.target_df = pd.DataFrame()
        if self.task_level == 1:
            # Easy: Just drop a column and rename one
            self.df = pd.DataFrame({
                "User Name": ["Alice", "Bob", "Charlie"],
                "Unnamed: 0": [0, 1, 2],
                "Age": [25, 30, 35]
            })
            self.target_df = pd.DataFrame({
                "user_name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35]
            })
        elif self.task_level == 2:
            # Medium: fill missing and cast type
            self.df = pd.DataFrame({
                "product_ID ": ["101", "102", "103"],
                "price": ["10.5", None, "12.0"],
                "bad_col": [None, None, None]
            })
            self.target_df = pd.DataFrame({
                "product_id": [101.0, 102.0, 103.0], 
                "price": [10.5, 0.0, 12.0]
            })
        else:
            # Hard: Multiple issues
            self.df = pd.DataFrame({
                "date_joined ": ["2020-01-01", "2021-05-15", None],
                "Sales_total": ["100", "200", "300"],
                "IsActive": [True, False, None],
                "DROPME_1": [1,2,3]
            })
            self.target_df = pd.DataFrame({
                "date_joined": [pd.Timestamp("2020-01-01"), pd.Timestamp("2021-05-15"), pd.Timestamp("1970-01-01")],
                "sales_total": [100.0, 200.0, 300.0],
                "is_active": [True, False, False]
            })

    def _get_obs(self, feedback: str = "Environment initialized.", done: bool = False, reward: float = 0.0) -> DataWranglerObservation:
        stats = {}
        for col in self.df.columns:
            stats[col] = {
                "dtype": str(self.df[col].dtype),
                "missing_count": int(self.df[col].isna().sum()),
                "sample_values": self.df[col].dropna().astype(str).tolist()[:3]
            }
        
        return DataWranglerObservation(
            columns=list(self.df.columns),
            row_count=len(self.df),
            column_stats=stats,
            last_action_feedback=feedback,
            is_done=done,
            reward=reward,
            done=done,
            metadata={"step": self._state.step_count}
        )

    def reset(self) -> DataWranglerObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        self._initialize_task()
        return self._get_obs()

    def step(self, action: DataWranglerAction) -> DataWranglerObservation: # type: ignore
        self._state.step_count += 1
        feedback = "Action executed successfully."
        reward = 0.0
        done = False
        
        try:
            if action.action_type == "drop_column":
                col = action.target_column
                if col in self.df.columns:
                    self.df.drop(columns=[col], inplace=True)
                    if col not in self.target_df.columns:
                        reward = 0.2
                    else:
                        reward = -0.5
                        feedback = f"Warning: dropped targeting column {col}"
                else:
                    feedback = f"Error: Column '{col}' not found."
            
            elif action.action_type == "rename_column":
                col = action.target_column
                new_col = action.new_name
                if col in self.df.columns:
                    self.df.rename(columns={col: new_col}, inplace=True)
                    if new_col in self.target_df.columns:
                        reward = 0.2
                else:
                    feedback = f"Error: Column '{col}' not found."
                    
            elif action.action_type == "fill_missing":
                col = action.target_column
                if col in self.df.columns:
                    self.df[col].fillna(action.fill_value, inplace=True)
                    reward = 0.1
                else:
                    feedback = f"Error: Column '{col}' not found."
                    
            elif action.action_type == "cast_type":
                col = action.target_column
                to_type = action.cast_to
                if col in self.df.columns:
                    if to_type == 'int':
                        self.df = self.df.astype({col: int})
                    elif to_type == 'float':
                        self.df = self.df.astype({col: float})
                    elif to_type == 'datetime':
                        self.df[col] = pd.to_datetime(self.df[col])
                    elif to_type == 'string':
                        self.df = self.df.astype({col: str})
                    reward = 0.2
                else:
                    feedback = f"Error: Column '{col}' not found."
                    
            elif action.action_type == "submit":
                score = self._grade()
                reward = score
                feedback = f"Submitted. Final Score: {score}"
                done = True
            else:
                feedback = f"Error: Unknown action type {action.action_type}"
                
        except Exception as e:
            feedback = f"Exception occurred: {str(e)}"
            reward = -0.1

        return self._get_obs(feedback=feedback, done=done, reward=reward)

    def _grade(self) -> float:
        score = 0.0
        if list(self.df.columns) == list(self.target_df.columns):
            score += 0.5
             # Match types and values
            value_matches = 0
            for col in self.df.columns:
                try:
                    # simple match check
                    match = (self.df[col] == self.target_df[col]).all()
                    if match:
                        value_matches += 1
                except:
                    pass
            score += 0.5 * (value_matches / max(len(self.target_df.columns), 1))
            
        return score

    @property
    def state(self) -> State:
        return self._state
