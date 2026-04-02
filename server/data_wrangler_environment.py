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
        
        # Priority 5 - Dynamic Hugging Face or CSV Datasets
        # If the user defines an external dataset via env var, load that instead.
        dataset_source = os.environ.get("DATASET_SOURCE")
        target_source = os.environ.get("TARGET_SOURCE")
        
        if dataset_source:
            if str(dataset_source).endswith(".csv"):
                self.df = pd.read_csv(dataset_source)
            elif str(dataset_source).endswith(".parquet"):
                self.df = pd.read_parquet(dataset_source)
            else:
                from datasets import load_dataset
                # Fallback to Hugging Face Hub (e.g. "scikit-learn/titanic", "argilla/news-summary")
                # We grab the 'train' split by default and convert it to pandas
                hf_data = load_dataset(dataset_source, split="train")
                self.df = hf_data.to_pandas()
                
            if target_source:
                if str(target_source).endswith(".csv"):
                    self.target_df = pd.read_csv(target_source)
                elif str(target_source).endswith(".parquet"):
                    self.target_df = pd.read_parquet(target_source)
                else:
                    from datasets import load_dataset
                    hf_target = load_dataset(target_source, split="train")
                    self.target_df = hf_target.to_pandas()
            else:
                # If there's no target provided, we force the LLM to simply drop all rows with missing values
                # as a baseline goal for Dynamic runs to prevent graded failures on unstructured tests
                self.target_df = self.df.dropna()
            return

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
        elif self.task_level == 3:
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
        else:
            # Level 4: Extremely Hard / Expanded Tools
            self.df = pd.DataFrame({
                "transaction_info": ["TXN-101 (2020/01/15)", "TXN-102 (2020/01/16)", "TXN-103 (2020/01/16)"],
                "customer_category": ["A", "B", "A"],
                "amount": ["$500.5", "$250.0", "$300.0"]
            })
            self.target_df = pd.DataFrame({
                "customer_category": ["A", "B"],
                "amount": [800.5, 250.0] # Sum'd up
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
            
            elif action.action_type == "extract_regex":
                col = action.target_column
                new_col = action.new_name
                pattern = action.regex_pattern
                if col in self.df.columns:
                    # extract the first capture group
                    extracted = self.df[col].astype(str).str.extract(pattern)[0]
                    self.df[new_col] = extracted
                    reward = 0.1
                else:
                    feedback = f"Error: Column '{col}' not found."
                    
            elif action.action_type == "datetime_parse":
                col = action.target_column
                fmt = action.format_string
                if col in self.df.columns:
                    self.df[col] = pd.to_datetime(self.df[col], format=fmt)
                    reward = 0.1
                else:
                    feedback = f"Error: Column '{col}' not found."
                    
            elif action.action_type == "group_by_aggregate":
                group_col = action.target_column
                agg_col = action.agg_column
                func = action.agg_func
                if group_col in self.df.columns and agg_col in self.df.columns:
                    self.df = self.df.groupby(group_col, as_index=False).agg({agg_col: func})
                    reward = 0.2
                else:
                    feedback = f"Error: Columns '{group_col}' or '{agg_col}' not found."
                    
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
        
        # Priority 2: Partial credit per correct column (name + dtype + values)
        correct_columns = 0
        target_cols = set(self.target_df.columns)
        current_cols = set(self.df.columns)
        
        for col in target_cols:
            if col in current_cols:
                try:
                    # Check dtype match
                    if self.df[col].dtype == self.target_df[col].dtype:
                        # Check value match
                        if (self.df[col].equals(self.target_df[col])):
                            correct_columns += 1
                except:
                    pass
                    
        # Max score from matching columns is 0.8 (leaving 0.2 for efficiency)
        column_score = (correct_columns / max(len(target_cols), 1)) * 0.8
        score += column_score
        
        # Priority 2: Step efficiency bonus
        # If solved in few steps, give up to 0.2 bonus
        ideal_steps = len(target_cols) # rough estimate
        if self._state.step_count <= ideal_steps + 2:
            score += 0.2
        elif self._state.step_count <= ideal_steps + 5:
            score += 0.1
            
        return min(max(score, 0.0), 1.0)

    @property
    def state(self) -> State:
        return self._state
