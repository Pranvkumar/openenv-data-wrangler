import os
import asyncio
import json
import re
from openai import AsyncOpenAI

try:
    from models import DataWranglerAction
except (ImportError, ModuleNotFoundError):
    import sys
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    from models import DataWranglerAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "data_wrangler")
TASK_NAME = "data_wrangler_task"
BENCHMARK = "data_wrangler"
MAX_STEPS = 15
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5
MAX_HISTORY_ITEMS = int(os.environ.get("MAX_HISTORY_ITEMS", "6"))

system_prompt = """\
SYSTEM INSTRUCTIONS: ELITE DATA ENGINEER AGENT

ROLE AND PERSONA
You are an elite Data Engineering AI Agent operating within an automated data-wrangling pipeline. Your core function is to autonomously clean, format, and standardize messy, real-world datasets until they perfectly match a hidden "ground truth" target. You operate systematically, analytically, and with absolute precision.

MISSION OBJECTIVE
At each step, you will receive an Observation of the current data state. You must analyze the data anomalies (missing values, bad schemas, incorrect data types) and issue exactly ONE valid operation from your Action Space. You will iterate on this process until the dataset is perfectly clean, at which point you will issue the submit action.

THE OBSERVATION
You will receive a state dictionary detailing the dataset's current form:
columns: Current list of headers.
row_count: Total number of rows in the dataset.
column_stats: Dictionary mapping column names to {dtype, missing_count, sample_values}.
last_action_feedback: Status/error message resulting from your previous action.
is_done: Boolean termination flag.

ACTION SPACE (AVAILABLE TOOLS)
You have a strict, highly constrained toolset. Your chosen action MUST be a valid JSON object matching exactly ONE of the schemas:
1. Drop Column: {"action_type": "drop_column", "target_column": "..."}
2. Rename Column: {"action_type": "rename_column", "target_column": "...", "new_name": "..."}
3. Fill Missing Values: {"action_type": "fill_missing", "target_column": "...", "fill_value": "..."}
4. Cast Data Type: {"action_type": "cast_type", "target_column": "...", "cast_to": "..."}
5. Extract Regex: {"action_type": "extract_regex", "target_column": "...", "new_name": "...", "regex_pattern": "..."}
6. Parse Datetime: {"action_type": "datetime_parse", "target_column": "...", "format_string": "..."}
7. Group By & Aggregate: {"action_type": "group_by_aggregate", "target_column": "...", "agg_column": "...", "agg_func": "sum|mean|count"}
8. Submit: {"action_type": "submit"}

REQUIRED OUTPUT FORMAT (CHAIN OF THOUGHT)
<thinking>
Analyze Observation: What is the current state? What did the last action do?
Identify Anomalies: Which columns have wrong types, bad names, or missing data?
Formulate Plan: What is the highest priority fix right now?
Select Action: Which action type and parameters will execute this fix?
</thinking>
{
"action_type": "...",
...
}
"""

async def get_model_message(client, step, obs_dict, last_reward, history, max_retries=3):
    obs_text = str(obs_dict)
    trimmed_history = history[-MAX_HISTORY_ITEMS:] if history else []
    prompt = (
        f"Step {step}.\nObservation: {obs_text}\nLast Reward: {last_reward}\n"
        f"History: {trimmed_history}\nChoose your next action (JSON matching schema)."
    )
    
    # Priority 3: Error Reflection. Pass previous feedback directly to LLM if there was an error.
    if "Error" in obs_dict.get("last_action_feedback", "") or "Exception" in obs_dict.get("last_action_feedback", ""):
        prompt += f"\nCRITICAL: Your last action failed with this error: {obs_dict['last_action_feedback']}. Review your <thinking> block to correct your mistake before trying a new action."

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=220,
            )
            content = response.choices[0].message.content

            match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', content or "", re.DOTALL)
            if match:
                return json.loads(match.group(1))
            else:
                prompt += f"\nWarning: Failed to extract JSON on attempt {attempt+1}. Provide ONLY valid JSON inside curly braces."
        except Exception as e:
            prompt += f"\nWarning: Exception on attempt {attempt+1}: {str(e)}. Provide valid JSON."
            
    # Fallback only if absolutely all retries fail
    return {"action_type": "submit"}

def _bool_str(value):
    return "true" if bool(value) else "false"


def _action_str(action):
    try:
        return json.dumps(action, separators=(",", ":"), ensure_ascii=False)
    except Exception:
        return str(action).replace("\n", " ")


def _reward_str(value):
    try:
        return f"{float(value):.2f}"
    except Exception:
        return "0.00"


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}")


def log_step(step, action, reward, done, error):
    error_str = "null" if error is None else str(error).replace("\n", " ")
    print(
        f"[STEP] step={step} action={_action_str(action)} "
        f"reward={_reward_str(reward)} done={_bool_str(done)} error={error_str}"
    )


def log_end(success, steps, rewards):
    rewards_csv = ",".join(_reward_str(r) for r in rewards)
    print(f"[END] success={_bool_str(success)} steps={steps} rewards={rewards_csv}")

async def main():
    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    if not HF_TOKEN:
        log_end(success=False, steps=0, rewards=[])
        return

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    try:
        from client import DataWranglerEnv
        env = DataWranglerEnv.from_docker_image(LOCAL_IMAGE_NAME)
    except Exception:
        from server.data_wrangler_environment import DataWranglerEnvironment
        env = DataWranglerEnvironment()

    history = []
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        if hasattr(env, 'reset') and not asyncio.iscoroutinefunction(env.reset):
            result = env.reset()
        else:
            result = await env.reset() # OpenENV.reset() as per sample

        obs = getattr(result, "observation", result)
        obs_dict = {
            "columns": getattr(obs, "columns", []),
            "row_count": getattr(obs, "row_count", 0),
            "column_stats": getattr(obs, "column_stats", {}),
            "last_action_feedback": getattr(obs, "last_action_feedback", ""),
            "is_done": getattr(obs, "is_done", False)
        }
        last_reward = getattr(result, "reward", getattr(obs, "reward", 0.0)) or 0.0

        for step in range(1, MAX_STEPS + 1):
            done = getattr(result, "done", getattr(obs, "is_done", False))
            if done:
                break

            action_data = await get_model_message(client, step, obs_dict, last_reward, history)

            action_obj = DataWranglerAction(**action_data)

            if hasattr(env, 'step') and not asyncio.iscoroutinefunction(env.step):
                result = env.step(action_obj)
            else:
                result = await env.step(action_obj)

            obs = getattr(result, "observation", result)
            obs_dict = {
                "columns": getattr(obs, "columns", []),
                "row_count": getattr(obs, "row_count", 0),
                "column_stats": getattr(obs, "column_stats", {}),
                "last_action_feedback": getattr(obs, "last_action_feedback", ""),
                "is_done": getattr(obs, "is_done", False)
            }

            reward = getattr(result, "reward", getattr(obs, "reward", 0.0)) or 0.0
            done = getattr(result, "done", getattr(obs, "is_done", False))
            feedback = obs_dict.get("last_action_feedback", "")
            error = feedback if ("Error" in feedback or "Exception" in feedback) else None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_data, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_data} -> reward {reward:+.2f}")

            if done:
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            if hasattr(env, 'close'):
                if asyncio.iscoroutinefunction(env.close):
                    await env.close()
                else:
                    env.close()
        except Exception as e:
            _ = e

        log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())