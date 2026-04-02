import os
import sys
import asyncio
from openai import AsyncOpenAI

# OpenEnv V5 specific client components
# We import directly since OpenEnv varies slightly in versions, but this mirrors the validator script expectations.
try:
    from openenv.core.client import EnvClient
except ImportError:
    pass

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
IMAGE_NAME = "data_wrangler"
TASK_NAME = "Data Writer Level 1"
BENCHMARK = "data_wrangler"
MAX_STEPS = 10
MAX_TOTAL_REWARD = 1.0
SUCCESS_SCORE_THRESHOLD = 0.5

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
5. Submit: {"action_type": "submit"}

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

async def get_model_message(client, step, obs_dict, last_reward, history):
    obs_text = str(obs_dict)
    prompt = f"Step {step}.\nObservation: {obs_text}\nLast Reward: {last_reward}\nHistory: {history}\nChoose your next action (JSON matching schema)."
    try:
        response = await client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content
        import json
        import re
        # Basic parsing of the JSON structure that follows the thinking tags
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # Fallback if unparseable
        return {"action_type": "submit"}
    except Exception as e:
        return {"action_type": "submit"}

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}")

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward} done={done} error={error}")

def log_end(success, steps, score, rewards):
    print(f"[END] success={success} steps={steps} score={score} rewards={rewards}")

async def main():
    if not API_KEY:
        print("Missing OPENAI_API_KEY environment variable.")
        return

    client = AsyncOpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    
    print(f"[DEBUG] Spinning up {IMAGE_NAME} environment container...", flush=True)
    try:
        from client import DataWranglerEnv
        env = DataWranglerEnv.from_docker_image(IMAGE_NAME)
    except Exception as e:
        print(f"[DEBUG] Docker env start failed ({e}). Falling back to local direct Python import.", flush=True)
        from server.data_wrangler_environment import DataWranglerEnvironment
        env = DataWranglerEnvironment() # Fallback for local debugging

    history = []
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

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
            
            from models import DataWranglerAction
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
            error = None

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
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
            
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())