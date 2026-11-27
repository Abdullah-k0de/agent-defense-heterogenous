"""
Phase 2 (Step 1): Benchmark Seed & Tool Schema Ingestion
Downloads InjecAgent seed cases and tool schemas if absent, maps tool schemas,
and outputs step1_prepared_data.json for payload synthesis.
"""
import json
import urllib.request
from pathlib import Path
import pandas as pd

PHASE2_DIR = Path(__file__).resolve().parent
ROOT_DIR = PHASE2_DIR.parent


def resolve_input(filename: str) -> Path:
    local_candidate = PHASE2_DIR / filename
    root_candidate = ROOT_DIR / filename
    if local_candidate.exists():
        return local_candidate
    return root_candidate


def main():
    print("--- Step 1: Data & Schema Ingestion ---")

    files_to_download = {
        "user_cases.jsonl": "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/user_cases.jsonl",
        "tools.json": "https://raw.githubusercontent.com/uiuc-kang-lab/InjecAgent/main/data/tools.json",
    }

    for filename, url in files_to_download.items():
        target = resolve_input(filename)
        if not target.exists():
            print(f"Downloading {filename} to {target}...")
            urllib.request.urlretrieve(url, target)
        else:
            print(f"Found {filename} at {target}.")

    with open(resolve_input("tools.json"), "r", encoding="utf-8") as f:
        toolkits = json.load(f)

    tools_dict = {}
    for toolkit in toolkits:
        toolkit_name = toolkit.get("name_for_model", "")
        for tool in toolkit.get("tools", []):
            concatenated_name = f"{toolkit_name}{tool['name']}"
            tools_dict[concatenated_name] = tool

    print(f"Mapped {len(tools_dict)} API tool schemas.")

    user_cases = pd.read_json(resolve_input("user_cases.jsonl"), lines=True)
    print(f"Loaded {len(user_cases)} benign parent user tasks.")

    prepared_data = []
    for idx, row in user_cases.iterrows():
        user_tool = row.get("User Tool", "")
        prepared_data.append({
            "task_id": f"task_{idx}",
            "user_tool": user_tool,
            "user_instruction": row.get("User Instruction", ""),
            "response_template": row.get("Tool Response Template", ""),
            "tool_schema": tools_dict.get(user_tool, {})
        })

    out_path = PHASE2_DIR / "step1_prepared_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prepared_data, f, indent=4)

    print(f"Saved prepared seed tasks to '{out_path}'.")


if __name__ == "__main__":
    main()
