"""
One-shot setup script for the ElevenLabs side of the project.

Reads the 7 tool specs from setup/elevenlabs_tools/, substitutes the
{{BACKEND_URL}} placeholder with the actual ngrok URL from .env, and
creates them on ElevenLabs Convai if they don't already exist. Then
either creates a new agent with the baseline system prompt and these
tools attached, or updates an existing agent to use them.

Idempotent: safe to run multiple times. Tools are matched by name; if a
tool with the same name already exists on the account, it's left alone.
The agent is identified by ELEVENLABS_AGENT_ID in .env; if that env var
is unset or the agent doesn't exist, a new agent is created and its ID
is printed for you to add to .env.

Usage:
    1. Make sure your backend is reachable at BACKEND_URL (typically via ngrok).
    2. Set BACKEND_URL and ELEVENLABS_API_KEY in .env.
    3. Run: python setup/configure_elevenlabs.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Make project root importable so we can pull the baseline prompt.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from dotenv import load_dotenv

# --- Setup --------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = PROJECT_ROOT / "setup" / "elevenlabs_tools"
ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(ENV_FILE)

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
BACKEND_URL = os.getenv("BACKEND_BASE_URL") or os.getenv("BACKEND_URL")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")

if not ELEVENLABS_API_KEY:
    print("ERROR: ELEVENLABS_API_KEY is missing from .env")
    sys.exit(1)
if not BACKEND_URL:
    print("ERROR: BACKEND_BASE_URL (or BACKEND_URL) is missing from .env")
    print("       Set it to your ngrok HTTPS URL, e.g. https://abc123.ngrok-free.dev")
    sys.exit(1)

HEADERS = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}


# --- Helpers ------------------------------------------------------------- #

def request(method: str, path: str, *, json_body: Any = None) -> httpx.Response:
    url = f"{ELEVENLABS_BASE}{path}"
    with httpx.Client(timeout=30.0) as client:
        resp = client.request(method, url, headers=HEADERS, json=json_body)
    return resp


def load_tool_specs() -> list[dict]:
    """Load all 7 tool JSONs, substituting {{BACKEND_URL}}."""
    specs = []
    for f in sorted(TOOLS_DIR.glob("*.json")):
        text = f.read_text(encoding="utf-8")
        text = text.replace("{{BACKEND_URL}}", BACKEND_URL.rstrip("/"))
        specs.append(json.loads(text))
    return specs


def _normalize_for_create(spec: dict) -> dict:
    """
    Convert array-form fields in a dashboard-exported tool spec into the
    dict/object form the ElevenLabs create API expects.

      request_headers:      [{type, name, value}, ...]  → {name: value, ...}
      path_params_schema:   [{id, required, ...}, ...]  → {properties: {id: {...}}, required: [...]}
      query_params_schema:  [{id, required, ...}, ...]  → {properties: {id: {...}}, required: [...]}
    """
    import copy
    out = copy.deepcopy(spec)
    api = out.get("api_schema") or {}

    headers = api.get("request_headers")
    if isinstance(headers, list):
        api["request_headers"] = {h["name"]: h["value"] for h in headers}

    for key in ("path_params_schema", "query_params_schema"):
        params = api.get(key)
        if isinstance(params, list):
            properties = {}
            required = []
            for p in params:
                p_copy = dict(p)
                param_id = p_copy.pop("id")
                if p_copy.pop("required", False):
                    required.append(param_id)
                properties[param_id] = p_copy
            api[key] = {"properties": properties, "required": required}

    out["api_schema"] = api
    return out


def list_existing_tools() -> dict[str, str]:
    """Return {tool_name: tool_id} for all tools on the account."""
    resp = request("GET", "/convai/tools")
    if resp.status_code != 200:
        print(f"Failed to list tools: {resp.status_code} {resp.text}")
        sys.exit(1)
    data = resp.json()
    tools = data.get("tools") or []
    result: dict[str, str] = {}
    for t in tools:
        # Tool name lives in tool_config.name in some versions, or top-level name.
        name = t.get("name") or (t.get("tool_config") or {}).get("name")
        tool_id = t.get("id") or t.get("tool_id")
        if name and tool_id:
            result[name] = tool_id
    return result


def create_tool(spec: dict) -> str | None:
    """Create a tool from a spec, return its ID, or None on failure."""
    normalized = _normalize_for_create(spec)
    payload = {"tool_config": normalized}
    resp = request("POST", "/convai/tools", json_body=payload)
    if resp.status_code not in (200, 201):
        print(f"  ✗ Failed to create tool {spec['name']}: {resp.status_code} {resp.text}")
        return None
    data = resp.json()
    tool_id = data.get("id") or data.get("tool_id")
    if not tool_id:
        print(f"  ✗ Created tool {spec['name']} but no ID in response: {data}")
        return None
    return tool_id


def get_agent(agent_id: str) -> dict | None:
    resp = request("GET", f"/convai/agents/{agent_id}")
    if resp.status_code == 200:
        return resp.json()
    return None


def create_agent(tool_ids: list[str], baseline_prompt: str) -> str:
    """Create a fresh agent with the given tool IDs attached."""
    payload = {
        "name": "Aegis Airlines Customer Service",
        "conversation_config": {
            "agent": {
                "prompt": {
                    "prompt": baseline_prompt,
                    "tool_ids": tool_ids,
                },
                "first_message": "Thank you for calling Aegis Airlines. How can I help you today?",
                "language": "en",
            },
        },
    }
    resp = request("POST", "/convai/agents/create", json_body=payload)
    if resp.status_code not in (200, 201):
        print(f"✗ Failed to create agent: {resp.status_code} {resp.text}")
        sys.exit(1)
    data = resp.json()
    agent_id = data.get("agent_id") or data.get("id")
    if not agent_id:
        print(f"✗ Created agent but no ID in response: {data}")
        sys.exit(1)
    return agent_id


def update_agent_tools(agent_id: str, tool_ids: list[str]) -> None:
    """Update an existing agent to use the given tool IDs."""
    payload = {
        "conversation_config": {
            "agent": {
                "prompt": {
                    "tool_ids": tool_ids,
                },
            },
        },
    }
    resp = request("PATCH", f"/convai/agents/{agent_id}", json_body=payload)
    if resp.status_code not in (200, 204):
        print(f"  ✗ Failed to update agent {agent_id}: {resp.status_code} {resp.text}")
        sys.exit(1)


def append_to_env(key: str, value: str) -> None:
    """Append a line to .env (or warn if the user needs to add it manually)."""
    line = f"{key}={value}\n"
    try:
        with open(ENV_FILE, "a", encoding="utf-8") as f:
            f.write(line)
        print(f"  ✓ Wrote {key} to .env")
    except Exception as e:
        print(f"  ⚠️  Could not write to .env ({e}). Add this line manually:")
        print(f"     {line.strip()}")


# --- Main ---------------------------------------------------------------- #

def main() -> None:
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Tools dir:   {TOOLS_DIR}")
    print()

    # 1. Load tool specs.
    print("Loading tool specs...")
    specs = load_tool_specs()
    print(f"  Found {len(specs)} tool spec(s): {[s['name'] for s in specs]}")
    print()

    # 2. List existing tools, create missing.
    print("Checking existing tools on ElevenLabs...")
    existing = list_existing_tools()
    print(f"  Found {len(existing)} existing tool(s) on account.")

    tool_ids: list[str] = []
    failed_tools: list[str] = []
    for spec in specs:
        name = spec["name"]
        if name in existing:
            print(f"  • {name}: already exists ({existing[name]}), skipping create")
            tool_ids.append(existing[name])
        else:
            print(f"  • {name}: creating...")
            tid = create_tool(spec)
            if tid is not None:
                print(f"    ✓ created with id {tid}")
                tool_ids.append(tid)
            else:
                failed_tools.append(name)
    if failed_tools:
        print(f"  ⚠️  Could not create {len(failed_tools)} tool(s): {failed_tools}")
        print(f"  ⚠️  These tools must be created manually in the ElevenLabs dashboard.")
        print(f"         See setup/README.md for instructions.")
        print(f"  ⚠️  Re-run this script after manual creation to attach them to your agent.")
    if not tool_ids:
        print("  ⚠️  No tools available; agent will be configured without any tools.")
    print()

    # 3. Load baseline system prompt.
    from agent.system_prompt import get_baseline_prompt
    baseline = get_baseline_prompt()

    # 4. Create or update agent.
    print("Configuring agent...")
    if ELEVENLABS_AGENT_ID:
        agent = get_agent(ELEVENLABS_AGENT_ID)
        if agent:
            print(f"  Found existing agent {ELEVENLABS_AGENT_ID}; attaching {len(tool_ids)} tool(s).")
            update_agent_tools(ELEVENLABS_AGENT_ID, tool_ids)
            print(f"  ✓ Agent updated.")
        else:
            print(f"  ELEVENLABS_AGENT_ID={ELEVENLABS_AGENT_ID} not found on account; creating a fresh one.")
            new_id = create_agent(tool_ids, baseline)
            print(f"  ✓ Created agent {new_id}")
            print(f"  Update your .env: ELEVENLABS_AGENT_ID={new_id}")
    else:
        print("  No ELEVENLABS_AGENT_ID in .env; creating a fresh agent.")
        new_id = create_agent(tool_ids, baseline)
        print(f"  ✓ Created agent {new_id}")
        append_to_env("ELEVENLABS_AGENT_ID", new_id)

    print()
    print("Setup complete. You can now run the refinement loop.")


if __name__ == "__main__":
    main()