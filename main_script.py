import os
import re
import sys
import json
import yaml
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from ddgs import DDGS
from json_repair import repair_json

# Ensure UTF-8 output on Windows consoles to prevent UnicodeEncodeError
if sys.platform == "win32":
    try:
        if sys.stdout and hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if sys.stderr and hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_ROUTER = os.getenv("MODEL_REVIEWER", "llama3.1:8b")
MODEL_WORKER = os.getenv("MODEL_WORKER", os.getenv("MODEL_REVIEWER", "llama3.1:8b"))

# Configure LLM Connections
llm_architect = LLM(model=f"ollama/{MODEL_ROUTER}", base_url=OLLAMA_BASE_URL)
llm_execution = LLM(model=f"ollama/{MODEL_WORKER}", base_url=OLLAMA_BASE_URL)

# Ensure required local directories exist
os.makedirs("config", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ------------------------------------------------------------------
# GENERALIST HELPERS: PATH EXTRACTION & CONTENT REPAIR
# ------------------------------------------------------------------
def extract_target_file_path(objective: str, default_filename: str = "resultado") -> str:
    """Intelligently detects or infers the target file path in 'output/' from any user prompt."""
    # Look for explicit output/filename.ext
    m = re.search(r"['\"]?(output/[a-zA-Z0-9_\-\.]+\.[a-zA-Z0-9]+)['\"]?", objective, re.IGNORECASE)
    if m:
        return m.group(1).replace("\\", "/")

    # Look for any filename.ext mentioned
    m = re.search(r"['\"]?([a-zA-Z0-9_\-]+\.(?:json|md|txt|csv|py|html|yaml|yml))['\"]?", objective, re.IGNORECASE)
    if m:
        return f"output/{m.group(1)}"

    # Infer by file format keywords in the prompt
    lower = objective.lower()
    if "json" in lower:
        return f"output/{default_filename}.json"
    elif any(k in lower for k in ["markdown", ".md", "relatorio", "relatório", "report", "artigo"]):
        return f"output/{default_filename}.md"
    elif "csv" in lower:
        return f"output/{default_filename}.csv"
    elif "html" in lower:
        return f"output/{default_filename}.html"
    elif any(k in lower for k in ["python", ".py", "script"]):
        return f"output/{default_filename}.py"
    else:
        return f"output/{default_filename}.txt"

def clean_and_repair_json(raw_text: str):
    """Generalist cleaner for JSON content: removes markdown wrappers, tool call artifacts, and repairs syntax."""
    if not raw_text or not isinstance(raw_text, str):
        return None

    # 1. Strip markdown code block wrappers if present
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text, flags=re.IGNORECASE)
    candidate = m.group(1).strip() if m else raw_text.strip()

    # 2. Strip leading hallucinated tool calls like {"name": "web_search_tool", ...}
    candidate = re.sub(r'^\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^\n]*\}\s*\}?', '', candidate).strip()

    # 3. Direct JSON parsing
    try:
        obj = json.loads(candidate)
        if isinstance(obj, list):
            data_items = [item for item in obj if isinstance(item, dict) and item.get("type") != "function" and "name" not in item]
            if len(data_items) == 1:
                return data_items[0]
            elif data_items:
                return data_items
        return obj
    except Exception:
        pass

    # 4. Use json_repair library on candidate
    try:
        obj = repair_json(candidate, return_objects=True)
        if isinstance(obj, list):
            data_items = [item for item in obj if isinstance(item, dict) and item.get("type") != "function" and "name" not in item]
            if len(data_items) == 1:
                return data_items[0]
            elif data_items:
                return data_items
        if isinstance(obj, (dict, list)) and obj:
            return obj
    except Exception:
        pass

    return None

# ------------------------------------------------------------------
# TOOL RELIABILITY TRACKING & ANTI-HALLUCINATION GUARDRAILS
# ------------------------------------------------------------------
# Some backends are blocked by TLS-inspecting proxies on certain networks; trying several
# in sequence lets the tool keep working even when the first choices are unreachable.
SEARCH_BACKENDS = ["google", "bing", "duckduckgo", "brave", "yahoo", "mojeek"]

TOOL_STATS = {"search_success": 0, "search_fail": 0, "scrape_success": 0, "scrape_fail": 0}

def reset_tool_stats():
    """Resets tool call counters at the start of each objective run."""
    for key in TOOL_STATS:
        TOOL_STATS[key] = 0

FABRICATION_MARKERS = (
    "internal knowledge", "as of my last update", "as of my knowledge cutoff",
    "i do not have real-time", "i don't have real-time", "i don't have access to real-time",
    "hypothetical", "simulated data", "for demonstration purposes", "illustrative purposes",
    "example data", "placeholder data",
)

def contains_fabrication_markers(text: str) -> bool:
    """Heuristically detects language indicating the model fabricated data instead of using tool results."""
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in FABRICATION_MARKERS)

INTEGRITY_CLAUSE = (
    "\n\nINTEGRITY RULE: Only report information actually returned by your tools. If a tool fails or "
    "returns no data, do NOT invent, guess, or rely on pre-trained/internal knowledge as if it were live data. "
    "In that case, explicitly state in your output that the data could not be retrieved and why."
)

# ------------------------------------------------------------------
# AGENT TOOLS & SKILLS DEFINITION
# ------------------------------------------------------------------
@tool("Web Search Tool")
def web_search(query: str) -> str:
    """Searches up-to-date information on the web, trying several search engine backends in sequence.
    Receives a search query string and returns top results with title, URL, and snippet summary.
    """
    if isinstance(query, dict):
        query = query.get("query", query.get("description", str(query)))
    query = str(query).strip()

    last_error = ""
    for backend in SEARCH_BACKENDS:
        try:
            results = []
            with DDGS(timeout=10) as ddgs:
                for r in ddgs.text(query, max_results=5, backend=backend):
                    results.append(f"Title: {r.get('title', '')}\nURL: {r.get('href', '')}\nSnippet: {r.get('body', '')}\n")
            if results:
                TOOL_STATS["search_success"] += 1
                return "\n".join(results)
        except Exception as e:
            last_error = str(e)
            continue

    TOOL_STATS["search_fail"] += 1
    return (
        f"Web search failed on all backends ({', '.join(SEARCH_BACKENDS)}). Last error: {last_error[:200]}. "
        "Do NOT invent or guess data to compensate. If no data can be retrieved, state clearly in your output "
        "that live data was unavailable and explain why."
    )

@tool("Web Scraping Tool")
def web_scraping(url: str) -> str:
    """Accesses a specific web page URL and extracts clean textual content."""
    try:
        if isinstance(url, dict):
            url = url.get("url", str(url))
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(str(url).strip(), headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.extract()
            
        clean_text = soup.get_text(separator=' ', strip=True)
        TOOL_STATS["scrape_success"] += 1
        return clean_text[:4000]
    except Exception as e:
        TOOL_STATS["scrape_fail"] += 1
        return (
            f"Scraping notice for URL {url}: {str(e)}. Do NOT invent data to replace this page's content; "
            "try a different URL or state that this source was unavailable."
        )

@tool("Save Local File Tool")
def save_local_file(filepath: str, content: str) -> str:
    """Saves text content (JSON, Markdown, TXT, CSV, Python, HTML, etc.) to a local file inside the 'output' directory.
    Parameters:
      filepath: target file path or filename (e.g. 'output/relatorio.md', 'output/dados.json')
      content: the complete textual content to write
    """
    try:
        if isinstance(filepath, dict):
            content = filepath.get("content", content)
            filepath = filepath.get("filepath", filepath.get("path", "output/resultado.txt"))

        filepath_str = str(filepath).replace("\\", "/").strip("'\" ")
        if not filepath_str.startswith("output/") and not os.path.isabs(filepath_str):
            if filepath_str.startswith("data/"):
                filepath_str = filepath_str.replace("data/", "output/", 1)
            else:
                filepath_str = f"output/{os.path.basename(filepath_str)}"

        directory = os.path.dirname(filepath_str) or "output"
        os.makedirs(directory, exist_ok=True)

        final_content = str(content)
        if filepath_str.endswith(".json"):
            parsed = clean_and_repair_json(final_content)
            if parsed is not None:
                final_content = json.dumps(parsed, indent=2, ensure_ascii=False)
        else:
            # If wrapped in full markdown codeblock, unwrap for clean file content
            if final_content.startswith("```") and final_content.endswith("```"):
                lines = final_content.splitlines()
                if len(lines) >= 2:
                    final_content = "\n".join(lines[1:-1]).strip()

        with open(filepath_str, "w", encoding="utf-8") as f:
            f.write(final_content)

        print(f"💾 [save_local_file] File saved at: '{filepath_str}' ({len(final_content)} bytes)")
        return f"[SUCCESS] File saved locally at '{filepath_str}'."
    except Exception as e:
        return f"[ERROR] Failed saving file to '{filepath}': {str(e)}"

# Dynamic tools map available for binding inside generated YAML configs
TOOL_MAP = {
    "web_search": web_search,
    "web_scraping": web_scraping,
    "save_local_file": save_local_file
}

# ------------------------------------------------------------------
# HELPER: ENSURE DELIVERABLE IS SAVED IN OUTPUT DIRECTORY
# ------------------------------------------------------------------
def ensure_output_file_saved(result_text: str, target_path: str = None) -> str:
    """Guarantees that the deliverable produced by the crew is saved on disk in the output directory."""
    os.makedirs("output", exist_ok=True)

    if not target_path:
        target_path = "output/resultado.txt"

    target_path = target_path.replace("\\", "/").strip("'\" ")
    if not target_path.startswith("output/") and not os.path.isabs(target_path):
        target_path = f"output/{os.path.basename(target_path)}"

    # 1. If file already exists and is non-empty, format JSON if applicable
    if os.path.exists(target_path) and os.path.getsize(target_path) > 0:
        if target_path.endswith(".json"):
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    disk_content = f.read()
                parsed = clean_and_repair_json(disk_content)
                if parsed is not None:
                    with open(target_path, "w", encoding="utf-8") as f:
                        json.dump(parsed, f, indent=2, ensure_ascii=False)
            except Exception:
                pass
        print(f"📁 [Auto-Save] Verified deliverable on disk at: '{target_path}'")
        return target_path

    # 2. Check if model embedded a save_local_file tool call in result_text
    tool_arg_match = re.search(r'\{"name":\s*"save_local_file[^"]*",\s*"arguments":\s*(\{.*?\})\s*\}', result_text, re.DOTALL)
    if tool_arg_match:
        try:
            args = json.loads(tool_arg_match.group(1))
            c = args.get("content", "")
            p = args.get("filepath", target_path)
            if c:
                save_local_file.run(filepath=p, content=c if isinstance(c, str) else json.dumps(c))
                if os.path.exists(p) and os.path.getsize(p) > 0:
                    return p
        except Exception:
            pass

    # 3. If target is JSON, extract and format JSON from result_text
    if target_path.endswith(".json"):
        parsed = clean_and_repair_json(result_text)
        if parsed is not None:
            with open(target_path, "w", encoding="utf-8") as f:
                json.dump(parsed, f, indent=2, ensure_ascii=False)
            print(f"📁 [Auto-Save] Extracted and saved JSON deliverable to: '{target_path}'")
            return target_path

    # 4. For Markdown, TXT, CSV, or other formats
    clean_text = result_text.strip()
    clean_text = re.sub(r'^\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^\n]*\}\s*\}?\s*', '', clean_text).strip()
    if clean_text.startswith("```") and clean_text.endswith("```"):
        lines = clean_text.splitlines()
        if len(lines) >= 2:
            clean_text = "\n".join(lines[1:-1]).strip()

    if clean_text:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(clean_text)
        print(f"📁 [Auto-Save] Saved deliverable to: '{target_path}'")

    return target_path

# ------------------------------------------------------------------
# PHASE 1: GENERATE YAMLS WITH SELF-CORRECTION LOOP (TSUBAS ARCHITECT)
# ------------------------------------------------------------------
def generate_team_configurations(user_prompt: str, max_attempts: int = 3):
    """Generates YAML configuration files for ANY user prompt using an agentic Self-Correction Loop."""
    print("🧠 Tsubas designing the agent team...")

    architect = Agent(
        role="Tsubas - The Samurai Architect",
        goal="Analyze any objective with serenity and forge the perfect multi-agent team in YAML.",
        backstory=(
            "You are Tsubas, a master samurai in systems architecture with unwavering discipline and composure. "
            "You analyze objectives with absolute precision and design efficient multi-agent pipelines with appropriate tools."
        ),
        llm=llm_architect,
        verbose=False
    )

    error_feedback = ""
    for attempt in range(1, max_attempts + 1):
        print(f"🔄 [Architectural Loop] Attempt {attempt}/{max_attempts}...")

        meta_prompt = f"""
Analyze the following objective and design a lean, highly effective multi-agent team in JSON to achieve it directly.

Objective: "{user_prompt}"

Available tools to assign to agents as needed:
- "web_search": Search live information on the web.
- "web_scraping": Extract cleaned textual content from web pages.
- "save_local_file": Write files (JSON, Markdown, TXT, CSV, etc.) into the 'output' directory.

RULES:
1. Design a LEAN team: Keep it to 1 or 2 focused agents and 1 or 2 clear sequential tasks maximum. Avoid creating unnecessary micro-tasks.
2. If the objective requires generating or saving a file (or report, data, code, JSON, etc.):
   - Assign the 'save_local_file' tool to the agent responsible for producing the deliverable.
   - Set "output_file": "output/<filename.ext>" on the task using an appropriate filename and extension based on the user's request.
   - In that task's description, explicitly instruct the agent to produce the complete deliverable and save it to 'output/<filename.ext>'.
3. If no file saving is requested, do not set "output_file".

{f"ATTENTION: Your previous attempt failed with error: {error_feedback}. Fix the JSON strictly!" if error_feedback else ""}

Respond with ONLY a single valid JSON object (no prose, no markdown fences, no headers/titles) in EXACTLY this shape:
{{
  "agents": {{
    "agent_1": {{
      "role": "...",
      "goal": "...",
      "backstory": "...",
      "tools": ["web_search", "save_local_file"]
    }}
  }},
  "tasks": {{
    "task_1": {{
      "description": "...",
      "expected_output": "...",
      "agent": "agent_1",
      "output_file": "output/<filename.ext>"
    }}
  }}
}}
"""

        design_task = Task(
            description=meta_prompt,
            expected_output="A single valid JSON object with 'agents' and 'tasks' keys, no prose or markdown fences",
            agent=architect
        )

        meta_crew = Crew(agents=[architect], tasks=[design_task], verbose=False)
        raw_response = str(meta_crew.kickoff())

        # JSON tolerates small-model formatting mistakes far better than indentation-sensitive YAML,
        # and we can reuse the existing repair_json-based recovery helper below.
        parsed = clean_and_repair_json(raw_response)

        try:
            if not isinstance(parsed, dict):
                raise ValueError("Response was not a JSON object.")

            agents_section = parsed.get("agents", parsed)
            tasks_section = parsed.get("tasks", parsed)
            all_items = {}
            if isinstance(agents_section, dict):
                all_items.update(agents_section)
            if isinstance(tasks_section, dict):
                all_items.update(tasks_section)
            if not all_items and isinstance(parsed, dict):
                all_items.update(parsed)

            agents_dict = {k: v for k, v in all_items.items() if isinstance(v, dict) and ("role" in v or "goal" in v)}
            tasks_dict = {k: v for k, v in all_items.items() if isinstance(v, dict) and ("description" in v or "expected_output" in v or "agent" in v)}

            if agents_dict and tasks_dict:
                with open("config/agents.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(agents_dict, f, allow_unicode=True, sort_keys=False)
                with open("config/tasks.yaml", "w", encoding="utf-8") as f:
                    yaml.dump(tasks_dict, f, allow_unicode=True, sort_keys=False)
                print("✅ Files 'config/agents.yaml' and 'config/tasks.yaml' validated and saved!")
                return
            else:
                error_feedback = "JSON did not contain distinct agent and task definitions."
        except Exception as e:
            error_feedback = f"JSON parsing error: {str(e)}"

        print(f"   \u26a0\ufe0f Attempt {attempt} rejected: {error_feedback}")

    raise RuntimeError(
        "\u274c Failed configuration generation loop after reaching maximum retry limit.\n"
        f"Last error: {error_feedback}\nLast raw response:\n{raw_response[:1500]}"
    )

# ------------------------------------------------------------------
# PHASE 2: DYNAMIC EXECUTION WITH REFINEMENT & VERIFIER LOOP
# ------------------------------------------------------------------
def execute_dynamic_team_with_loop(original_objective: str, max_quality_loops: int = 2):
    """Instantiates the dynamic crew and executes it within a Quality Verification & Feedback Loop."""
    print("🚀 Instantiating and executing the dynamic work crew...")
    reset_tool_stats()

    with open("config/agents.yaml", "r", encoding="utf-8") as f:
        agents_data = yaml.safe_load(f) or {}

    with open("config/tasks.yaml", "r", encoding="utf-8") as f:
        tasks_data = yaml.safe_load(f) or {}

    # Determine default target output file from tasks or prompt
    target_output_file = None
    for task_id, specs in tasks_data.items():
        if isinstance(specs, dict) and specs.get("output_file"):
            target_output_file = str(specs["output_file"]).replace("\\", "/")
            break

    if not target_output_file:
        target_output_file = extract_target_file_path(original_objective)

    if not target_output_file.startswith("output/") and not os.path.isabs(target_output_file):
        target_output_file = f"output/{os.path.basename(target_output_file)}"

    os.makedirs(os.path.dirname(target_output_file) or "output", exist_ok=True)

    mapped_agents = {}
    agents_list = []
    tasks_list = []

    for agent_id, specs in agents_data.items():
        agent_tools = []
        if isinstance(specs, dict) and "tools" in specs and isinstance(specs["tools"], list):
            for tool_name in specs["tools"]:
                if tool_name in TOOL_MAP:
                    agent_tools.append(TOOL_MAP[tool_name])

        agent = Agent(
            role=specs.get("role") or "Execution Agent",
            goal=specs.get("goal") or "Perform assigned task",
            backstory=specs.get("backstory") or "Task execution specialist",
            tools=agent_tools,
            llm=llm_execution,
            max_iter=8,
            verbose=True
        )
        mapped_agents[agent_id] = agent
        agents_list.append(agent)

    task_items = list(tasks_data.items())
    for idx, (task_id, specs) in enumerate(task_items):
        if not isinstance(specs, dict):
            continue
        responsible_agent = mapped_agents.get(specs.get("agent"))

        # Determine task output file
        task_output = specs.get("output_file")
        if not task_output and idx == len(task_items) - 1 and target_output_file:
            task_output = target_output_file

        if task_output:
            task_output = str(task_output).replace("\\", "/")
            if not task_output.startswith("output/") and not os.path.isabs(task_output):
                task_output = f"output/{os.path.basename(task_output)}"
            os.makedirs(os.path.dirname(task_output) or "output", exist_ok=True)

        exp_out = specs.get("expected_output", "")
        if isinstance(exp_out, (list, dict)):
            exp_out = json.dumps(exp_out, ensure_ascii=False)

        task_kwargs = {
            "description": specs.get("description", "") + INTEGRITY_CLAUSE,
            "expected_output": str(exp_out),
            "agent": responsible_agent,
        }
        if task_output:
            task_kwargs["output_file"] = task_output

        task = Task(**task_kwargs)
        tasks_list.append(task)

    # Primary Crew Execution
    crew = Crew(
        agents=agents_list,
        tasks=tasks_list,
        process=Process.sequential,
        verbose=True
    )

    execution_result = str(crew.kickoff())

    # Guarantee deliverable is saved and formatted in output/
    ensure_output_file_saved(execution_result, target_output_file)

    # ------------------------------------------------------------------
    # AGENTIC LOOP: INSPECTOR & QUALITY VERIFIER
    # ------------------------------------------------------------------
    inspector_agent = Agent(
        role="Quality and Compliance Inspector",
        goal="Validate whether the final deliverable strictly meets all requirements of the goal.",
        backstory="You are a meticulous auditor responsible for ensuring accuracy, completeness, and tool execution integrity.",
        llm=llm_architect,
        max_iter=5,
        verbose=True
    )

    for cycle in range(1, max_quality_loops + 1):
        print(f"\n🔍 [Quality & Verification Loop] Cycle {cycle}/{max_quality_loops}...")

        # List files currently present in output/
        saved_files = []
        if os.path.exists("output"):
            for fname in os.listdir("output"):
                fpath = os.path.join("output", fname)
                if os.path.isfile(fpath):
                    saved_files.append(f"- {fname} ({os.path.getsize(fpath)} bytes)")
        files_summary = "\n".join(saved_files) if saved_files else "None"

        # Deterministic guardrail: catch fabricated/hallucinated data, fully failed tool usage,
        # or a malformed deliverable, before trusting a (less reliable) local LLM's self-reported judgment.
        stats_snapshot = dict(TOOL_STATS)
        total_tool_calls = sum(stats_snapshot.values())
        zero_successful_tools = (stats_snapshot["search_success"] + stats_snapshot["scrape_success"]) == 0
        any_tool_failures = (stats_snapshot["search_fail"] + stats_snapshot["scrape_fail"]) > 0
        fabricated = contains_fabrication_markers(execution_result)

        invalid_json_file = False
        if target_output_file.endswith(".json") and os.path.exists(target_output_file):
            try:
                with open(target_output_file, "r", encoding="utf-8") as f:
                    json.load(f)
            except Exception:
                invalid_json_file = True

        if fabricated or (total_tool_calls > 0 and zero_successful_tools and any_tool_failures):
            verdict = (
                "STATUS: REJECTED\n"
                "REASON: Deterministic guardrail triggered - the output contains language indicating "
                f"fabricated/non-live data (detected={fabricated}), or no tool call succeeded "
                f"(stats={stats_snapshot}).\n"
                "REWORK_INSTRUCTION: Retry using the available tools with different queries or sources. "
                "If tools keep failing, explicitly disclose in the deliverable that live data was unavailable "
                "instead of presenting invented figures as real."
            )
            print(f"🛑 [Guardrail] Auto-rejected before LLM audit. Tool stats: {stats_snapshot}")
        elif invalid_json_file:
            verdict = (
                "STATUS: REJECTED\n"
                f"REASON: Deterministic guardrail triggered - '{target_output_file}' does not contain valid JSON "
                "even though a .json deliverable was required.\n"
                f"REWORK_INSTRUCTION: Use the 'save_local_file' tool to write a single valid JSON object/array "
                f"(no prose, no markdown fences) to '{target_output_file}'."
            )
            print(f"🛑 [Guardrail] Auto-rejected before LLM audit: '{target_output_file}' is not valid JSON.")
        else:
            audit_description = f"""
Original Objective: "{original_objective}"

Delivered Files in 'output/' Directory:
{files_summary}

Target Output File:
'{target_output_file}'

Tool Call Statistics So Far: {stats_snapshot}

Current Execution Result:
"{execution_result[:2500]}"

Verify if:
1. The objective requirements were fulfilled.
2. If file generation was requested, verify that the expected file exists in 'output/' and has meaningful content.
3. The content is well-structured and free of major defects or placeholder errors.
4. The data appears to genuinely come from tool calls (not invented). If the tool statistics show zero 
successful search/scrape calls yet the result presents specific real-world figures as fact, REJECT it.

Respond strictly in the format:
STATUS: [APPROVED or REJECTED]
REASON: [Brief explanation]
REWORK_INSTRUCTION: [Clear instructions for adjustment if REJECTED]
"""
            audit_task = Task(
                description=audit_description,
                expected_output="Formatted strictly with STATUS, REASON, and REWORK_INSTRUCTION",
                agent=inspector_agent
            )

            audit_crew = Crew(agents=[inspector_agent], tasks=[audit_task], verbose=False)
            verdict = str(audit_crew.kickoff())

        if re.search(r"STATUS:\s*APPROVED", verdict, re.IGNORECASE):
            print("✅ Output APPROVED by Quality Inspector!")
            break
        else:
            print(f"⚠️ Rework Requested by Inspector:\n{verdict}")
            feedback_loop = verdict

            # Rework task executed by operational workers
            correction_task = Task(
                description=(
                    f"Address all issues flagged by the inspector:\n{feedback_loop}\n"
                    f"Original Goal: {original_objective}\n"
                    f"Tool call statistics so far: {stats_snapshot}\n"
                    "If searches/scrapes keep failing, try substantially different queries or URLs. "
                    f"Save the final deliverable using 'save_local_file' into '{target_output_file}'."
                    + INTEGRITY_CLAUSE
                ),
                expected_output=f"Corrected result saved into '{target_output_file}'.",
                agent=agents_list[-1],
                output_file=target_output_file
            )
            correction_crew = Crew(agents=agents_list, tasks=[correction_task], verbose=True)
            execution_result = str(correction_crew.kickoff())
            ensure_output_file_saved(execution_result, target_output_file)

    # Return final file content if available, else execution result
    if os.path.exists(target_output_file):
        try:
            with open(target_output_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            pass

    return execution_result

# ------------------------------------------------------------------
# FULL WORKFLOW EXECUTION
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Generalist objective: can be changed to ANY request (research, reports, code, data, JSON, etc.)
    desired_objective = (
        "Pesquise o preco do playstation 5 no ano de 2026 e monte um historio temporal entre janeiro e agosto."
    )

    # Allow custom prompt via command line argument: python main_script.py "seu prompt aqui"
    if len(sys.argv) > 1:
        desired_objective = " ".join(sys.argv[1:])

    print(f"🎯 Objective: {desired_objective}\n")

    # 1. Tsubas generates and validates team structure using the Self-Correction Loop
    generate_team_configurations(desired_objective)

    # 2. Execute dynamic team under the Verifier Loop
    final_output = execute_dynamic_team_with_loop(desired_objective)

    print("\n================ FINAL RESULT ================\n")
    print(final_output)

    # List all output files created
    print("\n📂 Files saved in 'output/' directory:")
    if os.path.exists("output"):
        for f in os.listdir("output"):
            fp = os.path.join("output", f)
            if os.path.isfile(fp):
                print(f"   - output/{f} ({os.path.getsize(fp)} bytes)")