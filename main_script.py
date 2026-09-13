import os
import re
import yaml
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from duckduckgo_search import DDGS

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_ROUTER = os.getenv("MODEL_REVIEWER", "llama3.1:8b")
MODEL_WORKER = os.getenv("MODEL_CODER", "qwen2.5-coder:7b")

# Conexões LLM
llm_arquitetura = LLM(model=f"ollama/{MODEL_ROUTER}", base_url=OLLAMA_BASE_URL)
llm_execucao = LLM(model=f"ollama/{MODEL_WORKER}", base_url=OLLAMA_BASE_URL)

os.makedirs("config", exist_ok=True)

# ------------------------------------------------------------------
# DEFINIÇÃO DE FERRAMENTAS (TOOLS) DE PESQUISA E SCRAPING
# ------------------------------------------------------------------
@tool("Pesquisa na Internet")
def pesquisa_web(query: str) -> str:
    """Pesquisa informações atualizadas na internet usando o DuckDuckGo.
    Recebe um termo de busca e devolve os principais resultados com título, link e resumo.
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(f"Título: {r['title']}\nURL: {r['href']}\nResumo: {r['body']}\n")
        return "\n".join(results) if results else "Nenhum resultado encontrado."
    except Exception as e:
        return f"Erro ao realizar busca na web: {str(e)}"

@tool("Raspagem Web (Web Scraping)")
def web_scraping(url: str) -> str:
    """Acessa uma URL específica da internet e extrai o conteúdo de texto limpo da página."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove tags não textuais
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.extract()
            
        texto_limpo = soup.get_text(separator=' ', strip=True)
        # Limita caracteres para evitar estourar o contexto da LLM
        return texto_limpo[:4000]
    except Exception as e:
        return f"Erro ao realizar scraping na URL {url}: {str(e)}"

# Mapa de ferramentas para vincular aos agentes dinâmicos
MAPA_FERRAMENTAS = {
    "pesquisa_web": pesquisa_web,
    "web_scraping": web_scraping
}

# ------------------------------------------------------------------
# FASE 1: GERAR OS YAMLs AUTOMATICAMENTE (TSUBAS)
# ------------------------------------------------------------------
def gerar_configuracoes_equipe(prompt_usuario: str):
    print("🧠 Tsubas analisando a solicitação e projetando a equipe...")

    arquiteto = Agent(
        role="Tsubas - O Arquiteto Samurai",
        goal="Analisar a missão com serenidade e forjar a equipe de agentes perfeita em YAML.",
        backstory=(
            "Você é Tsubas, um samurai mestre em arquitetura de sistemas com alto autocontrole e disciplina inabalável. "
            "Você não age por impulso; analisa a missão com foco absoluto e desembainha a lógica precisa para projetar "
            "papéis e tarefas impecáveis em YAML."
        ),
        llm=llm_arquitetura,
        verbose=False
    )

    prompt_metaprogramacao = f"""
Com base no objetivo: "{prompt_usuario}"

Você tem à disposição as seguintes ferramentas para atribuir aos agentes se necessário:
- "pesquisa_web" (Para buscar informações atualizadas na internet)
- "web_scraping" (Para extrair conteúdo de páginas da web)

Gere EXATAMENTE duas estruturas YAML válidas separadas por '---'.

A primeira estrutura representa os 'agentes' (formato dict com chaves simples). Se o agente precisar de ferramentas, inclua a lista 'tools':
agente_1:
  role: "..."
  goal: "..."
  backstory: "..."
  tools: ["pesquisa_web", "web_scraping"] # Opcional, usar apenas se necessário
agente_2:
  role: "..."
  goal: "..."
  backstory: "..."

A segunda estrutura representa as 'tarefas' (associadas aos agentes acima):
tarefa_1:
  description: "..."
  expected_output: "..."
  agent: "agente_1"
tarefa_2:
  description: "..."
  expected_output: "..."
  agent: "agente_2"

Responda APENAS com o bloco YAML puro, sem explicações adicionais ou marcações markdown como ```yaml.
"""

    tarefa_design = Task(
        description=prompt_metaprogramacao,
        expected_output="Dois blocos YAML válidos separados por '---'",
        agent=arquiteto
    )

    meta_crew = Crew(agents=[arquiteto], tasks=[tarefa_design], verbose=False)
    resposta_yaml = str(meta_crew.kickoff())

    # Limpeza básica do texto retornado para garantir YAML válido
    resposta_limpa = re.sub(r'```yaml|```', '', resposta_yaml).strip()
    partes = resposta_limpa.split('---')

    if len(partes) < 2:
        raise ValueError("O modelo não separou a configuração dos agentes e tarefas corretamente.")

    # Salva os arquivos nas pastas locais
    with open("config/agents.yaml", "w", encoding="utf-8") as f:
        f.write(partes[0].strip())

    with open("config/tasks.yaml", "w", encoding="utf-8") as f:
        f.write(partes[1].strip())

    print("✅ Arquivos config/agents.yaml e config/tasks.yaml gerados com sucesso por Tsubas!")

# ------------------------------------------------------------------
# FASE 2: CARREGAR YAMLs E EXECUTAR A EQUIPE DINÂMICA
# ------------------------------------------------------------------
def executar_equipe_dinamica():
    print("🚀 Instanciando e executando a equipe gerada dinamicamente...")

    with open("config/agents.yaml", "r", encoding="utf-8") as f:
        agents_data = yaml.safe_load(f)

    with open("config/tasks.yaml", "r", encoding="utf-8") as f:
        tasks_data = yaml.safe_load(f)

    agentes_mapeados = {}
    lista_agentes = []
    lista_tarefas = []

    # Instancia cada agente declarado no YAML gerado
    for id_agente, specs in agents_data.items():
        # Identifica e associa as ferramentas caso especificadas no YAML
        ferramentas_agente = []
        if "tools" in specs and isinstance(specs["tools"], list):
            for nome_tool in specs["tools"]:
                if nome_tool in MAPA_FERRAMENTAS:
                    ferramentas_agente.append(MAPA_FERRAMENTAS[nome_tool])

        agente = Agent(
            role=specs["role"],
            goal=specs["goal"],
            backstory=specs["backstory"],
            tools=ferramentas_agente,
            llm=llm_execucao,
            verbose=True
        )
        agentes_mapeados[id_agente] = agente
        lista_agentes.append(agente)

    # Instancia cada tarefa e vincula ao seu respectivo agente
    for id_tarefa, specs in tasks_data.items():
        agente_responsavel = agentes_mapeados.get(specs["agent"])
        tarefa = Task(
            description=specs["description"],
            expected_output=specs["expected_output"],
            agent=agente_responsavel
        )
        lista_tarefas.append(tarefa)

    # Roda a equipe gerada em tempo de execução
    equipe = Crew(
        agents=lista_agentes,
        tasks=lista_tarefas,
        process=Process.sequential,
        verbose=True
    )

    return equipe.kickoff()

# ------------------------------------------------------------------
# EXECUÇÃO DO FLUXO COMPLETO
# ------------------------------------------------------------------
# DEFINITION OF SEARCH AND SCRAPING TOOLS
# ------------------------------------------------------------------
@tool("Web Search Tool")
def pesquisa_web(query: str) -> str:
    """Searches up-to-date information on the internet using DuckDuckGo.
    Receives a search query and returns top results with title, URL, and snippet.
    """
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}\n")
        return "\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Error executing web search: {str(e)}"

@tool("Web Scraping Tool")
def web_scraping(url: str) -> str:
    """Accesses a specific web page URL and extracts clean text content."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        # Remove non-textual tags
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.extract()
            
        texto_limpo = soup.get_text(separator=' ', strip=True)
        # Limit character count to avoid overflowing LLM context window
        return texto_limpo[:4000]
    except Exception as e:
        return f"Error scraping URL {url}: {str(e)}"

# Tool map to link dynamic agents
MAPA_FERRAMENTAS = {
    "pesquisa_web": pesquisa_web,
    "web_scraping": web_scraping
}

# ------------------------------------------------------------------
# PHASE 1: GENERATE YAML CONFIGURATIONS AUTOMATICALLY (TSUBAS)
# ------------------------------------------------------------------
def gerar_configuracoes_equipe(prompt_usuario: str):
    print("🧠 Tsubas analyzing the request and designing the team...")

    arquiteto = Agent(
        role="Tsubas - The Samurai Architect",
        goal="Analyze the mission with tranquility and forge the perfect agent team in YAML.",
        backstory=(
            "You are Tsubas, a master samurai in systems architecture with high self-control and unwavering discipline. "
            "You do not act on impulse; you analyze the mission with absolute focus and unsheathe precise logic to design "
            "impeccable roles and tasks in YAML."
        ),
        llm=llm_arquitetura,
        verbose=False
    )

    prompt_metaprogramacao = f"""
Based on the objective: "{prompt_usuario}"

You have the following tools available to assign to agents if necessary:
- "pesquisa_web" (To search for up-to-date information on the web)
- "web_scraping" (To extract clean content from web pages)

Generate EXACTLY two valid YAML structures separated by '---'. Do NOT include markdown code fences like ```yaml or conversational text.

Structure 1 represents 'agents' (dictionary format with simple keys). If an agent needs tools, include the 'tools' list:
agent_1:
  role: "..."
  goal: "..."
  backstory: "..."
  tools: ["pesquisa_web", "web_scraping"] # Optional, only if required
agent_2:
  role: "..."
  goal: "..."
  backstory: "..."

Structure 2 represents 'tasks' (associated with the agents above):
task_1:
  description: "..."
  expected_output: "..."
  agent: "agent_1"
task_2:
  description: "..."
  expected_output: "..."
  agent: "agent_2"
"""

    tarefa_design = Task(
        description=prompt_metaprogramacao,
        expected_output="Two valid YAML blocks separated by '---'",
        agent=arquiteto
    )

    meta_crew = Crew(agents=[arquiteto], tasks=[tarefa_design], verbose=False)
    resposta_yaml = str(meta_crew.kickoff())

    # Clean markdown formatting and split blocks safely
    resposta_limpa = re.sub(r'```(?:yaml)?', '', resposta_yaml).strip().rstrip('`')
    partes = [p.strip() for p in resposta_limpa.split('---') if p.strip()]

    if len(partes) < 2:
        raise ValueError(
            f"Tsubas failed to generate two distinct YAML blocks separated by '---'.\nRaw response:\n{resposta_yaml}"
        )

    # Validate YAML structure before writing to file
    try:
        parsed_agents = yaml.safe_load(partes[0])
        parsed_tasks = yaml.safe_load(partes[1])
        if not isinstance(parsed_agents, dict) or not isinstance(parsed_tasks, dict):
            raise ValueError("Generated YAML is not a valid dictionary.")
    except Exception as err:
        raise ValueError(f"Error parsing generated YAML: {err}\nRaw output:\n{resposta_yaml}")

    # Save configuration files
    with open("config/agents.yaml", "w", encoding="utf-8") as f:
        f.write(partes[0])

    with open("config/tasks.yaml", "w", encoding="utf-8") as f:
        f.write(partes[1])

    print("✅ Files config/agents.yaml and config/tasks.yaml successfully generated by Tsubas!")

# ------------------------------------------------------------------
# PHASE 2: LOAD YAMLS AND EXECUTE DYNAMIC TEAM
# ------------------------------------------------------------------
def executar_equipe_dinamica():
    print("🚀 Instantiating and executing dynamically generated team...")

    with open("config/agents.yaml", "r", encoding="utf-8") as f:
        agents_data = yaml.safe_load(f) or {}

    with open("config/tasks.yaml", "r", encoding="utf-8") as f:
        tasks_data = yaml.safe_load(f) or {}

    if not isinstance(agents_data, dict) or not agents_data:
        raise ValueError("config/agents.yaml is empty or invalid. Please re-run team generation.")

    if not isinstance(tasks_data, dict) or not tasks_data:
        raise ValueError("config/tasks.yaml is empty or invalid. Please re-run team generation.")

    agentes_mapeados = {}
    lista_agentes = []
    lista_tarefas = []

    # Instantiate each agent declared in generated YAML
    for id_agente, specs in agents_data.items():
        # Identify and associate tools if specified in YAML
        ferramentas_agente = []
        if "tools" in specs and isinstance(specs["tools"], list):
            for nome_tool in specs["tools"]:
                if nome_tool in MAPA_FERRAMENTAS:
                    ferramentas_agente.append(MAPA_FERRAMENTAS[nome_tool])

        agente = Agent(
            role=specs.get("role", "Agent"),
            goal=specs.get("goal", "Execute assigned task"),
            backstory=specs.get("backstory", "Experienced AI Agent"),
            tools=ferramentas_agente,
            llm=llm_execucao,
            verbose=True
        )
        agentes_mapeados[id_agente] = agente
        lista_agentes.append(agente)

    # Instantiate each task and bind to responsible agent
    for id_tarefa, specs in tasks_data.items():
        agente_responsavel = agentes_mapeados.get(specs.get("agent"))
        tarefa = Task(
            description=specs.get("description", ""),
            expected_output=specs.get("expected_output", ""),
            agent=agente_responsavel
        )
        lista_tarefas.append(tarefa)

    # Run generated crew sequentially
    equipe = Crew(
        agents=lista_agentes,
        tasks=lista_tarefas,
        process=Process.sequential,
        verbose=True
    )

    return equipe.kickoff()

# ------------------------------------------------------------------
# COMPLETE WORKFLOW EXECUTION
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Example objective requiring web search, scraping, and technical summary
    objetivo_desejado = (
    "Pesquise as cotações das principais moedas globais (USD, EUR, GBP, JPY e CAD) em relação ao Real (BRL) hoje. "
    "Obtenha o valor atual, a máxima, a mínima e a média do dia para cada uma delas, "
    "incluindo uma breve análise sobre as principais variações ocorridas no mercado. "
    "Formate todos os dados coletados e a análise em um arquivo JSON válido chamado 'cotacoes_hoje.json' e salve no disco."
)

    # 1. Tsubas creates procedural agents structure with proper tools
    gerar_configuracoes_equipe(objetivo_desejado)

    # 2. Execute generated team
    resultado = executar_equipe_dinamica()

    print("\n================ FINAL RESULT ================\n")
    print(resultado)