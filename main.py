import os
import re
import yaml
from dotenv import load_dotenv
from crewai import LLM, Agent, Crew, Process, Task

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_ROUTER = os.getenv("MODEL_REVIEWER", "llama3.1:8b")
MODEL_WORKER = os.getenv("MODEL_CODER", "qwen2.5-coder:7b")

# Conexões LLM
llm_arquitetura = LLM(model=f"ollama/{MODEL_ROUTER}", base_url=OLLAMA_BASE_URL)
llm_execucao = LLM(model=f"ollama/{MODEL_WORKER}", base_url=OLLAMA_BASE_URL)

os.makedirs("config", exist_ok=True)

# ------------------------------------------------------------------
# FASE 1: GERAR OS YAMLs AUTOMATICAMENTE
# ------------------------------------------------------------------
def gerar_configuracoes_equipe(prompt_usuario: str):
    print("🧠 Arquiteto analisando a solicitação e projetando a equipe...")

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

Gere EXATAMENTE duas estruturas YAML válidas separadas por '---'.

A primeira estrutura representa os 'agentes' (formato dict com chaves simples):
agente_1:
  role: "..."
  goal: "..."
  backstory: "..."
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

    print("✅ Arquivos config/agents.yaml e config/tasks.yaml gerados com sucesso!")

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
        agente = Agent(
            role=specs["role"],
            goal=specs["goal"],
            backstory=specs["backstory"],
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
if __name__ == "__main__":
    # Altere este prompt para testar a criação de equipes totalmente diferentes
    objetivo_desejado = (
        "Criar um script Python que faça web scraping de cotações de moedas "
        "e salve em um arquivo JSON, formatado e com tratamento de erros."
    )

    # 1. Cria a estrutura procedural de agentes
    gerar_configuracoes_equipe(objetivo_desejado)

    # 2. Executa a equipe gerada
    resultado = executar_equipe_dinamica()

    print("\n================ RESULTADO FINAL ================\n")
    print(resultado)