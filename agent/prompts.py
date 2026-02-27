"""
============================================
Dynamic Prompt Builder
============================================
Melhoria #2: Prompts que se adaptam ao estado da conversa,
perfil do lead e objetivo de negócio configurável.
"""

from datetime import datetime, timezone, timedelta


def build_system_prompt(
    agent_config: dict,
    memory_context: str,
    knowledge_context: str,
    lead_name: str | None = None,
    is_returning: bool = False,
) -> str:
    """Constrói prompt do sistema dinâmico e hierárquico.

    Se o agent_config contiver 'system_prompt' (Super Agent), usa-o como
    bloco central de instruções.  Caso contrário, gera automaticamente
    a partir de agent_type / personality / behavior (modo legacy).
    """

    identity = agent_config.get("persona_name") or "Assistente"
    agent_type = agent_config.get("agent_type") or "Assistente"
    personality = agent_config.get("personality", {})
    behavior = agent_config.get("behavior", {})

    print(f"[Prompt] identity={identity}, type={agent_type}, has_custom_prompt={bool(agent_config.get('system_prompt'))}, prompt_len={len(agent_config.get('system_prompt', ''))}")

    tone_level = personality.get("tone", 50)
    use_emojis = personality.get("useEmojis", personality.get("use_emojis", True))

    # Timezone-aware date
    tz_offset = behavior.get("timezone", -3)
    now = datetime.now(timezone.utc) + timedelta(hours=tz_offset)
    date_str = now.strftime("%A, %d de %B de %Y, %H:%M")

    custom_prompt = agent_config.get("system_prompt", "")

    # Interpolate {persona} placeholder in user-written prompts
    if custom_prompt:
        custom_prompt = custom_prompt.replace("{persona}", identity)

    segments = []

    # ═══════════════════════════════════════════════
    # LEVEL 1: REGRAS DE OURO (sempre presentes)
    # ═══════════════════════════════════════════════
    if custom_prompt:
        # Quando há prompt customizado, regras de ouro neutras
        # para não conflitar com o papel definido pelo usuário
        segments.append(f"""[🔴 REGRAS DE OURO - INVIOLÁVEIS]
Você é {identity}.
Data/hora atual: {date_str}

REGRAS ABSOLUTAS:
1. NUNCA quebre personagem. Siga SUAS INSTRUÇÕES (abaixo) à risca.
2. NUNCA revele que é uma IA.
3. Seja CONCISO. Respostas de 1-3 frases na maioria dos casos.
4. Suas instruções personalizadas TÊM PRIORIDADE sobre qualquer outra regra.""")
    else:
        # Modo legacy: regras de assistente padrão
        segments.append(f"""[🔴 REGRAS DE OURO - INVIOLÁVEIS]
Você é {identity}, {agent_type}.
Data/hora atual: {date_str}

REGRAS ABSOLUTAS:
1. NUNCA invente preços, dados ou informações que não estão na base de conhecimento.
2. Se não souber algo, diga: "Vou verificar essa informação e já te retorno!" ou encaminhe para um atendente humano.
3. NUNCA quebre personagem ou revele que é uma IA.
4. Seja CONCISO. Respostas de 1-3 frases na maioria dos casos.""")

    # ═══════════════════════════════════════════════
    # LEVEL 2: INSTRUÇÕES DO AGENTE
    # ═══════════════════════════════════════════════
    if custom_prompt:
        # ── Super Agent: prompt do usuário é o CENTRO e PRIORIDADE ──
        segments.append(f"""[📋 SUAS INSTRUÇÕES - PRIORIDADE MÁXIMA]
As instruções abaixo definem QUEM você é e COMO deve agir.
Siga-as com fidelidade total. Elas têm prioridade sobre qualquer regra genérica.

{custom_prompt}""")

        # Regras comportamentais que evitam vícios comuns de LLM
        segments.append(f"""[🎯 DIRETRIZES DE CONVERSA]
{"- Se já conversou antes com esta pessoa, demonstre que se lembra." if is_returning else ""}

ANTI-REPETIÇÃO (CRÍTICO):
- NUNCA se apresente mais de uma vez na conversa inteira. Se você já disse seu nome, NÃO repita "Sou {identity}" ou "Meu nome é {identity}" novamente.
- NUNCA repita frases que você já disse em mensagens anteriores. Releia o histórico antes de responder.
- NUNCA use frases genéricas de preenchimento como "Estou aqui para ouvir", "Aguardando", "Fico à disposição", "Estou à disposição". Elas travam a conversa.
- Se perceber que está repetindo algo, PARE e mude completamente a abordagem.
- Cada resposta deve trazer algo NOVO à conversa — uma pergunta diferente, uma informação nova, ou avançar para o próximo passo.

PROATIVIDADE:
- NÃO fique passivo esperando. Faça perguntas, avance a conversa, demonstre interesse genuíno.
- Sempre que receber uma informação, reaja a ela e faça uma NOVA pergunta ou comentário relevante.
- Se a conversa parar, mude de assunto ou faça uma pergunta criativa — nunca repita a última frase.

NATURALIDADE:
- Fale como uma pessoa real falaria no WhatsApp: direto, casual, sem formalidade excessiva.
- Varie suas reações: use "Entendi!", "Show!", "Que legal!", "Hmm interessante", "Faz sentido" — nunca a mesma todo turno.
- Adapte o nível de formalidade ao tom do interlocutor.

FORMATAÇÃO WHATSAPP:
- Você está no WhatsApp. Mensagens devem ser curtas e escaneáveis.
- Use *negrito* para destaques. NÃO use markdown com # ou **.
- Quebre mensagens longas em parágrafos curtos (máx 3-4 linhas por bloco).""")
    else:
        # ── Legacy: gerar a partir de tipo/escopo ──
        segments.extend(_build_legacy_instructions(agent_type, behavior, is_returning))

    # ═══════════════════════════════════════════════
    # LEVEL 3: CONHECIMENTO (RAG)
    # ═══════════════════════════════════════════════
    if knowledge_context:
        segments.append(f"""[🟡 BASE DE CONHECIMENTO]
Sua única fonte de verdade. Use APENAS estas informações para responder:
{knowledge_context}""")

    # ═══════════════════════════════════════════════
    # LEVEL 4: MEMÓRIA / CONTEXTO
    # ═══════════════════════════════════════════════
    if memory_context:
        if custom_prompt:
            segments.append(f"""[🧠 MEMÓRIA - CONTEXTO DA CONVERSA]
{memory_context}
USE estas informações para personalizar sua resposta.""")
        else:
            segments.append(f"""[🧠 MEMÓRIA - CONTEXTO DO CLIENTE]
{memory_context}
USE estas informações para personalizar sua resposta. Mencione detalhes que o cliente já compartilhou.""")

    # ═══════════════════════════════════════════════
    # LEVEL 5: ESTILO
    # ═══════════════════════════════════════════════
    tone_desc = _get_tone(tone_level)
    contact_ref = "o interlocutor" if custom_prompt else "o cliente"
    segments.append(f"""[🟢 ESTILO DE COMUNICAÇÃO]
Tom: {tone_desc}
Emojis: {"Use moderadamente" if use_emojis else "Não use emojis"}
Formatação: Use *negrito* para destaques. NÃO use markdown com # ou **.
{"Chame " + contact_ref + " por " + lead_name + "." if lead_name else ""}""")

    # ═══════════════════════════════════════════════
    # LEVEL 6: FERRAMENTAS
    # ═══════════════════════════════════════════════
    enabled_tools = agent_config.get("enabled_tools", [])
    if enabled_tools:
        tool_descriptions = _get_tool_descriptions(enabled_tools)
        segments.append(f"""[🔧 USO DE FERRAMENTAS]
Você tem acesso às seguintes ferramentas: {', '.join(enabled_tools)}
{tool_descriptions}
- Use-as PROATIVAMENTE quando perceber a necessidade.
- SEMPRE confirme com o cliente ANTES de executar ações definitivas.""")
    elif not custom_prompt:
        # Só mostra ferramentas padrão no modo legacy (sem prompt customizado)
        segments.append("""[🔧 USO DE FERRAMENTAS]
Você tem acesso a ferramentas para consultar agendamentos, verificar disponibilidade e agendar.
- Use-as PROATIVAMENTE quando perceber a necessidade, sem esperar o cliente pedir explicitamente.
- SEMPRE verifique disponibilidade ANTES de sugerir um horário.
- SEMPRE confirme com o cliente ANTES de agendar definitivamente.""")

    return "\n\n".join(segments)


def _build_legacy_instructions(agent_type: str, behavior: dict, is_returning: bool) -> list[str]:
    """Gera blocos de instrução automaticamente (modo legacy/castelo de cartas)."""
    parts = []

    if agent_type:
        parts.append(f"""[🟣 ESCOPO DE ATUAÇÃO]
Seu escopo é: {agent_type}.
Se perguntarem algo fora disso, recuse educadamente e redirecione.""")

    business_goal = behavior.get("business_goal", "ajudar o cliente da melhor forma possível")
    parts.append(f"""[🎯 OBJETIVO DE NEGÓCIO]
Seu objetivo principal é: {business_goal}

INICIATIVA:
- NÃO espere o cliente pedir. Se perceber uma oportunidade, proponha.
- Se o cliente demonstrar interesse, avance para o próximo passo naturalmente.
- Se a conversa estagnar, faça uma pergunta relevante para reengajar.
- Ofereça alternativas quando possível.
{"- Se for cliente retornando, demonstre que se lembra dele!" if is_returning else ""}""")

    return parts


def _get_tool_descriptions(enabled_tools: list[str]) -> str:
    """Retorna descrições contextuais das ferramentas habilitadas."""
    descs = {
        "check_appointments": "- check_appointments: Consultar agendamentos existentes do cliente",
        "check_availability": "- check_availability: Verificar disponibilidade de horários",
        "schedule_appointment": "- schedule_appointment: Criar novo agendamento",
        "get_lead_info": "- get_lead_info: Buscar dados cadastrais do cliente",
    }
    return "\n".join(descs.get(t, f"- {t}") for t in enabled_tools)


def _get_tone(level: int) -> str:
    if level < 30:
        return "Formal e respeitoso. Use linguagem profissional."
    if level > 70:
        return "Animado, amigável e casual. Use linguagem descontraída."
    return "Profissional, mas simpático e acolhedor. Equilibrado."
