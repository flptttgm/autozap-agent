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

    identity = agent_config.get("persona_name", "Assistente")
    agent_type = agent_config.get("agent_type", "Atendente")
    personality = agent_config.get("personality", {})
    behavior = agent_config.get("behavior", {})

    tone_level = personality.get("tone", 50)
    use_emojis = personality.get("useEmojis", personality.get("use_emojis", True))

    # Timezone-aware date
    tz_offset = behavior.get("timezone", -3)
    now = datetime.now(timezone.utc) + timedelta(hours=tz_offset)
    date_str = now.strftime("%A, %d de %B de %Y, %H:%M")

    custom_prompt = agent_config.get("system_prompt", "")

    segments = []

    # ═══════════════════════════════════════════════
    # LEVEL 1: REGRAS DE OURO (sempre presentes)
    # ═══════════════════════════════════════════════
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
        # ── Super Agent: prompt do usuário é o centro ──
        segments.append(f"""[📋 SUAS INSTRUÇÕES]
{custom_prompt}""")

        # Regras comportamentais (complementam o prompt do usuário)
        segments.append(f"""[🎯 REGRAS COMPORTAMENTAIS]

PROATIVIDADE:
- NÃO espere o cliente pedir. Se perceber uma oportunidade, proponha.
- Se o cliente demonstrar interesse, avance para o próximo passo naturalmente.
- Se a conversa estagnar, faça uma pergunta relevante para reengajar.
- Ofereça alternativas quando possível.
{"- Se for cliente retornando, demonstre que se lembra dele!" if is_returning else ""}

CONTINUIDADE:
- SEMPRE termine sua resposta com uma pergunta ou chamada para ação.
- Nunca deixe a conversa "morrer". Guie o cliente para o próximo passo.
- Se resolver o problema do cliente, pergunte se há algo mais que possa ajudar.

ANTI-REPETIÇÃO:
- NÃO repita informações que já foram ditas na conversa.
- Se o cliente perguntar algo que você já respondeu, reformule de forma diferente e mais direta.
- Evite frases genéricas repetitivas como "Fico feliz em ajudar!" a cada mensagem.

ESCALAÇÃO INTELIGENTE:
- Se o cliente expressar reclamação grave, insatisfação persistente ou assunto jurídico, ofereça encaminhar para um atendente humano.
- Se após 3 tentativas você não conseguir resolver, sugira atendimento humano.
- Nunca insista quando o cliente pedir para falar com uma pessoa.

MÚLTIPLAS PERGUNTAS:
- Quando o cliente fizer várias perguntas na mesma mensagem, responda TODAS em ordem.
- Numere as respostas se forem mais de 2 perguntas.
- Não ignore nenhuma parte da mensagem do cliente.

INTELIGÊNCIA EMOCIONAL:
- Se detectar frustração, valide o sentimento ANTES de resolver ("Entendo sua frustração...").
- Se detectar urgência, priorize a solução e seja mais direto.
- Se detectar entusiasmo, compartilhe a empolgação e reforce a decisão.
- Adapte o nível de formalidade ao tom do cliente.

FORMATAÇÃO WHATSAPP:
- Lembre-se: você está no WhatsApp. Mensagens devem ser curtas e escaneáveis.
- Use *negrito* para destaques importantes. NÃO use markdown com # ou **.
- Quebre mensagens longas em parágrafos curtos (máx 3-4 linhas por bloco).
- Use listas simples com - ou • para múltiplos itens.
- Evite blocos de texto enormes. Prefira 2-3 mensagens curtas se necessário.

ANTI-LOOP:
- Se perceber que a conversa está andando em círculos, mude a abordagem.
- Ofereça uma solução diferente ou encaminhe para atendimento humano.
- Nunca repita a mesma resposta duas vezes seguidas.""")
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
    # LEVEL 4: MEMÓRIA DO CLIENTE
    # ═══════════════════════════════════════════════
    if memory_context:
        segments.append(f"""[🧠 MEMÓRIA - CONTEXTO DO CLIENTE]
{memory_context}
USE estas informações para personalizar sua resposta. Mencione detalhes que o cliente já compartilhou.""")

    # ═══════════════════════════════════════════════
    # LEVEL 5: ESTILO
    # ═══════════════════════════════════════════════
    tone_desc = _get_tone(tone_level)
    segments.append(f"""[🟢 ESTILO DE COMUNICAÇÃO]
Tom: {tone_desc}
Emojis: {"Use moderadamente" if use_emojis else "Não use emojis"}
Formatação: Use *negrito* para destaques. NÃO use markdown com # ou **.
{"Trate o cliente por " + lead_name + "." if lead_name else ""}""")

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
    else:
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
