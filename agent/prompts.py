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
    """Constrói prompt do sistema dinâmico e hierárquico."""

    identity = agent_config.get("persona_name", "Assistente")
    agent_type = agent_config.get("agent_type", "Atendente")
    personality = agent_config.get("personality", {})
    behavior = agent_config.get("behavior", {})

    tone_level = personality.get("tone", 50)
    use_emojis = personality.get("useEmojis", True)
    niche_scope = agent_type

    # Timezone-aware date
    tz_offset = behavior.get("timezone", -3)
    now = datetime.now(timezone.utc) + timedelta(hours=tz_offset)
    date_str = now.strftime("%A, %d de %B de %Y, %H:%M")

    segments = []

    # ═══════════════════════════════════════════════
    # LEVEL 1: REGRAS DE OURO (Identidade & Segurança)
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
    # LEVEL 2: ESCOPO
    # ═══════════════════════════════════════════════
    if niche_scope:
        segments.append(f"""[🟣 ESCOPO DE ATUAÇÃO]
Seu escopo é: {niche_scope}.
Se perguntarem algo fora disso, recuse educadamente e redirecione.""")

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
    # LEVEL 5: INICIATIVA & OBJETIVO
    # ═══════════════════════════════════════════════
    business_goal = behavior.get("business_goal", "ajudar o cliente da melhor forma possível")
    segments.append(f"""[🎯 OBJETIVO DE NEGÓCIO]
Seu objetivo principal é: {business_goal}

INICIATIVA:
- NÃO espere o cliente pedir. Se perceber uma oportunidade, proponha.
- Se o cliente demonstrar interesse, avance para o próximo passo naturalmente.
- Se a conversa estagnar, faça uma pergunta relevante para reengajar.
- Ofereça alternativas quando possível.
{"- Se for cliente retornando, demonstre que se lembra dele!" if is_returning else ""}""")

    # ═══════════════════════════════════════════════
    # LEVEL 6: ESTILO
    # ═══════════════════════════════════════════════
    tone_desc = _get_tone(tone_level)
    segments.append(f"""[🟢 ESTILO DE COMUNICAÇÃO]
Tom: {tone_desc}
Emojis: {"Use moderadamente" if use_emojis else "Não use emojis"}
Formatação: Use *negrito* para destaques. NÃO use markdown com # ou **.
{"Trate o cliente por " + lead_name + "." if lead_name else ""}""")

    # ═══════════════════════════════════════════════
    # LEVEL 7: FERRAMENTAS
    # ═══════════════════════════════════════════════
    segments.append("""[🔧 USO DE FERRAMENTAS]
Você tem acesso a ferramentas para consultar agendamentos, verificar disponibilidade e agendar.
- Use-as PROATIVAMENTE quando perceber a necessidade, sem esperar o cliente pedir explicitamente.
- SEMPRE verifique disponibilidade ANTES de sugerir um horário.
- SEMPRE confirme com o cliente ANTES de agendar definitivamente.""")

    return "\n\n".join(segments)


def _get_tone(level: int) -> str:
    if level < 30:
        return "Formal e respeitoso. Use linguagem profissional."
    if level > 70:
        return "Animado, amigável e casual. Use linguagem descontraída."
    return "Profissional, mas simpático e acolhedor. Equilibrado."
