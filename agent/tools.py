"""
============================================
Agent Tools - Autonomous Tool Usage
============================================
Melhoria #2: Ferramentas que o agente pode usar autonomamente.

O agente decide QUANDO usar cada ferramenta baseado no contexto
da conversa, sem precisar de keyword matching.
"""

from datetime import datetime, timezone, timedelta
from langchain_core.tools import tool
from supabase import Client


def create_tools(supabase: Client, workspace_id: str, lead_id: str, enabled_tools: list[str] | None = None):
    """Cria ferramentas contextualizadas para o agente.

    Args:
        enabled_tools: Lista de nomes de ferramentas habilitadas. Se None, retorna todas.
    """

    @tool
    def check_appointments(query: str = "") -> str:
        """Consulta os agendamentos existentes do cliente.
        Use quando o cliente perguntar sobre horários, consultas marcadas,
        ou quando precisar verificar disponibilidade.

        Args:
            query: Descrição do que procurar (ex: "próximos agendamentos")
        """
        try:
            now = datetime.now(timezone.utc).isoformat()
            result = (
                supabase.table("appointments")
                .select("title, start_time, end_time, status")
                .eq("lead_id", lead_id)
                .eq("workspace_id", workspace_id)
                .neq("status", "cancelled")
                .gte("start_time", now)
                .order("start_time", desc=False)
                .limit(5)
                .execute()
            )

            if not result.data:
                return "O cliente não tem agendamentos futuros."

            appointments = []
            for a in result.data:
                start = datetime.fromisoformat(a["start_time"].replace("Z", "+00:00"))
                local_start = start - timedelta(hours=3)  # UTC-3
                appointments.append(
                    f"- {a['title']}: {local_start.strftime('%d/%m às %H:%M')} ({a['status']})"
                )

            return f"Agendamentos do cliente:\n" + "\n".join(appointments)

        except Exception as e:
            return f"Erro ao consultar agendamentos: {e}"

    @tool
    def check_availability(date: str, time: str = "") -> str:
        """Verifica se um horário está disponível para agendamento.
        Use quando o cliente quiser marcar algo ou perguntar se tem vaga.

        Args:
            date: Data no formato YYYY-MM-DD
            time: Horário no formato HH:MM (opcional)
        """
        import re
        try:
            if time:
                # Robust time cleaning (e.g., '10h' -> '10:00', '10:00:00' -> '10:00')
                clean_time = re.sub(r'[^0-9:]', '', time)
                if len(clean_time) == 1 or len(clean_time) == 2:
                    clean_time = f"{clean_time.zfill(2)}:00"
                elif len(clean_time) >= 5 and clean_time.count(':') >= 1:
                    clean_time = clean_time[:5]
                elif len(clean_time) == 4 and ':' not in clean_time:
                    clean_time = f"{clean_time[:2]}:{clean_time[2:]}"
                
                try:
                    start_dt = datetime.fromisoformat(f"{date}T{clean_time}:00-03:00")
                except ValueError:
                    return f"[ERRO] Formato de data/hora inválido. Recebido date='{date}' e time='{time}'. Corrija e tente novamente."

                start_utc = start_dt.astimezone(timezone.utc)
                end_utc = start_utc + timedelta(hours=1)

                result = (
                    supabase.table("appointments")
                    .select("id")
                    .eq("workspace_id", workspace_id)
                    .neq("status", "cancelled")
                    .lte("start_time", end_utc.isoformat())
                    .gte("end_time", start_utc.isoformat())
                    .limit(1)
                    .execute()
                )

                if result.data:
                    return f"O horário {date} às {time} já está OCUPADO."
                return f"O horário {date} às {time} está DISPONÍVEL! ✅"
            else:
                # Verifica todos os horários do dia
                day_start = datetime.fromisoformat(f"{date}T08:00:00-03:00").astimezone(timezone.utc)
                day_end = datetime.fromisoformat(f"{date}T18:00:00-03:00").astimezone(timezone.utc)

                result = (
                    supabase.table("appointments")
                    .select("start_time, end_time")
                    .eq("workspace_id", workspace_id)
                    .neq("status", "cancelled")
                    .gte("start_time", day_start.isoformat())
                    .lte("start_time", day_end.isoformat())
                    .order("start_time", desc=False)
                    .execute()
                )

                if not result.data:
                    return f"O dia {date} está completamente livre!"

                busy_times = []
                for a in result.data:
                    s = datetime.fromisoformat(a["start_time"].replace("Z", "+00:00")) - timedelta(hours=3)
                    e = datetime.fromisoformat(a["end_time"].replace("Z", "+00:00")) - timedelta(hours=3)
                    busy_times.append(f"- {s.strftime('%H:%M')} a {e.strftime('%H:%M')}")

                return f"Horários ocupados em {date}:\n" + "\n".join(busy_times)

        except Exception as e:
            return f"[ERRO INFERNO] Houve uma falha catastrófica ao verificar disponibilidade: {e}. VOCÊ DEVE INFORMAR AO CLIENTE QUE OCORREU UM ERRO E NÃO FOI POSSÍVEL VERIFICAR."

    @tool
    def schedule_appointment(date: str, time: str, purpose: str = "Agendamento via WhatsApp") -> str:
        """Cria um novo agendamento para o cliente.
        Use SOMENTE após confirmar com o cliente que ele deseja marcar.
        SEMPRE verifique disponibilidade antes de agendar.

        Args:
            date: Data no formato YYYY-MM-DD
            time: Horário no formato HH:MM
            purpose: Descrição do motivo do agendamento
        """
        import re
        try:
            # Robust time cleaning
            clean_time = re.sub(r'[^0-9:]', '', time)
            if len(clean_time) == 1 or len(clean_time) == 2:
                clean_time = f"{clean_time.zfill(2)}:00"
            elif len(clean_time) >= 5 and clean_time.count(':') >= 1:
                clean_time = clean_time[:5]
            elif len(clean_time) == 4 and ':' not in clean_time:
                clean_time = f"{clean_time[:2]}:{clean_time[2:]}"

            try:
                start_dt = datetime.fromisoformat(f"{date}T{clean_time}:00-03:00")
            except ValueError:
                return f"[ERRO CRÍTICO] Falha. O formato de hora '{time}' ou data '{date}' é inválido. O agendamento NÃO foi criado. VOCÊ DEVE INFORMAR O CLIENTE QUE FOI IMPOSSÍVEL AGENDAR."

            start_utc = start_dt.astimezone(timezone.utc)
            end_utc = start_utc + timedelta(hours=1)

            result = (
                supabase.table("appointments")
                .insert({
                    "workspace_id": workspace_id,
                    "lead_id": lead_id,
                    "title": purpose,
                    "start_time": start_utc.isoformat(),
                    "end_time": end_utc.isoformat(),
                    "status": "scheduled",
                })
                .execute()
            )

            return f"✅ Agendamento criado com sucesso: {purpose} em {date} às {clean_time}."

        except Exception as e:
            return f"[ERRO CRÍTICO ABSOLUTO] O agendamento FALHOU e NÃO FOI SALVO no banco de dados. Motivo: {e}. VOCÊ ESTÁ PROIBIDO DE DIZER QUE FOI CONFIRMADO. INFORME AO CLIENTE QUE OCORREU UM ERRO TÉCNICO E NÃO FOI POSSÍVEL AGENDAR."

    @tool
    def get_lead_info() -> str:
        """Busca informações cadastrais do lead/cliente atual.
        Use quando precisar do nome, telefone, tags ou classificação do cliente.
        """
        try:
            result = (
                supabase.table("leads")
                .select("name, phone, status, ai_enabled, tags, created_at")
                .eq("id", lead_id)
                .single()
                .execute()
            )

            if not result.data:
                return "Informações do lead não encontradas."

            lead = result.data
            created = datetime.fromisoformat(lead["created_at"].replace("Z", "+00:00"))
            days_since = (datetime.now(timezone.utc) - created).days

            return (
                f"Nome: {lead.get('name', 'Desconhecido')}\n"
                f"Telefone: {lead.get('phone', 'N/A')}\n"
                f"Status: {lead.get('status', 'N/A')}\n"
                f"Tags: {', '.join(lead.get('tags', []) or [])}\n"
                f"Cliente há: {days_since} dias"
            )
        except Exception as e:
            return f"Erro ao buscar info do lead: {e}"

    all_tools = {
        "check_appointments": check_appointments,
        "check_availability": check_availability,
        "schedule_appointment": schedule_appointment,
        "get_lead_info": get_lead_info,
    }

    if enabled_tools:
        return [all_tools[name] for name in enabled_tools if name in all_tools]
    return list(all_tools.values())
