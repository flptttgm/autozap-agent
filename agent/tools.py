"""
============================================
Agent Tools - Autonomous Tool Usage
============================================
Ferramentas que o agente pode usar autonomamente.

O agente decide QUANDO usar cada ferramenta baseado no contexto
da conversa, sem precisar de keyword matching.
"""

from datetime import datetime, timezone, timedelta
from langchain_core.tools import tool
from supabase import Client
import json


def create_tools(supabase: Client, workspace_id: str, lead_id: str, enabled_tools: list[str] | None = None):
    """Cria ferramentas contextualizadas para o agente.

    Args:
        enabled_tools: Lista de nomes de ferramentas habilitadas. Se None, retorna todas.
    """

    # ─────────────────────────────────────────────
    # CORE TOOLS (always active)
    # ─────────────────────────────────────────────

    @tool
    def get_lead_info() -> str:
        """Busca informações cadastrais do lead/cliente atual.
        Use quando precisar do nome, telefone, tags ou classificação do cliente.
        """
        try:
            result = (
                supabase.table("leads")
                .select("name, phone, email, status, score, ai_enabled, metadata, created_at, instagram, linkedin, facebook")
                .eq("id", lead_id)
                .single()
                .execute()
            )

            if not result.data:
                return "Informações do lead não encontradas."

            lead = result.data
            created = datetime.fromisoformat(lead["created_at"].replace("Z", "+00:00"))
            days_since = (datetime.now(timezone.utc) - created).days
            meta = lead.get("metadata") or {}

            lines = [
                f"Nome: {lead.get('name', 'Desconhecido')}",
                f"Telefone: {lead.get('phone', 'N/A')}",
                f"Email: {lead.get('email', 'N/A')}",
                f"Status: {lead.get('status', 'N/A')}",
                f"Score: {lead.get('score', 0)}/100",
                f"Cliente há: {days_since} dias",
            ]

            if lead.get("instagram"):
                lines.append(f"Instagram: {lead['instagram']}")
            if lead.get("linkedin"):
                lines.append(f"LinkedIn: {lead['linkedin']}")
            if lead.get("facebook"):
                lines.append(f"Facebook: {lead['facebook']}")
            if meta.get("company"):
                lines.append(f"Empresa: {meta['company']}")
            if meta.get("notes"):
                lines.append(f"Observações: {meta['notes']}")

            return "\n".join(lines)
        except Exception as e:
            return f"Erro ao buscar info do lead: {e}"

    @tool
    def create_update_lead(name: str = "", email: str = "", status: str = "", notes: str = "", company: str = "") -> str:
        """Captura e atualiza dados do cliente automaticamente.
        Use quando descobrir o nome, email, empresa ou qualquer informação relevante do cliente durante a conversa.
        NÃO precisa esperar o cliente pedir — capture proativamente.

        Args:
            name: Nome do cliente (se souber)
            email: Email do cliente (se fornecido)
            status: Status do lead (new, contacted, qualified, proposal, negotiation, won, lost)
            notes: Observações adicionais sobre o cliente
            company: Nome da empresa do cliente
        """
        try:
            update_data = {}
            if name:
                update_data["name"] = name
            if email:
                update_data["email"] = email
            if status and status in ["new", "contacted", "qualified", "proposal", "negotiation", "won", "lost"]:
                update_data["status"] = status

            # Handle metadata fields (notes, company)
            if notes or company:
                current = supabase.table("leads").select("metadata").eq("id", lead_id).single().execute()
                meta = (current.data or {}).get("metadata") or {}
                if notes:
                    existing_notes = meta.get("notes", "")
                    meta["notes"] = f"{existing_notes}\n{notes}".strip() if existing_notes else notes
                if company:
                    meta["company"] = company
                update_data["metadata"] = meta

            if not update_data:
                return "Nenhum dado para atualizar."

            update_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            supabase.table("leads").update(update_data).eq("id", lead_id).execute()

            updated_fields = [k for k in update_data if k not in ("updated_at", "metadata")]
            if notes:
                updated_fields.append("observações")
            if company:
                updated_fields.append("empresa")

            return f"✅ Lead atualizado: {', '.join(updated_fields)}"
        except Exception as e:
            return f"Erro ao atualizar lead: {e}"

    @tool
    def transfer_to_human(reason: str = "Solicitação do cliente") -> str:
        """Escala a conversa para atendimento humano.
        Use quando o cliente pedir para falar com um atendente, quando não souber responder,
        ou quando a situação exigir intervenção humana.

        Args:
            reason: Motivo da transferência
        """
        try:
            # Pause AI on chat_memory
            result = (
                supabase.table("chat_memory")
                .select("id")
                .eq("workspace_id", workspace_id)
                .eq("lead_id", lead_id)
                .limit(1)
                .execute()
            )

            if result.data:
                supabase.table("chat_memory").update({
                    "ai_paused": True,
                    "pause_reason": f"Transferência: {reason}",
                    "paused_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", result.data[0]["id"]).execute()

            # Also disable AI on the lead
            supabase.table("leads").update({
                "ai_enabled": False,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", lead_id).execute()

            return f"✅ Conversa transferida para atendimento humano. Motivo: {reason}. A IA foi pausada."
        except Exception as e:
            return f"Erro ao transferir: {e}. INFORME AO CLIENTE QUE HOUVE UM ERRO NA TRANSFERÊNCIA."

    @tool
    def search_knowledge_base(query: str) -> str:
        """Consulta a base de conhecimento para respostas precisas.
        Use quando o cliente fizer perguntas sobre produtos, serviços, políticas, ou qualquer informação
        que possa estar documentada na base de conhecimento.

        Args:
            query: Pergunta ou termos de busca
        """
        try:
            # Search by keyword matching in title, content, and keywords
            search_term = f"%{query}%"
            result = (
                supabase.table("knowledge_base")
                .select("title, content, category, keywords, priority")
                .eq("workspace_id", workspace_id)
                .eq("is_active", True)
                .or_(f"title.ilike.{search_term},content.ilike.{search_term}")
                .order("priority", desc=True)
                .limit(5)
                .execute()
            )

            if not result.data:
                return f"Nenhuma informação encontrada na base de conhecimento para: '{query}'."

            items = []
            for kb in result.data:
                items.append(
                    f"📖 {kb['title']} [{kb.get('category', 'geral')}]\n{kb['content'][:500]}"
                )

            return f"Base de conhecimento ({len(result.data)} resultado(s)):\n\n" + "\n\n---\n\n".join(items)
        except Exception as e:
            return f"Erro ao consultar base de conhecimento: {e}"

    @tool
    def register_note(note: str, category: str = "observação") -> str:
        """Adiciona observações e notas ao perfil do cliente.
        Use para registrar informações importantes mencionadas durante a conversa,
        como preferências, reclamações, feedback, etc.

        Args:
            note: Conteúdo da nota
            category: Categoria da nota (observação, preferência, reclamação, feedback)
        """
        try:
            # Store in lead metadata
            current = supabase.table("leads").select("metadata").eq("id", lead_id).single().execute()
            meta = (current.data or {}).get("metadata") or {}

            agent_notes = meta.get("agent_notes", [])
            agent_notes.append({
                "note": note,
                "category": category,
                "created_at": datetime.now(timezone.utc).isoformat(),
            })
            meta["agent_notes"] = agent_notes

            supabase.table("leads").update({
                "metadata": meta,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", lead_id).execute()

            return f"✅ Nota registrada: [{category}] {note}"
        except Exception as e:
            return f"Erro ao registrar nota: {e}"

    # ─────────────────────────────────────────────
    # OPTIONAL TOOLS (toggleable per agent)
    # ─────────────────────────────────────────────

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
            return f"[ERRO] Falha ao verificar disponibilidade: {e}. INFORME AO CLIENTE QUE OCORREU UM ERRO."

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
                return f"[ERRO CRÍTICO] O formato de hora '{time}' ou data '{date}' é inválido. O agendamento NÃO foi criado."

            start_utc = start_dt.astimezone(timezone.utc)
            end_utc = start_utc + timedelta(hours=1)

            supabase.table("appointments").insert({
                "workspace_id": workspace_id,
                "lead_id": lead_id,
                "title": purpose,
                "start_time": start_utc.isoformat(),
                "end_time": end_utc.isoformat(),
                "status": "scheduled",
            }).execute()

            return f"✅ Agendamento criado com sucesso: {purpose} em {date} às {clean_time}."

        except Exception as e:
            return f"[ERRO CRÍTICO] O agendamento FALHOU. Motivo: {e}. NÃO DIGA QUE FOI CONFIRMADO."

    @tool
    def cancel_reschedule(action: str, appointment_title: str = "", new_date: str = "", new_time: str = "") -> str:
        """Cancela ou reagenda um agendamento existente do cliente.
        SEMPRE confirme com o cliente antes de cancelar ou mover.

        Args:
            action: 'cancel' para cancelar, 'reschedule' para reagendar
            appointment_title: Título ou descrição do agendamento (para identificar)
            new_date: Nova data (YYYY-MM-DD) - obrigatório para reagendamento
            new_time: Novo horário (HH:MM) - obrigatório para reagendamento
        """
        import re
        try:
            # Find the appointment
            now = datetime.now(timezone.utc).isoformat()
            query = (
                supabase.table("appointments")
                .select("id, title, start_time, end_time")
                .eq("lead_id", lead_id)
                .eq("workspace_id", workspace_id)
                .neq("status", "cancelled")
                .gte("start_time", now)
                .order("start_time", desc=False)
                .limit(5)
                .execute()
            )

            if not query.data:
                return "O cliente não tem agendamentos futuros para alterar."

            # Find best match
            target = query.data[0]  # Default to next appointment
            if appointment_title:
                for a in query.data:
                    if appointment_title.lower() in a["title"].lower():
                        target = a
                        break

            if action == "cancel":
                supabase.table("appointments").update({
                    "status": "cancelled",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", target["id"]).execute()

                start = datetime.fromisoformat(target["start_time"].replace("Z", "+00:00")) - timedelta(hours=3)
                return f"✅ Agendamento cancelado: {target['title']} ({start.strftime('%d/%m às %H:%M')})"

            elif action == "reschedule":
                if not new_date or not new_time:
                    return "Para reagendar, informe a nova data (new_date) e horário (new_time)."

                clean_time = re.sub(r'[^0-9:]', '', new_time)
                if len(clean_time) <= 2:
                    clean_time = f"{clean_time.zfill(2)}:00"
                elif len(clean_time) >= 5:
                    clean_time = clean_time[:5]

                try:
                    new_start = datetime.fromisoformat(f"{new_date}T{clean_time}:00-03:00").astimezone(timezone.utc)
                except ValueError:
                    return f"[ERRO] Data/hora inválida: {new_date} {new_time}"

                new_end = new_start + timedelta(hours=1)

                supabase.table("appointments").update({
                    "start_time": new_start.isoformat(),
                    "end_time": new_end.isoformat(),
                    "status": "rescheduled",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", target["id"]).execute()

                return f"✅ Agendamento reagendado: {target['title']} → {new_date} às {clean_time}"
            else:
                return "Ação inválida. Use 'cancel' ou 'reschedule'."

        except Exception as e:
            return f"Erro ao alterar agendamento: {e}"

    @tool
    def send_quote(description: str, amount: float, due_days: int = 7) -> str:
        """Gera e envia um orçamento com base nos serviços discutidos.
        Use quando o cliente pedir um orçamento ou quando os serviços e valores forem acordados.

        Args:
            description: Descrição dos serviços/produtos incluídos no orçamento
            amount: Valor total em reais (R$)
            due_days: Dias de validade do orçamento (padrão: 7)
        """
        try:
            due_date = (datetime.now(timezone.utc) + timedelta(days=due_days)).strftime("%Y-%m-%d")

            result = supabase.table("invoices").insert({
                "workspace_id": workspace_id,
                "lead_id": lead_id,
                "amount": amount,
                "description": description,
                "due_date": due_date,
                "status": "pending",
                "source": "agent",
            }).execute()

            return (
                f"✅ Orçamento criado com sucesso!\n"
                f"Descrição: {description}\n"
                f"Valor: R$ {amount:.2f}\n"
                f"Válido até: {due_date}\n"
                f"O orçamento foi salvo e pode ser visualizado pelo administrador."
            )
        except Exception as e:
            return f"Erro ao criar orçamento: {e}"

    @tool
    def query_products(search: str) -> str:
        """Busca no catálogo de produtos (preço, disponibilidade, descrição).
        Use quando o cliente perguntar sobre produtos, preços ou disponibilidade.

        Args:
            search: Nome ou descrição do produto a buscar
        """
        try:
            # Search in knowledge_base with category 'products' or 'pricing'
            search_term = f"%{search}%"
            result = (
                supabase.table("knowledge_base")
                .select("title, content, keywords")
                .eq("workspace_id", workspace_id)
                .eq("is_active", True)
                .in_("category", ["products", "pricing", "services"])
                .or_(f"title.ilike.{search_term},content.ilike.{search_term}")
                .limit(5)
                .execute()
            )

            if not result.data:
                # Fallback: search all knowledge base
                result = (
                    supabase.table("knowledge_base")
                    .select("title, content")
                    .eq("workspace_id", workspace_id)
                    .eq("is_active", True)
                    .or_(f"title.ilike.{search_term},content.ilike.{search_term}")
                    .limit(3)
                    .execute()
                )

            if not result.data:
                return f"Nenhum produto encontrado para '{search}'. Consulte a base de conhecimento ou o administrador."

            items = []
            for p in result.data:
                items.append(f"📦 {p['title']}\n{p['content'][:300]}")

            return f"Produtos encontrados ({len(result.data)}):\n\n" + "\n\n---\n\n".join(items)
        except Exception as e:
            return f"Erro ao consultar produtos: {e}"

    @tool
    def check_conversation_history(query: str = "") -> str:
        """Busca conversas anteriores do cliente para contexto.
        Use quando precisar de contexto sobre interações passadas,
        ou quando o cliente mencionar algo discutido anteriormente.

        Args:
            query: Termo opcional para filtrar nas mensagens
        """
        try:
            # Get recent messages for this lead
            builder = (
                supabase.table("messages")
                .select("content, direction, message_type, created_at")
                .eq("workspace_id", workspace_id)
                .eq("lead_id", lead_id)
                .order("created_at", desc=True)
                .limit(30)
            )

            result = builder.execute()

            if not result.data:
                return "Sem histórico de conversas para este cliente."

            messages = list(reversed(result.data))

            # Filter by query if provided
            if query:
                query_lower = query.lower()
                messages = [m for m in messages if m.get("content") and query_lower in m["content"].lower()]
                if not messages:
                    return f"Nenhuma mensagem encontrada com o termo '{query}'."

            history = []
            for m in messages[-20:]:  # Last 20 relevant messages
                direction = "👤 Cliente" if m["direction"] == "inbound" else "🤖 Agente"
                content = (m.get("content") or "[mídia]")[:200]
                dt = datetime.fromisoformat(m["created_at"].replace("Z", "+00:00")) - timedelta(hours=3)
                history.append(f"[{dt.strftime('%d/%m %H:%M')}] {direction}: {content}")

            return f"Histórico ({len(history)} mensagens):\n" + "\n".join(history)
        except Exception as e:
            return f"Erro ao buscar histórico: {e}"

    @tool
    def check_order_status(order_reference: str = "") -> str:
        """Consulta status de pedidos para e-commerce/delivery.
        Use quando o cliente perguntar sobre status de pedido, entrega ou envio.

        Args:
            order_reference: Número ou referência do pedido (opcional)
        """
        try:
            # Check in invoices as order proxy
            builder = (
                supabase.table("invoices")
                .select("id, description, amount, status, due_date, created_at, paid_at")
                .eq("workspace_id", workspace_id)
                .eq("lead_id", lead_id)
                .order("created_at", desc=True)
                .limit(5)
            )

            result = builder.execute()

            if not result.data:
                return "Nenhum pedido/orçamento encontrado para este cliente."

            orders = []
            status_map = {
                "pending": "⏳ Pendente",
                "sent": "📤 Enviado",
                "paid": "✅ Pago",
                "overdue": "⚠️ Vencido",
                "canceled": "❌ Cancelado",
            }

            for o in result.data:
                dt = datetime.fromisoformat(o["created_at"].replace("Z", "+00:00")) - timedelta(hours=3)
                status_label = status_map.get(o["status"], o["status"])
                orders.append(
                    f"- {o['description'] or 'Sem descrição'} | R$ {o['amount']:.2f} | {status_label} | {dt.strftime('%d/%m/%Y')}"
                )

            return f"Pedidos/Orçamentos do cliente:\n" + "\n".join(orders)
        except Exception as e:
            return f"Erro ao consultar pedidos: {e}"

    @tool
    def send_payment_link(amount: float, description: str = "Pagamento") -> str:
        """Gera um link de pagamento (PIX, Stripe, etc.).
        Use quando o cliente quiser pagar ou quando um orçamento for aprovado.

        Args:
            amount: Valor em reais
            description: Descrição do pagamento
        """
        try:
            # Check if PIX is configured
            pix_config = (
                supabase.table("pix_config")
                .select("pix_key, pix_key_type, receiver_name, receiver_city, is_active")
                .eq("workspace_id", workspace_id)
                .eq("is_active", True)
                .limit(1)
                .execute()
            )

            if pix_config.data:
                config = pix_config.data[0]
                # Create invoice with PIX info
                invoice = supabase.table("invoices").insert({
                    "workspace_id": workspace_id,
                    "lead_id": lead_id,
                    "amount": amount,
                    "description": description,
                    "due_date": (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%Y-%m-%d"),
                    "status": "sent",
                    "sent_at": datetime.now(timezone.utc).isoformat(),
                    "source": "agent",
                }).execute()

                return (
                    f"✅ Cobrança criada!\n"
                    f"Valor: R$ {amount:.2f}\n"
                    f"Chave PIX ({config['pix_key_type']}): {config['pix_key']}\n"
                    f"Beneficiário: {config['receiver_name']}\n"
                    f"Informe a chave PIX ao cliente para pagamento."
                )
            else:
                return (
                    f"⚠️ PIX não configurado para este workspace.\n"
                    f"Cobrança de R$ {amount:.2f} registrada, mas sem link de pagamento automático.\n"
                    f"O administrador precisa configurar o PIX nas configurações."
                )
        except Exception as e:
            return f"Erro ao gerar link de pagamento: {e}"

    @tool
    def calculate_shipping(zip_code: str = "", product_description: str = "") -> str:
        """Calcula frete e prazo de entrega.
        Use quando o cliente perguntar sobre frete, envio ou prazo de entrega.

        Args:
            zip_code: CEP de destino
            product_description: Descrição do produto para cálculo
        """
        try:
            # Search in knowledge base for shipping info
            result = (
                supabase.table("knowledge_base")
                .select("title, content")
                .eq("workspace_id", workspace_id)
                .eq("is_active", True)
                .in_("category", ["shipping", "delivery", "logistics", "services"])
                .limit(3)
                .execute()
            )

            if result.data:
                info = []
                for kb in result.data:
                    info.append(f"📦 {kb['title']}: {kb['content'][:300]}")
                return (
                    f"Informações de frete/entrega encontradas:\n\n"
                    + "\n\n".join(info)
                    + f"\n\nCEP informado: {zip_code or 'não informado'}"
                )

            return (
                f"ℹ️ Informações de frete não configuradas na base de conhecimento.\n"
                f"CEP: {zip_code or 'não informado'}\n"
                f"Produto: {product_description or 'não especificado'}\n"
                f"Consulte o administrador para detalhes de frete e envio."
            )
        except Exception as e:
            return f"Erro ao calcular frete: {e}"

    @tool
    def summarize_conversation() -> str:
        """Gera um resumo da conversa para handoff humano.
        Use antes de transferir para um atendente humano, ou quando solicitado.
        O resumo ajuda o atendente a entender rapidamente o contexto.
        """
        try:
            # Get recent messages
            result = (
                supabase.table("messages")
                .select("content, direction, created_at")
                .eq("workspace_id", workspace_id)
                .eq("lead_id", lead_id)
                .eq("message_type", "text")
                .order("created_at", desc=True)
                .limit(30)
                .execute()
            )

            if not result.data:
                return "Sem histórico para resumir."

            messages = list(reversed(result.data))

            # Get lead info for context
            lead_result = (
                supabase.table("leads")
                .select("name, phone, status, score")
                .eq("id", lead_id)
                .single()
                .execute()
            )

            lead_info = lead_result.data or {}

            # Build summary context
            total_msgs = len(messages)
            client_msgs = [m for m in messages if m["direction"] == "inbound"]
            agent_msgs = [m for m in messages if m["direction"] != "inbound"]

            first_msg = datetime.fromisoformat(messages[0]["created_at"].replace("Z", "+00:00")) - timedelta(hours=3)
            last_msg = datetime.fromisoformat(messages[-1]["created_at"].replace("Z", "+00:00")) - timedelta(hours=3)

            # Extract key messages (first and last from each side)
            key_topics = []
            for m in client_msgs[-5:]:
                if m.get("content"):
                    key_topics.append(m["content"][:100])

            summary = (
                f"📋 RESUMO DA CONVERSA\n"
                f"Cliente: {lead_info.get('name', 'Desconhecido')} ({lead_info.get('phone', 'N/A')})\n"
                f"Status: {lead_info.get('status', 'N/A')} | Score: {lead_info.get('score', 0)}\n"
                f"Período: {first_msg.strftime('%d/%m %H:%M')} → {last_msg.strftime('%d/%m %H:%M')}\n"
                f"Mensagens: {total_msgs} total ({len(client_msgs)} do cliente, {len(agent_msgs)} do agente)\n"
                f"\nÚltimas mensagens do cliente:\n"
                + "\n".join(f"  • {t}" for t in key_topics)
            )

            return summary
        except Exception as e:
            return f"Erro ao gerar resumo: {e}"

    @tool
    def create_followup_task(description: str, days_from_now: int = 1, time: str = "09:00") -> str:
        """Agenda um lembrete para recontatar o cliente.
        Use quando for necessário fazer follow-up futuro,
        ou quando o cliente pedir para ser contatado em outro momento.

        Args:
            description: O que precisa ser feito no follow-up
            days_from_now: Daqui a quantos dias (padrão: 1 = amanhã)
            time: Horário do lembrete (padrão: 09:00)
        """
        import re
        try:
            clean_time = re.sub(r'[^0-9:]', '', time)
            if len(clean_time) <= 2:
                clean_time = f"{clean_time.zfill(2)}:00"
            elif len(clean_time) >= 5:
                clean_time = clean_time[:5]

            target_date = (datetime.now(timezone.utc) - timedelta(hours=3) + timedelta(days=days_from_now)).strftime("%Y-%m-%d")

            try:
                start_dt = datetime.fromisoformat(f"{target_date}T{clean_time}:00-03:00").astimezone(timezone.utc)
            except ValueError:
                return f"[ERRO] Horário inválido: {time}"

            end_dt = start_dt + timedelta(minutes=30)

            supabase.table("appointments").insert({
                "workspace_id": workspace_id,
                "lead_id": lead_id,
                "title": f"📌 Follow-up: {description}",
                "description": f"Tarefa de follow-up criada pelo agente IA.\nMotivo: {description}",
                "start_time": start_dt.isoformat(),
                "end_time": end_dt.isoformat(),
                "status": "scheduled",
                "metadata": json.dumps({"type": "followup", "created_by": "agent"}),
            }).execute()

            return (
                f"✅ Follow-up agendado!\n"
                f"Data: {target_date} às {clean_time}\n"
                f"Tarefa: {description}"
            )
        except Exception as e:
            return f"Erro ao criar follow-up: {e}"

    # ─────────────────────────────────────────────
    # TOOL REGISTRY
    # ─────────────────────────────────────────────

    all_tools = {
        # Core (always active)
        "get_lead_info": get_lead_info,
        "create_update_lead": create_update_lead,
        "transfer_to_human": transfer_to_human,
        "search_knowledge_base": search_knowledge_base,
        "register_note": register_note,
        # Optional (toggleable)
        "check_appointments": check_appointments,
        "check_availability": check_availability,
        "schedule_appointment": schedule_appointment,
        "cancel_reschedule": cancel_reschedule,
        "send_quote": send_quote,
        "query_products": query_products,
        "check_conversation_history": check_conversation_history,
        "check_order_status": check_order_status,
        "send_payment_link": send_payment_link,
        "calculate_shipping": calculate_shipping,
        "summarize_conversation": summarize_conversation,
        "create_followup_task": create_followup_task,
    }

    if enabled_tools:
        return [all_tools[name] for name in enabled_tools if name in all_tools]
    return list(all_tools.values())
