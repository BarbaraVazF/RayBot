from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponse
from django.contrib.auth.decorators import login_required
from .models import Chat, Message
from django.contrib import messages
import json
from django.contrib.auth import authenticate, login
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Flowable
from datetime import datetime
import urllib.parse
from django.contrib.auth import update_session_auth_hash
from utils.pandas_agent import carregar_dataframe, criar_agente
from utils.prompts import gerar_prompt

def home(request):
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect("dashboard")
        else:
            return render(request, "home.html", {"form": {"errors": True}})

    return render(request, "home.html")


@login_required
def dashboard(request):
    return render(request, "dashboard.html")

@login_required
def get_chats(request):
    chats = Chat.objects.filter(user=request.user).values("id", "name")
    return JsonResponse(list(chats), safe=False)

@login_required
def get_messages(request, chat_id):
    chat = get_object_or_404(Chat, id=chat_id, user=request.user)
    messages = chat.messages.values("sender", "content")
    return JsonResponse(list(messages), safe=False)

@login_required
def create_chat(request):
    if request.method == "POST":
        data = json.loads(request.body)
        chat_name = data.get("name")
        chat = Chat.objects.create(user=request.user, name=chat_name)
        Message.objects.create(chat=chat, sender="bot", content="👋 Novo chat criado! Vamos conversar.")
        return JsonResponse({"id": chat.id, "name": chat.name})

@login_required
def send_message(request, chat_id):
    if request.method == "POST":
        chat = get_object_or_404(Chat, id=chat_id, user=request.user)
        data = json.loads(request.body)
        pergunta = data.get("content")
        # Salvar pergunta do usuário
        Message.objects.create(chat=chat, sender="user", content=pergunta)
        # ==== Carregar dataframe e agente ====
        df = carregar_dataframe()
        agente = criar_agente(df)
        # ==== Capturar histórico ====
        historico = list(
            chat.messages.filter(sender="user").values_list("content", flat=True)
        )
        # ==== Criar prompt ====
        prompt = gerar_prompt(pergunta, historico, df)
        # ==== Rodar análise ====
        try:
            resposta = agente.invoke({"input": prompt})
            bot_texto = resposta.get("output", "Não consegui processar sua solicitação.")
        except Exception as e:
            bot_texto = f"❌ Erro ao analisar os dados: {str(e)}"
        # ==== Salvar resposta ====
        Message.objects.create(chat=chat, sender="bot", content=bot_texto)
        return JsonResponse({"reply": bot_texto})

@login_required
def delete_chat(request, chat_id):
    if request.method == "DELETE":
        chat = get_object_or_404(Chat, id=chat_id, user=request.user)
        chat.delete()
        return JsonResponse({"success": True})
    
@login_required
def clear_chat(request, chat_id):
    if request.method == "POST":
        chat = get_object_or_404(Chat, id=chat_id, user=request.user)
        chat.messages.all().delete()
        Message.objects.create(chat=chat, sender="bot", content="🧹 Chat limpo! Pode começar uma nova conversa.")
        return JsonResponse({"success": True})

class Bubble(Flowable):
    def __init__(self, width, height):
        Flowable.__init__(self)
        self.width = width
        self.height = height

def exportar_pdf(request, chat_id):
    chat = get_object_or_404(Chat, id=chat_id)

    filename_qs = request.GET.get("filename")
    safe_name = (filename_qs.strip() if filename_qs else chat.name).replace(" ", "_")
    safe_name = urllib.parse.quote(safe_name, safe='')  
    filename = f"{safe_name}.pdf"

    messages = Message.objects.filter(chat=chat).order_by("created_at")

    response = HttpResponse(content_type="application/pdf")
    response['Content-Disposition'] = f'attachment; filename="{filename}"'

    doc = SimpleDocTemplate(response, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    story = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading2'],
        alignment=1,  
        spaceAfter=12,
    )

    meta_style = ParagraphStyle(
        'Meta',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.grey,
        alignment=1,
        spaceAfter=12
    )

    bot_style = ParagraphStyle(
        "Bot",
        parent=styles['BodyText'],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=colors.black,
        backColor=colors.HexColor("#E6E7E8"), 
        leftIndent=0,
        rightIndent=60,
        borderPadding=6,
        spaceAfter=8,
    )

    user_style = ParagraphStyle(
        "User",
        parent=styles['BodyText'],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#040213"),  
        backColor=colors.HexColor("#7C5CE6"),  
        leftIndent=60,
        rightIndent=0,
        borderPadding=6,
        spaceAfter=8,
    )

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    story.append(Paragraph(f"RayBot — Conversa: {chat.name}", title_style))
    story.append(Paragraph(f"Usuário: {chat.user.username} • Exportado em: {now_str}", meta_style))
    story.append(Spacer(1, 0.2*cm))

    if not messages.exists():
        story.append(Paragraph("Sem mensagens nesta conversa.", styles['Normal']))
    else:
        for m in messages:
            text = m.content.replace("\n", "<br/>")
            if m.sender == "user":
                story.append(Paragraph(text, user_style))
            else:
                # bot
                story.append(Paragraph(text, bot_style))

    doc.build(story)
    return response

@login_required
def trocar_senha(request):
    if request.method == "POST":
        senha_atual = request.POST.get("senha_atual")
        nova_senha = request.POST.get("nova_senha")
        confirmar_senha = request.POST.get("confirmar_senha")

        if not request.user.check_password(senha_atual):
            messages.error(request, "❌ A senha atual está incorreta.")
        elif nova_senha != confirmar_senha:
            messages.error(request, "⚠️ As senhas não coincidem.")
        elif len(nova_senha) < 6:
            messages.error(request, "🔒 A nova senha deve ter pelo menos 6 caracteres.")
        else:
            request.user.set_password(nova_senha)
            request.user.save()
            update_session_auth_hash(request, request.user)
            return redirect("dashboard")

    return render(request, "trocar_senha.html")