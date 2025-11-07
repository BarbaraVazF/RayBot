from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import Chat, Message
import json
from django.contrib.auth import authenticate, login

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
        content = data.get("content")
        Message.objects.create(chat=chat, sender="user", content=content)
        bot_reply = f"💬 Processando sua pergunta sobre {content}..."
        Message.objects.create(chat=chat, sender="bot", content=bot_reply)
        return JsonResponse({"reply": bot_reply})

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