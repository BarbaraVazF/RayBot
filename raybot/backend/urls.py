from django.contrib.auth.views import LogoutView
from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path('dashboard/', views.dashboard, name='dashboard'),
    path("logout/", LogoutView.as_view(next_page="/"), name="logout"),
    path("api/chats/<int:chat_id>/clear/", views.clear_chat, name="clear_chat"),
    path("api/chats/", views.get_chats, name="get_chats"),
    path("api/chats/new/", views.create_chat, name="create_chat"),
    path("api/chats/<int:chat_id>/messages/", views.get_messages, name="get_messages"),
    path("api/chats/<int:chat_id>/send/", views.send_message, name="send_message"),
    path("api/chats/<int:chat_id>/delete/", views.delete_chat, name="delete_chat"),
]