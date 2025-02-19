from django.urls import path
from .views import player_stats, home


urlpatterns = [
    path("api/player-stats/", player_stats),
    path('', home),  # Homepage at `/`
]