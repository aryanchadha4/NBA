from django.urls import path
from .views import predict_game, home

urlpatterns = [
    path("api/game-prediction/", predict_game),
    path('', home),
]
