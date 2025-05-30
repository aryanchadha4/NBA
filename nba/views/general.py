from django.http import JsonResponse

def home(request):
    """ API: Welcome Message """
    return JsonResponse({"message": "Welcome to NBA Predictions API!"})