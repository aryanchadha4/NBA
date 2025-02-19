import os
import pandas as pd
from django.http import JsonResponse

# Set BASE_DIR to `NBA/src/`
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# CSV files are stored in `NBA/src/`
CSV_DIR = BASE_DIR

def home(request):
    return JsonResponse({"message": "Welcome to the NBA Predictions API! Visit /api/player-stats/ for data."})


def player_stats(request):
    try:
        # Ensure directory exists
        if not os.path.exists(CSV_DIR):
            return JsonResponse({"error": f"Directory not found: {CSV_DIR}"}, status=500)

        # Get all CSV files
        csv_files = [f for f in os.listdir(CSV_DIR) if f.endswith(".csv")]
        if not csv_files:
            return JsonResponse({"error": "No CSV files found in the directory"}, status=404)

        data = {}
        for file in csv_files:
            file_path = os.path.join(CSV_DIR, file)
            df = pd.read_csv(file_path)

            # Convert to records and format as a list
            data[file] = df.to_dict(orient="records")

        # Return pretty-formatted JSON response
        return JsonResponse({"files": data}, json_dumps_params={'indent': 4})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
