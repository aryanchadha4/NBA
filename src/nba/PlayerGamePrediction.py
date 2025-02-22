from django.http import JsonResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from thefuzz import process
import os

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data_files")