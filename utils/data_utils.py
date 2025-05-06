"""
Fonctions utilitaires pour la manipulation des données
"""
import logging
import pandas as pd
import numpy as np
import json
import sys
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json_file(file_path):
    """
    Charge un fichier JSON
    
    Args:
        file_path (str): Chemin vers le fichier JSON
        
    Returns:
        dict: Données JSON chargées
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Fichier JSON chargé: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier JSON {file_path}: {e}")
        return None

def save_json_file(data, file_path):
    """
    Sauvegarde des données au format JSON
    
    Args:
        data: Données à sauvegarder
        file_path (str): Chemin où sauvegarder le fichier
        
    Returns:
        bool: True si la sauvegarde a réussi, False sinon
    """
    try:
        # Créer le répertoire parent s'il n'existe pas
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Données sauvegardées dans: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde des données dans {file_path}: {e}")
        return False

def load_csv_file(file_path, **kwargs):
    """
    Charge un fichier CSV en DataFrame pandas
    
    Args:
        file_path (str): Chemin vers le fichier CSV
        **kwargs: Arguments supplémentaires pour pd.read_csv
        
    Returns:
        DataFrame: DataFrame pandas contenant les données
    """
    try:
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Fichier CSV chargé: {file_path} ({len(df)} lignes)")
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement du fichier CSV {file_path}: {e}")
        return None

def save_csv_file(df, file_path, **kwargs):
    """
    Sauvegarde un DataFrame pandas au format CSV
    
    Args:
        df (DataFrame): DataFrame à sauvegarder
        file_path (str): Chemin où sauvegarder le fichier
        **kwargs: Arguments supplémentaires pour df.to_csv
        
    Returns:
        bool: True si la sauvegarde a réussi, False sinon
    """
    try:
        # Créer le répertoire parent s'il n'existe pas
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        df.to_csv(file_path, **kwargs)
        logger.info(f"DataFrame sauvegardé dans: {file_path} ({len(df)} lignes)")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du DataFrame dans {file_path}: {e}")
        return False

def normalize_features(df, features):
    """
    Normalise les caractéristiques numériques d'un DataFrame
    
    Args:
        df (DataFrame): DataFrame contenant les données
        features (list): Liste des colonnes à normaliser
        
    Returns:
        DataFrame: DataFrame avec les caractéristiques normalisées
    """
    df_copy = df.copy()
    
    for feature in features:
        if feature in df.columns:
            # Remplacer les valeurs manquantes par la médiane
            df_copy[feature] = df_copy[feature].fillna(df_copy[feature].median())
            
            # Normaliser entre 0 et 1
            min_val = df_copy[feature].min()
            max_val = df_copy[feature].max()
            
            if max_val > min_val:
                df_copy[feature] = (df_copy[feature] - min_val) / (max_val - min_val)
            else:
                df_copy[feature] = 0.5  # Valeur par défaut si min = max
    
    return df_copy

def generate_time_periods(start_date=None, end_date=None, n_periods=10):
    """
    Génère une liste de périodes temporelles
    
    Args:
        start_date (str, optional): Date de début au format 'YYYY-MM-DD'
        end_date (str, optional): Date de fin au format 'YYYY-MM-DD'
        n_periods (int): Nombre de périodes à générer
        
    Returns:
        list: Liste de tuples (date_début, date_fin)
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    delta = (end - start) / n_periods
    
    periods = []
    for i in range(n_periods):
        period_start = start + i * delta
        period_end = start + (i + 1) * delta
        periods.append((period_start.strftime('%Y-%m-%d'), period_end.strftime('%Y-%m-%d')))
    
    return periods

def generate_user_profile(user_id, tracks_df, n_favorites=10):
    """
    Génère un profil utilisateur fictif
    
    Args:
        user_id (str): ID de l'utilisateur
        tracks_df (DataFrame): DataFrame contenant les pistes
        n_favorites (int): Nombre de pistes favorites à générer
        
    Returns:
        dict: Profil utilisateur
    """
    # Sélectionner des pistes aléatoires comme favorites
    if len(tracks_df) < n_favorites:
        n_favorites = len(tracks_df)
    
    favorite_indices = np.random.choice(len(tracks_df), n_favorites, replace=False)
    favorite_tracks = tracks_df.iloc[favorite_indices]
    
    # Extraire les caractéristiques audio moyennes
    audio_features = ['danceability', 'energy', 'valence', 'acousticness', 'instrumentalness', 'tempo']
    feature_means = {}
    
    for feature in audio_features:
        if feature in favorite_tracks.columns:
            feature_means[feature] = float(favorite_tracks[feature].mean())
    
    # Extraire les tags les plus courants
    tags = []
    if 'tags' in favorite_tracks.columns:
        tag_counts = {}
        for _, track in favorite_tracks.iterrows():
            if isinstance(track['tags'], list):
                for tag in track['tags']:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Prendre les 5 tags les plus courants
        tags = [tag for tag, _ in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]]
    
    # Déterminer l'humeur préférée
    mood = "neutral"
    if 'mood' in favorite_tracks.columns:
        mood_counts = favorite_tracks['mood'].value_counts()
        if not mood_counts.empty:
            mood = mood_counts.index[0]
    
    # Créer le profil utilisateur
    user_profile = {
        'user_id': user_id,
        'favorite_tracks': favorite_tracks['id'].tolist(),
        'audio_preferences': feature_means,
        'tags': tags,
        'mood': mood,
        'created_at': datetime.now().isoformat()
    }
    
    return user_profile

def calculate_similarity(features1, features2, weights=None):
    """
    Calcule la similarité entre deux ensembles de caractéristiques
    
    Args:
        features1 (dict): Premier ensemble de caractéristiques
        features2 (dict): Deuxième ensemble de caractéristiques
        weights (dict, optional): Poids pour chaque caractéristique
        
    Returns:
        float: Score de similarité entre 0 et 1
    """
    if not features1 or not features2:
        return 0.0
    
    # Trouver les caractéristiques communes
    common_features = set(features1.keys()) & set(features2.keys())
    
    if not common_features:
        return 0.0
    
    # Calculer la distance euclidienne pondérée
    squared_diff_sum = 0.0
    weight_sum = 0.0
    
    for feature in common_features:
        weight = weights.get(feature, 1.0) if weights else 1.0
        squared_diff = (features1[feature] - features2[feature]) ** 2
        squared_diff_sum += weight * squared_diff
        weight_sum += weight
    
    if weight_sum == 0:
        return 0.0
    
    # Normaliser par la somme des poids
    distance = (squared_diff_sum / weight_sum) ** 0.5
    
    # Convertir la distance en similarité (1 = identique, 0 = complètement différent)
    similarity = 1.0 - min(distance, 1.0)
    
    return similarity

if __name__ == "__main__":
    # Test des fonctions
    print("Génération de périodes temporelles:")
    periods = generate_time_periods(n_periods=5)
    for start, end in periods:
        print(f"  {start} à {end}")
    
    print("\nCalcul de similarité:")
    features1 = {'danceability': 0.8, 'energy': 0.6, 'valence': 0.7}
    features2 = {'danceability': 0.7, 'energy': 0.5, 'valence': 0.8}
    similarity = calculate_similarity(features1, features2)
    print(f"  Similarité: {similarity:.4f}")
    
    # Test avec des poids
    weights = {'danceability': 2.0, 'energy': 1.0, 'valence': 3.0}
    weighted_similarity = calculate_similarity(features1, features2, weights)
    print(f"  Similarité pondérée: {weighted_similarity:.4f}")