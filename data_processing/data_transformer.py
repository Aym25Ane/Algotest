"""
Module pour transformer les données Spotify avant indexation dans Elasticsearch
"""
import pandas as pd
import numpy as np
import json
import logging
import sys
import os
from sklearn.preprocessing import MinMaxScaler

# Import configuration
sys.path.append('..')
from config import AUDIO_FEATURES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(dataset_path):
    """
    Charge les fichiers CSV du dataset
    
    Args:
        dataset_path (str): Chemin vers le dossier contenant les données
        
    Returns:
        tuple: (tracks_df, artists_df) DataFrames pandas contenant les données
    """
    tracks_path = os.path.join(dataset_path, "tracks.csv")
    artists_path = os.path.join(dataset_path, "artists.csv")
    
    # Charger les pistes
    try:
        tracks_df = pd.read_csv(tracks_path)
        logger.info(f"Chargé {len(tracks_df)} pistes depuis {tracks_path}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement des pistes: {e}")
        return None, None
    
    # Charger les artistes si disponibles
    artists_df = None
    if os.path.exists(artists_path):
        try:
            artists_df = pd.read_csv(artists_path)
            logger.info(f"Chargé {len(artists_df)} artistes depuis {artists_path}")
        except Exception as e:
            logger.warning(f"Erreur lors du chargement des artistes: {e}")
    
    return tracks_df, artists_df

def transform_tracks_data(df):
    """
    Transforme les données des pistes pour l'indexation dans Elasticsearch
    
    Args:
        df (DataFrame): DataFrame pandas contenant les données des pistes
        
    Returns:
        DataFrame: DataFrame transformé
    """
    logger.info("Transformation des données des pistes...")
    
    # Copier le DataFrame pour éviter de modifier l'original
    df = df.copy()
    
    # Gérer les valeurs manquantes dans les colonnes numériques
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Convertir release_date au format approprié si elle existe
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        df['release_date'] = df['release_date'].fillna('1970-01-01')
    
    # Normaliser les caractéristiques audio
    audio_features_present = [f for f in AUDIO_FEATURES if f in df.columns]
    if audio_features_present:
        scaler = MinMaxScaler()
        df[audio_features_present] = scaler.fit_transform(df[audio_features_present])
        logger.info(f"Caractéristiques audio normalisées: {audio_features_present}")
    
    # Générer des tags et l'humeur basés sur les caractéristiques audio
    df = generate_tags_and_mood(df)
    
    # Assurer que les colonnes id et name existent
    if 'id' not in df.columns:
        logger.error("La colonne 'id' est manquante dans les données des pistes")
        if 'track_id' in df.columns:
            df['id'] = df['track_id']
            logger.info("Utilisé 'track_id' comme 'id'")
        else:
            df['id'] = df.index.astype(str)
            logger.warning("Généré des IDs à partir de l'index")
    
    if 'name' not in df.columns:
        logger.error("La colonne 'name' est manquante dans les données des pistes")
        if 'track_name' in df.columns:
            df['name'] = df['track_name']
            logger.info("Utilisé 'track_name' comme 'name'")
        else:
            df['name'] = "Unknown Track"
            logger.warning("Utilisé 'Unknown Track' comme nom par défaut")
    
    # Assurer que les colonnes artists et artist_names existent
    if 'artists' not in df.columns and 'id_artists' in df.columns:
        df['artists'] = df['id_artists']
        logger.info("Utilisé 'id_artists' comme 'artists'")
    
    if 'artist_names' not in df.columns and 'artists' in df.columns:
        # Essayer de convertir la chaîne en liste si c'est une chaîne
        if df['artists'].dtype == 'object':
            df['artist_names'] = df['artists']
        else:
            df['artist_names'] = "Unknown Artist"
            logger.warning("Utilisé 'Unknown Artist' comme nom d'artiste par défaut")
    
    logger.info(f"Transformation des données des pistes terminée: {len(df)} pistes")
    return df

def transform_artists_data(df):
    """
    Transforme les données des artistes pour l'indexation dans Elasticsearch
    
    Args:
        df (DataFrame): DataFrame pandas contenant les données des artistes
        
    Returns:
        DataFrame: DataFrame transformé
    """
    if df is None:
        return None
    
    logger.info("Transformation des données des artistes...")
    
    # Copier le DataFrame pour éviter de modifier l'original
    df = df.copy()
    
    # Gérer les valeurs manquantes
    if 'followers' in df.columns:
        df['followers'] = df['followers'].fillna(0).astype(int)
    
    if 'popularity' in df.columns:
        df['popularity'] = df['popularity'].fillna(0).astype(float)
    
    # Convertir genres de chaîne à liste si nécessaire
    if 'genres' in df.columns and df['genres'].dtype == 'object':
        df['genres'] = df['genres'].apply(lambda x: 
            json.loads(x.replace("'", "\"")) if isinstance(x, str) else [] if pd.isna(x) else x
        )
    
    # Assurer que les colonnes id et name existent
    if 'id' not in df.columns:
        logger.error("La colonne 'id' est manquante dans les données des artistes")
        if 'artist_id' in df.columns:
            df['id'] = df['artist_id']
            logger.info("Utilisé 'artist_id' comme 'id'")
        else:
            df['id'] = df.index.astype(str)
            logger.warning("Généré des IDs à partir de l'index")
    
    if 'name' not in df.columns:
        logger.error("La colonne 'name' est manquante dans les données des artistes")
        if 'artist_name' in df.columns:
            df['name'] = df['artist_name']
            logger.info("Utilisé 'artist_name' comme 'name'")
        else:
            df['name'] = "Unknown Artist"
            logger.warning("Utilisé 'Unknown Artist' comme nom par défaut")
    
    logger.info(f"Transformation des données des artistes terminée: {len(df)} artistes")
    return df

def generate_tags_and_mood(df):
    """
    Génère des tags et l'humeur basés sur les caractéristiques audio
    
    Args:
        df (DataFrame): DataFrame pandas contenant les données des pistes
        
    Returns:
        DataFrame: DataFrame avec les colonnes tags et mood ajoutées
    """
    # Initialiser les colonnes tags et mood
    df['tags'] = df.apply(lambda _: [], axis=1)
    df['mood'] = ""
    
    # Générer des tags basés sur les caractéristiques audio
    if all(col in df.columns for col in ['danceability', 'energy', 'valence', 'tempo', 'acousticness']):
        # Tags de danceability
        df.loc[df['danceability'] >= 0.7, 'tags'] = df.loc[df['danceability'] >= 0.7, 'tags'].apply(lambda x: x + ['danceable'])
        df.loc[df['danceability'] <= 0.3, 'tags'] = df.loc[df['danceability'] <= 0.3, 'tags'].apply(lambda x: x + ['not_danceable'])
        
        # Tags d'énergie
        df.loc[df['energy'] >= 0.7, 'tags'] = df.loc[df['energy'] >= 0.7, 'tags'].apply(lambda x: x + ['energetic'])
        df.loc[df['energy'] <= 0.3, 'tags'] = df.loc[df['energy'] <= 0.3, 'tags'].apply(lambda x: x + ['calm'])
        
        # Tags de tempo
        df.loc[df['tempo'] >= 120, 'tags'] = df.loc[df['tempo'] >= 120, 'tags'].apply(lambda x: x + ['fast'])
        df.loc[df['tempo'] <= 80, 'tags'] = df.loc[df['tempo'] <= 80, 'tags'].apply(lambda x: x + ['slow'])
        
        # Tags d'acousticness
        df.loc[df['acousticness'] >= 0.7, 'tags'] = df.loc[df['acousticness'] >= 0.7, 'tags'].apply(lambda x: x + ['acoustic'])
        df.loc[df['acousticness'] <= 0.3, 'tags'] = df.loc[df['acousticness'] <= 0.3, 'tags'].apply(lambda x: x + ['electronic'])
        
        # Détermination de l'humeur basée sur valence et énergie
        # Valence élevée et énergie élevée: happy/energetic
        df.loc[(df['valence'] >= 0.6) & (df['energy'] >= 0.6), 'mood'] = 'happy'
        
        # Valence élevée et énergie basse: peaceful/relaxed
        df.loc[(df['valence'] >= 0.6) & (df['energy'] <= 0.4), 'mood'] = 'relaxed'
        
        # Valence basse et énergie élevée: angry/intense
        df.loc[(df['valence'] <= 0.4) & (df['energy'] >= 0.6), 'mood'] = 'intense'
        
        # Valence basse et énergie basse: sad/depressed
        df.loc[(df['valence'] <= 0.4) & (df['energy'] <= 0.4), 'mood'] = 'sad'
        
        # Tout le reste: neutral
        df.loc[df['mood'] == "", 'mood'] = 'neutral'
    
    # Ajouter des tags de genre si disponibles
    if 'track_genre' in df.columns:
        df['tags'] = df.apply(lambda row: row['tags'] + [row['track_genre']] if pd.notna(row['track_genre']) else row['tags'], axis=1)
    
    logger.info("Tags et humeurs générés pour les pistes")
    return df

def generate_mock_user_interactions(tracks_df, n_users=1000, n_interactions=50000):
    """
    Génère des interactions utilisateur fictives pour les tests
    
    Args:
        tracks_df (DataFrame): DataFrame pandas contenant les données des pistes
        n_users (int): Nombre d'utilisateurs à générer
        n_interactions (int): Nombre d'interactions à générer
        
    Returns:
        DataFrame: DataFrame contenant les interactions utilisateur
    """
    logger.info(f"Génération de {n_interactions} interactions utilisateur fictives pour {n_users} utilisateurs...")
    
    # Obtenir les IDs des pistes
    track_ids = tracks_df['id'].tolist()
    
    # Générer des IDs utilisateur
    user_ids = [f"user_{i}" for i in range(n_users)]
    
    # Générer des interactions aléatoires
    interactions = []
    for _ in range(n_interactions):
        user_id = np.random.choice(user_ids)
        track_id = np.random.choice(track_ids)
        play_count = np.random.randint(1, 20)
        timestamp = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 90))
        liked = np.random.choice([True, False], p=[0.2, 0.8])  # 20% de chance d'aimer
        
        interactions.append({
            'user_id': user_id,
            'track_id': track_id,
            'play_count': play_count,
            'timestamp': timestamp.isoformat(),
            'liked': liked
        })
    
    interactions_df = pd.DataFrame(interactions)
    logger.info(f"Généré {len(interactions_df)} interactions utilisateur")
    return interactions_df

if __name__ == "__main__":
    # Test du module
    from data_downloader import download_kaggle_dataset, verify_dataset
    
    dataset_path = download_kaggle_dataset()
    if dataset_path and verify_dataset(dataset_path):
        tracks_df, artists_df = load_dataset(dataset_path)
        
        if tracks_df is not None:
            tracks_df = transform_tracks_data(tracks_df)
            print(f"Colonnes des pistes: {tracks_df.columns.tolist()}")
            print(f"Exemple de piste:\n{tracks_df.iloc[0]}")
        
        if artists_df is not None:
            artists_df = transform_artists_data(artists_df)
            print(f"Colonnes des artistes: {artists_df.columns.tolist()}")
            print(f"Exemple d'artiste:\n{artists_df.iloc[0]}")
        
        # Générer des interactions utilisateur
        if tracks_df is not None:
            interactions_df = generate_mock_user_interactions(tracks_df, n_users=10, n_interactions=100)
            print(f"Colonnes des interactions: {interactions_df.columns.tolist()}")
            print(f"Exemple d'interaction:\n{interactions_df.iloc[0]}")