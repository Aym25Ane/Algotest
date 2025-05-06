"""
Module pour indexer les données Spotify dans Elasticsearch
"""
import logging
import pandas as pd
import sys
import time
from elasticsearch import helpers
from tqdm import tqdm

# Import configuration
sys.path.append('..')
from config import ES_INDEX_TRACKS, ES_INDEX_ARTISTS, ES_INDEX_USER_INTERACTIONS
from utils.elasticsearch_client import get_elasticsearch_client, create_indices

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def index_tracks(es, tracks_df, chunk_size=500):
    """
    Indexe les données des pistes dans Elasticsearch
    
    Args:
        es: Client Elasticsearch
        tracks_df (DataFrame): DataFrame pandas contenant les données des pistes
        chunk_size (int): Taille des lots pour l'indexation par lots
        
    Returns:
        tuple: (success_count, failed_count) Nombre de documents indexés avec succès et échoués
    """
    if es is None:
        logger.error("Client Elasticsearch non disponible")
        return 0, 0
    
    if tracks_df is None or len(tracks_df) == 0:
        logger.error("Aucune donnée de piste à indexer")
        return 0, 0
    
    logger.info(f"Indexation de {len(tracks_df)} pistes dans Elasticsearch...")
    
    # Convertir DataFrame en liste de dictionnaires pour l'indexation par lots
    tracks_actions = []
    for _, track in tqdm(tracks_df.iterrows(), total=len(tracks_df), desc="Préparation des pistes"):
        track_dict = track.to_dict()
        
        # S'assurer que tous les champs sont sérialisables en JSON
        for key, value in track_dict.items():
            if pd.isna(value):
                if key in ['tags']:
                    track_dict[key] = []
                elif key in ['mood']:
                    track_dict[key] = 'neutral'
                else:
                    track_dict[key] = None
        
        action = {
            "_index": ES_INDEX_TRACKS,
            "_id": track_dict['id'],
            "_source": track_dict
        }
        tracks_actions.append(action)
    
    # Indexation par lots
    if tracks_actions:
        success, failed = 0, 0
        for success_count, failed_items in tqdm(
            helpers.streaming_bulk(es, tracks_actions, chunk_size=chunk_size, max_retries=3),
            total=len(tracks_actions),
            desc="Indexation des pistes"
        ):
            if success_count:
                success += 1
            else:
                failed += 1
        
        logger.info(f"Indexé {success} pistes avec succès, {failed} échecs")
        return success, failed
    
    return 0, 0

def index_artists(es, artists_df, chunk_size=500):
    """
    Indexe les données des artistes dans Elasticsearch
    
    Args:
        es: Client Elasticsearch
        artists_df (DataFrame): DataFrame pandas contenant les données des artistes
        chunk_size (int): Taille des lots pour l'indexation par lots
        
    Returns:
        tuple: (success_count, failed_count) Nombre de documents indexés avec succès et échoués
    """
    if es is None:
        logger.error("Client Elasticsearch non disponible")
        return 0, 0
    
    if artists_df is None or len(artists_df) == 0:
        logger.warning("Aucune donnée d'artiste à indexer")
        return 0, 0
    
    logger.info(f"Indexation de {len(artists_df)} artistes dans Elasticsearch...")
    
    # Convertir DataFrame en liste de dictionnaires pour l'indexation par lots
    artists_actions = []
    for _, artist in tqdm(artists_df.iterrows(), total=len(artists_df), desc="Préparation des artistes"):
        artist_dict = artist.to_dict()
        
        # S'assurer que tous les champs sont sérialisables en JSON
        for key, value in artist_dict.items():
            if pd.isna(value):
                if key in ['genres']:
                    artist_dict[key] = []
                else:
                    artist_dict[key] = None
        
        action = {
            "_index": ES_INDEX_ARTISTS,
            "_id": artist_dict['id'],
            "_source": artist_dict
        }
        artists_actions.append(action)
    
    # Indexation par lots
    if artists_actions:
        success, failed = 0, 0
        for success_count, failed_items in tqdm(
            helpers.streaming_bulk(es, artists_actions, chunk_size=chunk_size, max_retries=3),
            total=len(artists_actions),
            desc="Indexation des artistes"
        ):
            if success_count:
                success += 1
            else:
                failed += 1
        
        logger.info(f"Indexé {success} artistes avec succès, {failed} échecs")
        return success, failed
    
    return 0, 0

def index_user_interactions(es, interactions_df, chunk_size=500):
    """
    Indexe les interactions utilisateur dans Elasticsearch
    
    Args:
        es: Client Elasticsearch
        interactions_df (DataFrame): DataFrame pandas contenant les interactions utilisateur
        chunk_size (int): Taille des lots pour l'indexation par lots
        
    Returns:
        tuple: (success_count, failed_count) Nombre de documents indexés avec succès et échoués
    """
    if es is None:
        logger.error("Client Elasticsearch non disponible")
        return 0, 0
    
    if interactions_df is None or len(interactions_df) == 0:
        logger.error("Aucune interaction utilisateur à indexer")
        return 0, 0
    
    logger.info(f"Indexation de {len(interactions_df)} interactions utilisateur dans Elasticsearch...")
    
    # Convertir DataFrame en liste de dictionnaires pour l'indexation par lots
    interactions_actions = []
    for idx, interaction in tqdm(interactions_df.iterrows(), total=len(interactions_df), desc="Préparation des interactions"):
        interaction_dict = interaction.to_dict()
        
        # Créer un ID unique pour chaque interaction
        interaction_id = f"{interaction_dict['user_id']}_{interaction_dict['track_id']}_{idx}"
        
        action = {
            "_index": ES_INDEX_USER_INTERACTIONS,
            "_id": interaction_id,
            "_source": interaction_dict
        }
        interactions_actions.append(action)
    
    # Indexation par lots
    if interactions_actions:
        success, failed = 0, 0
        for success_count, failed_items in tqdm(
            helpers.streaming_bulk(es, interactions_actions, chunk_size=chunk_size, max_retries=3),
            total=len(interactions_actions),
            desc="Indexation des interactions"
        ):
            if success_count:
                success += 1
            else:
                failed += 1
        
        logger.info(f"Indexé {success} interactions avec succès, {failed} échecs")
        return success, failed
    
    return 0, 0

def run_indexing_pipeline(tracks_df, artists_df=None, interactions_df=None):
    """
    Exécute le pipeline complet d'indexation des données dans Elasticsearch
    
    Args:
        tracks_df (DataFrame): DataFrame pandas contenant les données des pistes
        artists_df (DataFrame, optional): DataFrame pandas contenant les données des artistes
        interactions_df (DataFrame, optional): DataFrame pandas contenant les interactions utilisateur
        
    Returns:
        bool: True si l'indexation a réussi, False sinon
    """
    # Obtenir le client Elasticsearch
    es = get_elasticsearch_client()
    if es is None:
        return False
    
    # Créer les indices s'ils n'existent pas
    if not create_indices(es):
        return False
    
    # Indexer les pistes
    tracks_success, tracks_failed = index_tracks(es, tracks_df)
    
    # Indexer les artistes si disponibles
    artists_success, artists_failed = 0, 0
    if artists_df is not None:
        artists_success, artists_failed = index_artists(es, artists_df)
    
    # Indexer les interactions utilisateur si disponibles
    interactions_success, interactions_failed = 0, 0
    if interactions_df is not None:
        interactions_success, interactions_failed = index_user_interactions(es, interactions_df)
    
    # Résumé de l'indexation
    logger.info("Résumé de l'indexation:")
    logger.info(f"Pistes: {tracks_success} succès, {tracks_failed} échecs")
    logger.info(f"Artistes: {artists_success} succès, {artists_failed} échecs")
    logger.info(f"Interactions: {interactions_success} succès, {interactions_failed} échecs")
    
    # Attendre que l'indexation soit complète
    logger.info("Attente de la fin de l'indexation...")
    time.sleep(2)
    
    # Rafraîchir les indices
    es.indices.refresh(index=[ES_INDEX_TRACKS, ES_INDEX_ARTISTS, ES_INDEX_USER_INTERACTIONS])
    
    return True

if __name__ == "__main__":
    # Test du module
    from data_downloader import download_kaggle_dataset, verify_dataset
    from data_transformer import load_dataset, transform_tracks_data, transform_artists_data, generate_mock_user_interactions
    
    # Télécharger et vérifier le dataset
    dataset_path = download_kaggle_dataset()
    if dataset_path and verify_dataset(dataset_path):
        # Charger et transformer les données
        tracks_df, artists_df = load_dataset(dataset_path)
        
        if tracks_df is not None:
            tracks_df = transform_tracks_data(tracks_df)
            
            if artists_df is not None:
                artists_df = transform_artists_data(artists_df)
            
            # Générer des interactions utilisateur
            interactions_df = generate_mock_user_interactions(tracks_df, n_users=100, n_interactions=1000)
            
            # Exécuter le pipeline d'indexation
            run_indexing_pipeline(tracks_df, artists_df, interactions_df)