"""
Point d'entrée principal pour le système de recommandation Spotify
"""
import logging
import argparse
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_data_pipeline():
    """Configure et exécute le pipeline de données"""
    from data_processing.data_downloader import download_kaggle_dataset, verify_dataset
    from data_processing.data_transformer import load_dataset, transform_tracks_data, transform_artists_data, generate_mock_user_interactions
    from data_processing.elasticsearch_indexer import run_indexing_pipeline
    
    # Télécharger le dataset
    dataset_path = download_kaggle_dataset()
    if not dataset_path or not verify_dataset(dataset_path):
        logger.error("Échec du téléchargement ou de la vérification du dataset")
        return False
    
    # Charger et transformer les données
    tracks_df, artists_df = load_dataset(dataset_path)
    
    if tracks_df is None:
        logger.error("Échec du chargement des données des pistes")
        return False
    
    tracks_df = transform_tracks_data(tracks_df)
    
    if artists_df is not None:
        artists_df = transform_artists_data(artists_df)
    
    # Générer des interactions utilisateur
    interactions_df = generate_mock_user_interactions(tracks_df, n_users=1000, n_interactions=50000)
    
    # Exécuter le pipeline d'indexation
    success = run_indexing_pipeline(tracks_df, artists_df, interactions_df)
    
    return success

def start_api_server():
    """Démarre le serveur API Flask"""
    from api.recommendation_service import run_api_server
    
    # Démarrer le serveur API
    run_api_server()

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Système de recommandation Spotify avec Elasticsearch")
    parser.add_argument('--setup', action='store_true', help="Configurer le pipeline de données")
    parser.add_argument('--api', action='store_true', help="Démarrer le serveur API")
    
    args = parser.parse_args()
    
    if args.setup:
        logger.info("Configuration du pipeline de données...")
        if setup_data_pipeline():
            logger.info("Pipeline de données configuré avec succès")
        else:
            logger.error("Échec de la configuration du pipeline de données")
            return 1
    
    if args.api:
        logger.info("Démarrage du serveur API...")
        start_api_server()
    
    # Si aucune option n'est spécifiée, afficher l'aide
    if not args.setup and not args.api:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())