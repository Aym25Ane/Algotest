"""
Module pour gérer le dataset Spotify (version pour téléchargement manuel)
"""
import os
import logging
import sys
import shutil

# Import configuration
sys.path.append('..')
from config import DATASET_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_kaggle_dataset(force_download=False):
    """
    Vérifie la présence du dataset Spotify téléchargé manuellement

    Args:
        force_download (bool): Ignoré dans cette version (pour compatibilité)

    Returns:
        str: Chemin vers le dossier contenant les données
    """
    # Créer le dossier de destination s'il n'existe pas
    os.makedirs(DATASET_PATH, exist_ok=True)

    # Vérifier si les fichiers existent déjà
    tracks_file = os.path.join(DATASET_PATH, "tracks.csv")
    if os.path.exists(tracks_file):
        logger.info(f"Le fichier {tracks_file} existe. Utilisation des fichiers existants.")
        return DATASET_PATH
    else:
        logger.warning(f"Le fichier {tracks_file} n'existe pas.")
        logger.warning("Veuillez télécharger manuellement le dataset depuis Kaggle:")
        logger.warning("1. Allez sur https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset")
        logger.warning(f"2. Téléchargez le dataset et décompressez-le dans le dossier {DATASET_PATH}")
        logger.warning("3. Assurez-vous que le fichier tracks.csv est présent dans ce dossier")
        return None

def verify_dataset(dataset_path):
    """
    Vérifie que les fichiers nécessaires sont présents dans le dataset

    Args:
        dataset_path (str): Chemin vers le dossier contenant les données

    Returns:
        bool: True si les fichiers nécessaires sont présents, False sinon
    """
    if dataset_path is None:
        return False

    required_files = ["tracks.csv"]
    optional_files = ["artists.csv"]

    # Vérifier les fichiers requis
    for file in required_files:
        file_path = os.path.join(dataset_path, file)
        if not os.path.exists(file_path):
            logger.error(f"Fichier requis manquant: {file}")
            logger.error(f"Veuillez vous assurer que {file} est présent dans {dataset_path}")
            return False

    # Vérifier les fichiers optionnels
    for file in optional_files:
        file_path = os.path.join(dataset_path, file)
        if not os.path.exists(file_path):
            logger.warning(f"Fichier optionnel manquant: {file}")

    logger.info(f"Vérification du dataset réussie: {dataset_path}")
    return True

if __name__ == "__main__":
    # Test du module
    dataset_path = download_kaggle_dataset()
    if dataset_path:
        verify_dataset(dataset_path)
    else:
        logger.error("Dataset non disponible. Veuillez suivre les instructions pour le téléchargement manuel.")