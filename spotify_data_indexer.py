import pandas as pd
import numpy as np
import json
import os
from elasticsearch import Elasticsearch, helpers
import kagglehub
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Elasticsearch connection settings
ES_HOST = "localhost"  # Change to your Elasticsearch host
ES_PORT = 9200         # Change to your Elasticsearch port
ES_INDEX_TRACKS = "spotify_tracks"
ES_INDEX_ARTISTS = "spotify_artists"
ES_INDEX_USER_INTERACTIONS = "spotify_user_interactions"

def connect_to_elasticsearch():
    """Connect to Elasticsearch"""
    try:
        es = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT, 'scheme': 'http'}])
        if es.ping():
            logger.info("Connected to Elasticsearch")
            return es
        else:
            logger.error("Could not connect to Elasticsearch")
            return None
    except Exception as e:
        logger.error(f"Error connecting to Elasticsearch: {e}")
        return None

def create_indices(es):
    """Create Elasticsearch indices with appropriate mappings"""
    # Tracks index mapping
    tracks_mapping = {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "name": {"type": "text", "analyzer": "standard", "fields": {"keyword": {"type": "keyword"}}},
                "popularity": {"type": "float"},
                "duration_ms": {"type": "integer"},
                "explicit": {"type": "boolean"},
                "artists": {"type": "keyword"},
                "artist_names": {"type": "text", "analyzer": "standard"},
                "id_artists": {"type": "keyword"},
                "release_date": {"type": "date", "format": "yyyy-MM-dd||yyyy-MM||yyyy"},
                "danceability": {"type": "float"},
                "energy": {"type": "float"},
                "key": {"type": "integer"},
                "loudness": {"type": "float"},
                "mode": {"type": "integer"},
                "speechiness": {"type": "float"},
                "acousticness": {"type": "float"},
                "instrumentalness": {"type": "float"},
                "liveness": {"type": "float"},
                "valence": {"type": "float"},
                "tempo": {"type": "float"},
                "time_signature": {"type": "integer"},
                "track_genre": {"type": "keyword"},
                "tags": {"type": "keyword"},
                "mood": {"type": "keyword"}
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    
    # Artists index mapping
    artists_mapping = {
        "mappings": {
            "properties": {
                "id": {"type": "keyword"},
                "name": {"type": "text", "analyzer": "standard", "fields": {"keyword": {"type": "keyword"}}},
                "genres": {"type": "keyword"},
                "popularity": {"type": "float"},
                "followers": {"type": "integer"}
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    
    # User interactions index mapping
    interactions_mapping = {
        "mappings": {
            "properties": {
                "user_id": {"type": "keyword"},
                "track_id": {"type": "keyword"},
                "play_count": {"type": "integer"},
                "timestamp": {"type": "date"},
                "liked": {"type": "boolean"}
            }
        },
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        }
    }
    
    # Create indices if they don't exist
    if not es.indices.exists(index=ES_INDEX_TRACKS):
        es.indices.create(index=ES_INDEX_TRACKS, body=tracks_mapping)
        logger.info(f"Created index: {ES_INDEX_TRACKS}")
    
    if not es.indices.exists(index=ES_INDEX_ARTISTS):
        es.indices.create(index=ES_INDEX_ARTISTS, body=artists_mapping)
        logger.info(f"Created index: {ES_INDEX_ARTISTS}")
    
    if not es.indices.exists(index=ES_INDEX_USER_INTERACTIONS):
        es.indices.create(index=ES_INDEX_USER_INTERACTIONS, body=interactions_mapping)
        logger.info(f"Created index: {ES_INDEX_USER_INTERACTIONS}")

def download_kaggle_dataset():
    """Download the Spotify dataset from Kaggle"""
    logger.info("Downloading Spotify dataset from Kaggle...")
    try:
        path = kagglehub.dataset_download("maharshipandya/-spotify-tracks-dataset")
        logger.info(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return None

def load_and_transform_data(dataset_path):
    """Load and transform the Spotify dataset"""
    logger.info("Loading and transforming data...")
    
    # Load tracks data
    tracks_path = os.path.join(dataset_path, "tracks.csv")
    if not os.path.exists(tracks_path):
        logger.error(f"Tracks file not found at {tracks_path}")
        return None, None
    
    tracks_df = pd.read_csv(tracks_path)
    logger.info(f"Loaded {len(tracks_df)} tracks")
    
    # Load artists data if available
    artists_df = None
    artists_path = os.path.join(dataset_path, "artists.csv")
    if os.path.exists(artists_path):
        artists_df = pd.read_csv(artists_path)
        logger.info(f"Loaded {len(artists_df)} artists")
    
    # Transform tracks data
    tracks_df = transform_tracks_data(tracks_df)
    
    # Transform artists data if available
    if artists_df is not None:
        artists_df = transform_artists_data(artists_df)
    
    return tracks_df, artists_df

def transform_tracks_data(df):
    """Transform tracks data for Elasticsearch"""
    logger.info("Transforming tracks data...")
    
    # Handle missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    
    # Convert release_date to proper format if it exists
    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce').dt.strftime('%Y-%m-%d')
        df['release_date'] = df['release_date'].fillna('1970-01-01')
    
    # Generate tags based on audio features
    df = generate_tags_and_mood(df)
    
    return df

def transform_artists_data(df):
    """Transform artists data for Elasticsearch"""
    logger.info("Transforming artists data...")
    
    # Handle missing values
    if 'followers' in df.columns:
        df['followers'] = df['followers'].fillna(0).astype(int)
    
    if 'popularity' in df.columns:
        df['popularity'] = df['popularity'].fillna(0).astype(float)
    
    # Convert genres from string to list if needed
    if 'genres' in df.columns and df['genres'].dtype == 'object':
        df['genres'] = df['genres'].apply(lambda x: 
            json.loads(x.replace("'", "\"")) if isinstance(x, str) else [] if pd.isna(x) else x
        )
    
    return df

def generate_tags_and_mood(df):
    """Generate tags and mood based on audio features"""
    # Initialize tags and mood columns
    df['tags'] = df.apply(lambda _: [], axis=1)
    df['mood'] = ""
    
    # Generate tags based on audio features
    if all(col in df.columns for col in ['danceability', 'energy', 'valence', 'tempo', 'acousticness']):
        # Danceability tags
        df.loc[df['danceability'] >= 0.7, 'tags'] = df.loc[df['danceability'] >= 0.7, 'tags'].apply(lambda x: x + ['danceable'])
        df.loc[df['danceability'] <= 0.3, 'tags'] = df.loc[df['danceability'] <= 0.3, 'tags'].apply(lambda x: x + ['not_danceable'])
        
        # Energy tags
        df.loc[df['energy'] >= 0.7, 'tags'] = df.loc[df['energy'] >= 0.7, 'tags'].apply(lambda x: x + ['energetic'])
        df.loc[df['energy'] <= 0.3, 'tags'] = df.loc[df['energy'] <= 0.3, 'tags'].apply(lambda x: x + ['calm'])
        
        # Tempo tags
        df.loc[df['tempo'] >= 120, 'tags'] = df.loc[df['tempo'] >= 120, 'tags'].apply(lambda x: x + ['fast'])
        df.loc[df['tempo'] <= 80, 'tags'] = df.loc[df['tempo'] <= 80, 'tags'].apply(lambda x: x + ['slow'])
        
        # Acousticness tags
        df.loc[df['acousticness'] >= 0.7, 'tags'] = df.loc[df['acousticness'] >= 0.7, 'tags'].apply(lambda x: x + ['acoustic'])
        df.loc[df['acousticness'] <= 0.3, 'tags'] = df.loc[df['acousticness'] <= 0.3, 'tags'].apply(lambda x: x + ['electronic'])
        
        # Mood determination based on valence and energy
        # High valence and high energy: happy/energetic
        df.loc[(df['valence'] >= 0.6) & (df['energy'] >= 0.6), 'mood'] = 'happy'
        
        # High valence and low energy: peaceful/relaxed
        df.loc[(df['valence'] >= 0.6) & (df['energy'] <= 0.4), 'mood'] = 'relaxed'
        
        # Low valence and high energy: angry/intense
        df.loc[(df['valence'] <= 0.4) & (df['energy'] >= 0.6), 'mood'] = 'intense'
        
        # Low valence and low energy: sad/depressed
        df.loc[(df['valence'] <= 0.4) & (df['energy'] <= 0.4), 'mood'] = 'sad'
        
        # Everything else: neutral
        df.loc[df['mood'] == "", 'mood'] = 'neutral'
    
    # Add genre tags if available
    if 'track_genre' in df.columns:
        df['tags'] = df.apply(lambda row: row['tags'] + [row['track_genre']] if pd.notna(row['track_genre']) else row['tags'], axis=1)
    
    return df

def generate_mock_user_interactions(tracks_df, n_users=1000, n_interactions=50000):
    """Generate mock user listening history"""
    logger.info("Generating mock user interactions...")
    
    # Get track IDs
    track_ids = tracks_df['id'].tolist()
    
    # Generate random interactions
    user_ids = [f"user_{i}" for i in range(n_users)]
    
    interactions = []
    for _ in tqdm(range(n_interactions), desc="Generating interactions"):
        user_id = np.random.choice(user_ids)
        track_id = np.random.choice(track_ids)
        play_count = np.random.randint(1, 20)
        timestamp = pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 90))
        liked = np.random.choice([True, False], p=[0.2, 0.8])  # 20% chance of liking
        
        interactions.append({
            'user_id': user_id,
            'track_id': track_id,
            'play_count': play_count,
            'timestamp': timestamp.isoformat(),
            'liked': liked
        })
    
    return pd.DataFrame(interactions)

def index_to_elasticsearch(es, tracks_df, artists_df=None, interactions_df=None):
    """Index data to Elasticsearch"""
    # Index tracks
    if tracks_df is not None:
        logger.info(f"Indexing {len(tracks_df)} tracks to Elasticsearch...")
        
        # Convert DataFrame to list of dicts for bulk indexing
        tracks_actions = []
        for _, track in tqdm(tracks_df.iterrows(), total=len(tracks_df), desc="Preparing tracks"):
            track_dict = track.to_dict()
            
            # Ensure all fields are JSON serializable
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
        
        # Bulk index tracks
        if tracks_actions:
            success, failed = 0, 0
            for success_count, failed_items in tqdm(
                helpers.streaming_bulk(es, tracks_actions, chunk_size=500, max_retries=3),
                total=len(tracks_actions),
                desc="Indexing tracks"
            ):
                if success_count:
                    success += 1
                else:
                    failed += 1
            
            logger.info(f"Indexed {success} tracks successfully, {failed} failed")
    
    # Index artists
    if artists_df is not None:
        logger.info(f"Indexing {len(artists_df)} artists to Elasticsearch...")
        
        # Convert DataFrame to list of dicts for bulk indexing
        artists_actions = []
        for _, artist in tqdm(artists_df.iterrows(), total=len(artists_df), desc="Preparing artists"):
            artist_dict = artist.to_dict()
            
            # Ensure all fields are JSON serializable
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
        
        # Bulk index artists
        if artists_actions:
            success, failed = 0, 0
            for success_count, failed_items in tqdm(
                helpers.streaming_bulk(es, artists_actions, chunk_size=500, max_retries=3),
                total=len(artists_actions),
                desc="Indexing artists"
            ):
                if success_count:
                    success += 1
                else:
                    failed += 1
            
            logger.info(f"Indexed {success} artists successfully, {failed} failed")
    
    # Index user interactions
    if interactions_df is not None:
        logger.info(f"Indexing {len(interactions_df)} user interactions to Elasticsearch...")
        
        # Convert DataFrame to list of dicts for bulk indexing
        interactions_actions = []
        for idx, interaction in tqdm(interactions_df.iterrows(), total=len(interactions_df), desc="Preparing interactions"):
            interaction_dict = interaction.to_dict()
            
            # Create a unique ID for each interaction
            interaction_id = f"{interaction_dict['user_id']}_{interaction_dict['track_id']}_{idx}"
            
            action = {
                "_index": ES_INDEX_USER_INTERACTIONS,
                "_id": interaction_id,
                "_source": interaction_dict
            }
            interactions_actions.append(action)
        
        # Bulk index interactions
        if interactions_actions:
            success, failed = 0, 0
            for success_count, failed_items in tqdm(
                helpers.streaming_bulk(es, interactions_actions, chunk_size=500, max_retries=3),
                total=len(interactions_actions),
                desc="Indexing interactions"
            ):
                if success_count:
                    success += 1
                else:
                    failed += 1
            
            logger.info(f"Indexed {success} interactions successfully, {failed} failed")

def main():
    """Main function to download, transform and index Spotify data"""
    # Connect to Elasticsearch
    es = connect_to_elasticsearch()
    if not es:
        return
    
    # Create indices
    create_indices(es)
    
    # Download dataset
    dataset_path = download_kaggle_dataset()
    if not dataset_path:
        return
    
    # Load and transform data
    tracks_df, artists_df = load_and_transform_data(dataset_path)
    if tracks_df is None:
        return
    
    # Generate mock user interactions
    interactions_df = generate_mock_user_interactions(tracks_df)
    
    # Index data to Elasticsearch
    index_to_elasticsearch(es, tracks_df, artists_df, interactions_df)
    
    logger.info("Data indexing complete!")

if __name__ == "__main__":
    main()