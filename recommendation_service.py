from flask import Flask, request, jsonify
from elasticsearch import Elasticsearch
import numpy as np
import pandas as pd
from collections import defaultdict
import logging
import json
import os
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Elasticsearch connection settings
ES_HOST = os.environ.get("ES_HOST", "localhost")
ES_PORT = int(os.environ.get("ES_PORT", 9200))
ES_INDEX_TRACKS = "spotify_tracks"
ES_INDEX_ARTISTS = "spotify_artists"
ES_INDEX_USER_INTERACTIONS = "spotify_user_interactions"

# Connect to Elasticsearch
es = Elasticsearch([{'host': ES_HOST, 'port': ES_PORT, 'scheme': 'http'}])

# Check Elasticsearch connection
@app.before_first_request
def check_elasticsearch():
    if not es.ping():
        logger.error("Could not connect to Elasticsearch")
    else:
        logger.info("Connected to Elasticsearch")
        # Check if indices exist
        for index in [ES_INDEX_TRACKS, ES_INDEX_ARTISTS, ES_INDEX_USER_INTERACTIONS]:
            if not es.indices.exists(index=index):
                logger.warning(f"Index {index} does not exist")

# Global variables to store our data and models
tracks_df = None
artists_df = None
user_listening_df = None  # This would be your user listening history
audio_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 
                 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']

# Load and prepare data
def load_data(data_path):
    global tracks_df, artists_df
    
    logger.info(f"Loading data from {data_path}")
    
    # Load tracks data
    tracks_path = os.path.join(data_path, "tracks.csv")
    tracks_df = pd.read_csv(tracks_path)
    
    # Load artists data if available
    artists_path = os.path.join(data_path, "artists.csv")
    if os.path.exists(artists_path):
        artists_df = pd.read_csv(artists_path)
    
    # Create a mock user listening history if not available
    # In a real system, this would come from your database
    global user_listening_df
    user_listening_df = create_mock_user_listening()
    
    logger.info(f"Loaded {len(tracks_df)} tracks and {len(user_listening_df)} user interactions")
    
    # Preprocess data
    preprocess_data()

def preprocess_data():
    """Preprocess the data for recommendation algorithms"""
    global tracks_df
    
    # Handle missing values in audio features
    for feature in audio_features:
        if feature in tracks_df.columns:
            tracks_df[feature].fillna(tracks_df[feature].mean(), inplace=True)
    
    # Normalize audio features
    scaler = MinMaxScaler()
    if all(feature in tracks_df.columns for feature in audio_features):
        tracks_df[audio_features] = scaler.fit_transform(tracks_df[audio_features])
    
    # Extract genres from artists and add to tracks
    if artists_df is not None and 'genres' in artists_df.columns:
        # Create a mapping of artist_id to genres
        artist_genres = artists_df.set_index('id')['genres'].to_dict()
        
        # Add genres to tracks
        if 'artists' in tracks_df.columns:
            # This assumes the artists column contains artist IDs
            # You might need to adjust this based on your actual data structure
            tracks_df['genres'] = tracks_df['artists'].apply(
                lambda x: get_genres_for_artists(x, artist_genres)
            )

def get_genres_for_artists(artist_ids, artist_genres):
    """Extract genres for a list of artists"""
    # This function needs to be adapted based on how artist IDs are stored in your dataset
    # For example, they might be stored as a string like "[id1, id2, ...]"
    try:
        if isinstance(artist_ids, str):
            # Try to parse as a list if it's a string
            import ast
            artist_list = ast.literal_eval(artist_ids)
        elif isinstance(artist_ids, list):
            artist_list = artist_ids
        else:
            return []
        
        # Collect genres for all artists
        genres = []
        for artist_id in artist_list:
            if artist_id in artist_genres:
                artist_genre = artist_genres[artist_id]
                if isinstance(artist_genre, str):
                    # If genres are stored as a string, parse them
                    import ast
                    genre_list = ast.literal_eval(artist_genre)
                    genres.extend(genre_list)
                elif isinstance(artist_genre, list):
                    genres.extend(artist_genre)
        
        return list(set(genres))  # Remove duplicates
    except:
        return []

def create_mock_user_listening():
    """Create mock user listening history for demonstration"""
    # In a real system, this would come from your database
    
    # Create a sample of 1000 users
    n_users = 1000
    n_interactions = 50000
    
    # Generate random user-track interactions
    user_ids = [f"user_{i}" for i in range(n_users)]
    
    interactions = []
    for _ in range(n_interactions):
        user_id = np.random.choice(user_ids)
        # Assuming tracks_df is loaded and has an 'id' column
        if tracks_df is not None and len(tracks_df) > 0:
            track_idx = np.random.randint(0, len(tracks_df))
            track_id = tracks_df.iloc[track_idx]['id']
            play_count = np.random.randint(1, 20)
            
            interactions.append({
                'user_id': user_id,
                'track_id': track_id,
                'play_count': play_count
            })
    
    return pd.DataFrame(interactions)

# Recommendation Algorithms

def content_based_recommend(track_id=None, user_tags=None, n=10):
    """
    Content-based recommendation using Elasticsearch
    Can recommend based on a seed track or user tags
    """
    if track_id:
        # Get the track's details
        try:
            track = es.get(index=ES_INDEX_TRACKS, id=track_id)['_source']
        except:
            logger.error(f"Track {track_id} not found")
            return []
        
        # Extract features for similarity search
        query = {
            "query": {
                "function_score": {
                    "query": {
                        "bool": {
                            "must_not": [
                                {"term": {"id": track_id}}  # Exclude the seed track
                            ],
                            "should": []
                        }
                    },
                    "functions": []
                }
            },
            "size": n
        }
        
        # Add mood matching
        if 'mood' in track and track['mood']:
            query["query"]["function_score"]["query"]["bool"]["should"].append(
                {"term": {"mood": track['mood']}}
            )
        
        # Add tags matching
        if 'tags' in track and track['tags']:
            query["query"]["function_score"]["query"]["bool"]["should"].append(
                {"terms": {"tags": track['tags']}}
            )
        
        # Add audio feature similarity
        audio_features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness']
        for feature in audio_features:
            if feature in track:
                query["query"]["function_score"]["functions"].append({
                    "gauss": {
                        feature: {
                            "origin": track[feature],
                            "scale": "0.2"
                        }
                    },
                    "weight": 1.0
                })
        
    elif user_tags:
        # Recommend based on user tags
        query = {
            "query": {
                "bool": {
                    "should": [
                        {"terms": {"tags": user_tags}}
                    ],
                    "minimum_should_match": 1
                }
            },
            "size": n
        }
    else:
        return []
    
    # Execute the query
    try:
        results = es.search(index=ES_INDEX_TRACKS, body=query)
        recommendations = []
        
        for hit in results['hits']['hits']:
            track = hit['_source']
            recommendations.append({
                'id': track['id'],
                'name': track.get('name', 'Unknown'),
                'artists': track.get('artist_names', 'Unknown'),
                'similarity_score': hit['_score']
            })
        
        return recommendations
    except Exception as e:
        logger.error(f"Error in content-based recommendation: {e}")
        return []

def collaborative_recommend(user_id, n=10):
    """Collaborative filtering recommendation using Elasticsearch"""
    if not user_id:
        return []
    
    # Step 1: Get the user's listening history
    user_query = {
        "query": {
            "term": {"user_id": user_id}
        },
        "size": 100  # Limit to recent 100 interactions
    }
    
    try:
        user_results = es.search(index=ES_INDEX_USER_INTERACTIONS, body=user_query)
        
        if len(user_results['hits']['hits']) == 0:
            logger.warning(f"No listening history found for user {user_id}")
            return []
        
        # Extract the tracks the user has listened to
        user_tracks = [hit['_source']['track_id'] for hit in user_results['hits']['hits']]
        
        # Step 2: Find users with similar listening history
        similar_users_query = {
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"track_id": user_tracks}},
                        {"bool": {"must_not": {"term": {"user_id": user_id}}}}
                    ]
                }
            },
            "aggs": {
                "similar_users": {
                    "terms": {
                        "field": "user_id",
                        "size": 20,
                        "order": {"track_overlap": "desc"}
                    },
                    "aggs": {
                        "track_overlap": {
                            "cardinality": {
                                "field": "track_id"
                            }
                        }
                    }
                }
            },
            "size": 0
        }
        
        similar_users_results = es.search(index=ES_INDEX_USER_INTERACTIONS, body=similar_users_query)
        
        similar_users = [
            bucket['key'] 
            for bucket in similar_users_results['aggregations']['similar_users']['buckets']
        ]
        
        if not similar_users:
            logger.warning(f"No similar users found for user {user_id}")
            return []
        
        # Step 3: Get tracks that similar users have listened to but the user hasn't
        recommendations_query = {
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"user_id": similar_users}}
                    ],
                    "must_not": [
                        {"terms": {"track_id": user_tracks}}
                    ]
                }
            },
            "aggs": {
                "track_recommendations": {
                    "terms": {
                        "field": "track_id",
                        "size": n * 2,  # Get more than needed to filter later
                        "order": {"avg_play_count": "desc"}
                    },
                    "aggs": {
                        "avg_play_count": {
                            "avg": {
                                "field": "play_count"
                            }
                        }
                    }
                }
            },
            "size": 0
        }
        
        recommendations_results = es.search(index=ES_INDEX_USER_INTERACTIONS, body=recommendations_query)
        
        # Extract recommended track IDs and scores
        track_scores = {
            bucket['key']: bucket['avg_play_count']['value']
            for bucket in recommendations_results['aggregations']['track_recommendations']['buckets']
        }
        
        # Step 4: Get track details
        if not track_scores:
            return []
        
        tracks_query = {
            "query": {
                "terms": {
                    "id": list(track_scores.keys())
                }
            },
            "size": n
        }
        
        tracks_results = es.search(index=ES_INDEX_TRACKS, body=tracks_query)
        
        # Format recommendations
        recommendations = []
        for hit in tracks_results['hits']['hits']:
            track = hit['_source']
            track_id = track['id']
            recommendations.append({
                'id': track_id,
                'name': track.get('name', 'Unknown'),
                'artists': track.get('artist_names', 'Unknown'),
                'score': track_scores[track_id]
            })
        
        # Sort by score and limit to n
        recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)[:n]
        
        return recommendations
    except Exception as e:
        logger.error(f"Error in collaborative recommendation: {e}")
        return []

def mood_based_recommend(mood, n=10):
    """Mood-based recommendations"""
    query = {
        "query": {
            "function_score": {
                "query": {
                    "term": {"mood": mood}
                },
                "random_score": {},  # Add some randomness for variety
                "boost_mode": "sum"
            }
        },
        "size": n
    }
    
    try:
        results = es.search(index=ES_INDEX_TRACKS, body=query)
        recommendations = []
        
        for hit in results['hits']['hits']:
            track = hit['_source']
            recommendations.append({
                'id': track['id'],
                'name': track.get('name', 'Unknown'),
                'artists': track.get('artist_names', 'Unknown'),
                'score': hit['_score']
            })
        
        return recommendations
    except Exception as e:
        logger.error(f"Error in mood-based recommendation: {e}")
        return []

def get_popular_songs(n=10):
    """Get popular songs based on Elasticsearch data"""
    # Use popularity field if available, otherwise use play count from interactions
    query = {
        "query": {
            "match_all": {}
        },
        "sort": [
            {"popularity": {"order": "desc"}}
        ],
        "size": n
    }
    
    try:
        results = es.search(index=ES_INDEX_TRACKS, body=query)
        recommendations = []
        
        for hit in results['hits']['hits']:
            track = hit['_source']
            recommendations.append({
                'id': track['id'],
                'name': track.get('name', 'Unknown'),
                'artists': track.get('artist_names', 'Unknown'),
                'popularity': track.get('popularity', 0)
            })
        
        return recommendations
    except Exception as e:
        logger.error(f"Error in popularity recommendation: {e}")
        return []

def get_user_recent_listens(user_id, n=5):
    """Get user's recent listens"""
    query = {
        "query": {
            "term": {"user_id": user_id}
        },
        "sort": [
            {"timestamp": {"order": "desc"}}
        ],
        "size": n
    }
    
    try:
        results = es.search(index=ES_INDEX_USER_INTERACTIONS, body=query)
        recent_track_ids = [hit['_source']['track_id'] for hit in results['hits']['hits']]
        
        if not recent_track_ids:
            return []
        
        # Get track details
        tracks_query = {
            "query": {
                "terms": {
                    "id": recent_track_ids
                }
            },
            "size": n
        }
        
        tracks_results = es.search(index=ES_INDEX_TRACKS, body=tracks_query)
        
        recent_tracks = []
        for hit in tracks_results['hits']['hits']:
            track = hit['_source']
            recent_tracks.append(track)
        
        return recent_tracks
    except Exception as e:
        logger.error(f"Error getting user's recent listens: {e}")
        return []

def extract_current_mood(tracks):
    """Extract current mood from tracks"""
    if not tracks:
        return "neutral"
    
    # Count mood occurrences
    mood_counts = defaultdict(int)
    for track in tracks:
        if 'mood' in track:
            mood_counts[track['mood']] += 1
    
    # Return the most common mood, or neutral if none found
    if mood_counts:
        return max(mood_counts.items(), key=lambda x: x[1])[0]
    
    # If no mood information, try to infer from audio features
    valence_sum = sum(track.get('valence', 0.5) for track in tracks)
    energy_sum = sum(track.get('energy', 0.5) for track in tracks)
    
    avg_valence = valence_sum / len(tracks)
    avg_energy = energy_sum / len(tracks)
    
    # Determine mood based on valence and energy
    if avg_valence > 0.6 and avg_energy > 0.6:
        return "happy"
    elif avg_valence > 0.6 and avg_energy <= 0.4:
        return "relaxed"
    elif avg_valence <= 0.4 and avg_energy > 0.6:
        return "intense"
    elif avg_valence <= 0.4 and avg_energy <= 0.4:
        return "sad"
    else:
        return "neutral"

def get_user_tags(user_id):
    """Get user's preferred tags based on listening history"""
    # Get user's recent listens
    recent_tracks = get_user_recent_listens(user_id, n=20)
    
    if not recent_tracks:
        return []
    
    # Count tag occurrences
    tag_counts = defaultdict(int)
    for track in recent_tracks:
        if 'tags' in track and track['tags']:
            for tag in track['tags']:
                tag_counts[tag] += 1
    
    # Return the most common tags
    sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
    return [tag for tag, count in sorted_tags[:10]]  # Return top 10 tags

def hybrid_recommend(user_id=None, seed_tracks=None, n=10):
    """Hybrid recommendation combining multiple approaches"""
    # Default weights for different recommendation types
    weights = {
        'content': 0.4,
        'collaborative': 0.3,
        'mood': 0.2,
        'popularity': 0.1
    }
    
    # Initialize scores
    track_scores = defaultdict(float)
    
    # Content-based recommendations from seed tracks or user tags
    if seed_tracks:
        for track_id in seed_tracks:
            content_recs = content_based_recommend(track_id=track_id, n=20)
            for rec in content_recs:
                track_scores[rec['id']] += weights['content'] * rec['similarity_score']
    elif user_id:
        # Get user tags from listening history
        user_tags = get_user_tags(user_id)
        if user_tags:
            content_recs = content_based_recommend(user_tags=user_tags, n=20)
            for rec in content_recs:
                track_scores[rec['id']] += weights['content'] * rec['similarity_score']
    
    # Collaborative filtering recommendations
    if user_id:
        collab_recs = collaborative_recommend(user_id, n=20)
        for rec in collab_recs:
            track_scores[rec['id']] += weights['collaborative'] * rec['score']
        
        # Mood-based recommendations
        recent_tracks = get_user_recent_listens(user_id)
        if recent_tracks:
            current_mood = extract_current_mood(recent_tracks)
            mood_recs = mood_based_recommend(current_mood, n=20)
            for rec in mood_recs:
                track_scores[rec['id']] += weights['mood'] * rec['score']
    
    # Popularity recommendations (as fallback)
    pop_recs = get_popular_songs(n=20)
    for rec in pop_recs:
        track_scores[rec['id']] += weights['popularity'] * (rec['popularity'] / 100 if 'popularity' in rec else 0.5)
    
    # If we still don't have enough recommendations, add more popular songs
    if len(track_scores) < n:
        more_pop_recs = get_popular_songs(n=n*2)
        for rec in more_pop_recs:
            if rec['id'] not in track_scores:
                track_scores[rec['id']] = weights['popularity'] * (rec['popularity'] / 100 if 'popularity' in rec else 0.5)
    
    # Sort and get top N
    top_tracks = sorted(track_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    
    # Get track details
    track_ids = [track_id for track_id, _ in top_tracks]
    
    if not track_ids:
        return []
    
    tracks_query = {
        "query": {
            "terms": {
                "id": track_ids
            }
        },
        "size": n
    }
    
    try:
        tracks_results = es.search(index=ES_INDEX_TRACKS, body=tracks_query)
        
        # Create a mapping of track_id to track details
        tracks_map = {hit['_source']['id']: hit['_source'] for hit in tracks_results['hits']['hits']}
        
        # Format recommendations
        recommendations = []
        for track_id, score in top_tracks:
            if track_id in tracks_map:
                track = tracks_map[track_id]
                recommendations.append({
                    'id': track_id,
                    'name': track.get('name', 'Unknown'),
                    'artists': track.get('artist_names', 'Unknown'),
                    'score': float(score)
                })
        
        return recommendations
    except Exception as e:
        logger.error(f"Error in hybrid recommendation: {e}")
        return []

# API Endpoints

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    es_status = "healthy" if es.ping() else "unhealthy"
    return jsonify({
        'status': 'healthy',
        'elasticsearch': es_status
    })

@app.route('/recommendations/content-based', methods=['GET'])
def get_content_based():
    """Get content-based recommendations"""
    track_id = request.args.get('track_id')
    user_tags = request.args.get('user_tags')
    n = int(request.args.get('n', 10))
    
    if user_tags:
        user_tags = user_tags.split(',')
    
    if not track_id and not user_tags:
        return jsonify({'error': 'Either track_id or user_tags parameter is required'}), 400
    
    recommendations = content_based_recommend(track_id=track_id, user_tags=user_tags, n=n)
    return jsonify({'recommendations': recommendations})

@app.route('/recommendations/collaborative', methods=['GET'])
def get_collaborative():
    """Get collaborative filtering recommendations"""
    user_id = request.args.get('user_id')
    n = int(request.args.get('n', 10))
    
    if not user_id:
        return jsonify({'error': 'user_id parameter is required'}), 400
    
    recommendations = collaborative_recommend(user_id, n)
    return jsonify({'recommendations': recommendations})

@app.route('/recommendations/mood', methods=['GET'])
def get_mood_based():
    """Get mood-based recommendations"""
    mood = request.args.get('mood')
    n = int(request.args.get('n', 10))
    
    if not mood:
        return jsonify({'error': 'mood parameter is required'}), 400
    
    recommendations = mood_based_recommend(mood, n)
    return jsonify({'recommendations': recommendations})

@app.route('/recommendations/popular', methods=['GET'])
def get_popular():
    """Get popularity-based recommendations"""
    n = int(request.args.get('n', 10))
    recommendations = get_popular_songs(n)
    return jsonify({'recommendations': recommendations})

@app.route('/recommendations/hybrid', methods=['GET'])
def get_hybrid():
    """Get hybrid recommendations"""
    user_id = request.args.get('user_id')
    seed_tracks = request.args.get('seed_tracks')
    n = int(request.args.get('n', 10))
    
    # Parse seed tracks if provided
    seed_track_list = None
    if seed_tracks:
        seed_track_list = seed_tracks.split(',')
    
    recommendations = hybrid_recommend(user_id, seed_track_list, n)
    return jsonify({'recommendations': recommendations})

@app.route('/track/<track_id>', methods=['GET'])
def get_track(track_id):
    """Get track details"""
    try:
        track = es.get(index=ES_INDEX_TRACKS, id=track_id)['_source']
        return jsonify({'track': track})
    except:
        return jsonify({'error': 'Track not found'}), 404

@app.route('/user/<user_id>/recent', methods=['GET'])
def get_user_recent(user_id):
    """Get user's recent listens"""
    n = int(request.args.get('n', 5))
    recent_tracks = get_user_recent_listens(user_id, n)
    return jsonify({'recent_tracks': recent_tracks})

@app.route('/user/<user_id>/mood', methods=['GET'])
def get_user_current_mood(user_id):
    """Get user's current mood based on recent listens"""
    recent_tracks = get_user_recent_listens(user_id)
    mood = extract_current_mood(recent_tracks)
    return jsonify({'mood': mood})

@app.route('/track-interaction', methods=['POST'])
def track_interaction():
    """Record a user's interaction with a track"""
    data = request.json
    
    if not data or 'user_id' not in data or 'track_id' not in data or 'interaction_type' not in data:
        return jsonify({'error': 'Missing required fields'}), 400
    
    user_id = data['user_id']
    track_id = data['track_id']
    interaction_type = data['interaction_type']
    
    # Create interaction document
    interaction = {
        'user_id': user_id,
        'track_id': track_id,
        'timestamp': pd.Timestamp.now().isoformat(),
        'play_count': 1 if interaction_type == 'play' else 0,
        'liked': interaction_type == 'like'
    }
    
    # Generate a unique ID for the interaction
    interaction_id = f"{user_id}_{track_id}_{pd.Timestamp.now().timestamp()}"
    
    try:
        # Index the interaction
        es.index(index=ES_INDEX_USER_INTERACTIONS, id=interaction_id, body=interaction)
        return jsonify({'success': True})
    except Exception as e:
        logger.error(f"Error recording track interaction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)