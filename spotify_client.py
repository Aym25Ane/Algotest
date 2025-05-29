import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
import logging
from typing import Dict, List, Optional
from datetime import datetime
import config

logger = logging.getLogger(__name__)


class SpotifyClient:
    def __init__(self):
        # Client credentials flow (pour les opérations publiques)
        self.public_client = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=config.SPOTIFY_CONFIG['client_id'],
                client_secret=config.SPOTIFY_CONFIG['client_secret']
            )
        )

        # OAuth flow (pour les opérations utilisateur)
        self.user_client = spotipy.Spotify(
            auth_manager=SpotifyOAuth(
                client_id=config.SPOTIFY_CONFIG['client_id'],
                client_secret=config.SPOTIFY_CONFIG['client_secret'],
                redirect_uri=config.SPOTIFY_CONFIG['redirect_uri'],
                scope=config.SPOTIFY_CONFIG['scope']
            )
        )

    def get_track_details(self, track_id: str) -> Optional[Dict]:
        """Récupère les détails d'une chanson"""
        try:
            track = self.public_client.track(track_id)
            features = self.public_client.audio_features(track_id)[0]
            artist = self.public_client.artist(track['artists'][0]['id'])

            return {
                'id': track_id,
                'name': track['name'],
                'artist': {
                    'id': artist['id'],
                    'name': artist['name'],
                    'genres': artist['genres']
                },
                'album': {
                    'id': track['album']['id'],
                    'name': track['album']['name'],
                    'release_date': track['album']['release_date']
                },
                'audio_features': features,
                'popularity': track['popularity']
            }
        except Exception as e:
            logger.error(f"Error getting track details: {str(e)}")
            return None

    def get_user_top_tracks(self, user_id: str, time_range: str = 'medium_term', limit: int = 20) -> List[Dict]:
        """Récupère les chansons les plus écoutées d'un utilisateur"""
        try:
            results = self.user_client.current_user_top_tracks(
                time_range=time_range,
                limit=limit
            )
            return results['items']
        except Exception as e:
            logger.error(f"Error getting user top tracks: {str(e)}")
            return []

    def get_user_recently_played(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Récupère l'historique d'écoute récent d'un utilisateur"""
        try:
            results = self.user_client.current_user_recently_played(limit=limit)
            return results['items']
        except Exception as e:
            logger.error(f"Error getting recently played: {str(e)}")
            return []

    def get_user_playlists(self, user_id: str) -> List[Dict]:
        """Récupère les playlists d'un utilisateur"""
        try:
            results = self.user_client.user_playlists(user_id)
            return results['items']
        except Exception as e:
            logger.error(f"Error getting user playlists: {str(e)}")
            return []

    def get_playlist_tracks(self, playlist_id: str) -> List[Dict]:
        """Récupère les chansons d'une playlist"""
        try:
            results = self.public_client.playlist_tracks(playlist_id)
            tracks = results['items']

            while results['next']:
                results = self.public_client.next(results)
                tracks.extend(results['items'])

            return tracks
        except Exception as e:
            logger.error(f"Error getting playlist tracks: {str(e)}")
            return []

    def get_recommendations(self, seed_tracks: List[str], limit: int = 20) -> List[Dict]:
        """Obtient des recommandations basées sur des chansons"""
        try:
            results = self.public_client.recommendations(
                seed_tracks=seed_tracks[:5],  # Spotify limite à 5 seed tracks
                limit=limit
            )
            return results['tracks']
        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []

    def search_tracks(self, query: str, limit: int = 20) -> List[Dict]:
        """Recherche des chansons"""
        try:
            results = self.public_client.search(
                q=query,
                limit=limit,
                type='track'
            )
            return results['tracks']['items']
        except Exception as e:
            logger.error(f"Error searching tracks: {str(e)}")
            return []

    def get_audio_features(self, track_ids: List[str]) -> List[Dict]:
        """Récupère les caractéristiques audio de plusieurs chansons"""
        try:
            features = self.public_client.audio_features(track_ids)
            return features
        except Exception as e:
            logger.error(f"Error getting audio features: {str(e)}")
            return []

    def get_artist_top_tracks(self, artist_id: str) -> List[Dict]:
        """Récupère les meilleures chansons d'un artiste"""
        try:
            results = self.public_client.artist_top_tracks(artist_id)
            return results['tracks']
        except Exception as e:
            logger.error(f"Error getting artist top tracks: {str(e)}")
            return []

    def get_related_artists(self, artist_id: str) -> List[Dict]:
        """Récupère les artistes similaires"""
        try:
            results = self.public_client.artist_related_artists(artist_id)
            return results['artists']
        except Exception as e:
            logger.error(f"Error getting related artists: {str(e)}")
            return []