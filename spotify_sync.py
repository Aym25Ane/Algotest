import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import logging
from elasticsearch import Elasticsearch
from datetime import datetime
import asyncio
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class SpotifySync:
    def __init__(self, client_id: str, client_secret: str, es_client: Elasticsearch):
        self.sp = spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=client_id,
                client_secret=client_secret
            )
        )
        self.es = es_client
        self.index_name = 'spotify_tracks'

    async def sync_playlist(self, playlist_id: str):
        """Sync a Spotify playlist to Elasticsearch"""
        try:
            # Get playlist tracks
            results = self.sp.playlist_tracks(playlist_id)
            tracks = results['items']

            while results['next']:
                results = self.sp.next(results)
                tracks.extend(results['items'])

            # Process each track
            for item in tracks:
                track = item['track']
                if track:
                    await self.sync_track(track['id'])

        except Exception as e:
            logger.error(f"Error syncing playlist {playlist_id}: {str(e)}")

    async def sync_track(self, track_id: str) -> Optional[Dict]:
        """Sync a single track to Elasticsearch"""
        try:
            # Get track details
            track = self.sp.track(track_id)
            if not track:
                return None

            # Get audio features
            features = self.sp.audio_features(track_id)[0]
            if not features:
                return None

            # Get artist details
            artist = self.sp.artist(track['artists'][0]['id'])

            # Get album details
            album = self.sp.album(track['album']['id'])

            # Prepare document
            doc = {
                'id': track_id,
                'title': track['name'],
                'artist': {
                    'id': artist['id'],
                    'name': artist['name'],
                    'genres': artist['genres'],
                    'popularity': artist['popularity']
                },
                'album': {
                    'id': album['id'],
                    'name': album['name'],
                    'release_date': album['release_date'],
                    'total_tracks': album['total_tracks'],
                    'genres': album['genres']
                },
                'duration_ms': track['duration_ms'],
                'popularity': track['popularity'],
                'audio_features': {
                    'danceability': features['danceability'],
                    'energy': features['energy'],
                    'key': features['key'],
                    'loudness': features['loudness'],
                    'mode': features['mode'],
                    'speechiness': features['speechiness'],
                    'acousticness': features['acousticness'],
                    'instrumentalness': features['instrumentalness'],
                    'liveness': features['liveness'],
                    'valence': features['valence'],
                    'tempo': features['tempo']
                },
                'user_metrics': {
                    'viewCount': 0,
                    'totalReactionCount': 0,
                    'commentCount': 0,
                    'comments': []
                },
                'last_synced': datetime.now().isoformat()
            }

            # Index in Elasticsearch
            self.es.index(index=self.index_name, id=track_id, body=doc)
            return doc

        except Exception as e:
            logger.error(f"Error syncing track {track_id}: {str(e)}")
            return None

    async def search_tracks(self, query: str, limit: int = 10) -> List[Dict]:
        """Search tracks in Spotify and sync results to Elasticsearch"""
        try:
            results = self.sp.search(q=query, limit=limit, type='track')
            tracks = []

            for item in results['tracks']['items']:
                track = await self.sync_track(item['id'])
                if track:
                    tracks.append(track)

            return tracks

        except Exception as e:
            logger.error(f"Error searching tracks: {str(e)}")
            return []

    async def sync_user_playlists(self, user_id: str):
        """Sync all playlists of a user"""
        try:
            playlists = self.sp.user_playlists(user_id)
            for playlist in playlists['items']:
                await self.sync_playlist(playlist['id'])
        except Exception as e:
            logger.error(f"Error syncing user playlists: {str(e)}")

    async def update_track_metrics(self, track_id: str, metrics: Dict):
        """Update user interaction metrics for a track"""
        try:
            self.es.update(
                index=self.index_name,
                id=track_id,
                body={
                    'script': {
                        'source': '''
                            ctx._source.user_metrics.viewCount += params.viewCount;
                            ctx._source.user_metrics.totalReactionCount += params.reactionCount;
                            ctx._source.user_metrics.commentCount += params.commentCount;
                            if (params.comment != null) {
                                ctx._source.user_metrics.comments.add(params.comment);
                            }
                        ''',
                        'lang': 'painless',
                        'params': metrics
                    }
                }
            )
        except Exception as e:
            logger.error(f"Error updating track metrics: {str(e)}")

    async def get_recommendations(self, seed_tracks: List[str], limit: int = 20) -> List[Dict]:
        """Get recommendations from Spotify based on seed tracks"""
        try:
            recommendations = self.sp.recommendations(
                seed_tracks=seed_tracks[:5],  # Spotify allows max 5 seed tracks
                limit=limit
            )

            tracks = []
            for track in recommendations['tracks']:
                synced_track = await self.sync_track(track['id'])
                if synced_track:
                    tracks.append(synced_track)

            return tracks

        except Exception as e:
            logger.error(f"Error getting recommendations: {str(e)}")
            return []