

SPOTIFY_CONFIG = {
    'client_id': '2cdafe028d9e4a49a1ba5d18c8ce31b8',  # hada compte dyali
    'client_secret': 'dcae0ed3414742e6bc6021fbc6b648f5',  # hada compte dyali
    'redirect_uri': 'http://127.0.0.1:4040',
    'scope': [
        'user-read-private',
        'user-read-email',
        'user-library-read',
        'user-top-read',
        'user-read-recently-played',
        'user-read-playback-state',
        'user-modify-playback-state',
        'streaming',
        'app-remote-control',
        'playlist-read-private',
        'playlist-read-collaborative',
        'user-read-currently-playing',
        'user-read-playback-position',
        'user-read-recently-played',
        'user-read-currently-playing',
        'user-read-playback-state',
        'user-modify-playback-state'
    ]
}

# Elasticsearch Configuration
ES_CONFIG = {
    'hosts': ['http://localhost:9200'],
    'index_prefix': 'spotify_'
}

# Kafka Configuration
KAFKA_CONFIG = {
    'bootstrap_servers': ['localhost:9092'],
    'group_id': 'spotify-recommendation-group'
}