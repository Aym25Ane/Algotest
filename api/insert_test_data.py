from elasticsearch import Elasticsearch

# Connect to Elasticsearch (adjust host/port if needed)
es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

# Example songs
songs = [
    {
        "id": "song1",
        "title": "Test Song 1",
        "artist": "Test Artist",
        "genre": "Pop",
        "album": "Test Album",
        "duration": 180,
        "releaseDate": "2023-01-01T00:00:00Z",
        "language": "en",
        "tags": ["happy", "test"],
        "createdAt": "2023-01-01T00:00:00Z",
        "commentCount": 2,
        "totalReactionCount": 5,
        "viewCount": 100
    },
    {
        "id": "song2",
        "title": "Test Song 2",
        "artist": "Another Artist",
        "genre": "Rock",
        "album": "Another Album",
        "duration": 200,
        "releaseDate": "2023-02-01T00:00:00Z",
        "language": "en",
        "tags": ["energetic", "guitar"],
        "createdAt": "2023-02-01T00:00:00Z",
        "commentCount": 1,
        "totalReactionCount": 3,
        "viewCount": 50
    },
    {
        "id": "song3",
        "title": "Test Song 3",
        "artist": "Test Artist",
        "genre": "Jazz",
        "album": "Jazz Album",
        "duration": 240,
        "releaseDate": "2023-03-01T00:00:00Z",
        "language": "fr",
        "tags": ["smooth", "evening"],
        "createdAt": "2023-03-01T00:00:00Z",
        "commentCount": 0,
        "totalReactionCount": 1,
        "viewCount": 20
    }
]

# Example user
user = {
    "id": "user1",
    "username": "testuser",
    "email": "test@example.com",
    "favorites": ["song1", "song2"],
    "playlists": [],
    "createdAt": "2023-01-01T00:00:00Z"
}

# Insert songs
for song in songs:
    es.index(index="songs", id=song["id"], body=song)
    print(f"Inserted song: {song['title']}")

# Insert user
es.index(index="users", id=user["id"], body=user)
print(f"Inserted user: {user['username']}")