# models/models.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from enum import Enum

class Emoji(str, Enum):
    HEART = "❤️"
    SMILE = "😊"
    MUSIC = "🎵"
    FIRE = "🔥"

class Song(BaseModel):
    id: str
    title: str
    artist: str
    genre: str
    duration: int
    commentCount: int
    totalReactionCount: int
    viewCount: int
    releaseDate: datetime
    language: str
    tags: List[str]

class Reaction(BaseModel):
    id: str
    songId: str
    emojis: Emoji
    reactorName: str
    date: datetime

class User(BaseModel):
    id: str
    username: str
    email: str
    role: str
    favorites: List[str]
    playlists: List[str]

class RecommendationRequest(BaseModel):
    userId: str
    limit: int = 10