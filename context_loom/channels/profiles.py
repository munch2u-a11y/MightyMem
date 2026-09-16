"""Backend SQLite User Profile Store for multi-channel user identity routing."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class UserProfile:
    """User profile mapping external channel identity to internal memory scope."""

    profile_id: str
    channel_id: str
    external_user_id: str
    display_name: str
    memory_scope: str
    created_at: float
    last_active_at: float
    preferences: Dict[str, Any] = field(default_factory=dict)


class UserProfileStore:
    """Persistent SQLite store for managing user profiles and memory scopes across channel adapters."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    profile_id TEXT PRIMARY KEY,
                    channel_id TEXT NOT NULL,
                    external_user_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    memory_scope TEXT NOT NULL UNIQUE,
                    created_at REAL NOT NULL,
                    last_active_at REAL NOT NULL,
                    preferences_json TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(channel_id, external_user_id)
                );
                """
            )

    def get_or_create_profile(
        self,
        channel_id: str,
        external_user_id: str,
        display_name: str = "User",
        preferences: Optional[Dict[str, Any]] = None,
    ) -> UserProfile:
        """Fetch existing profile by channel & external user ID, or create a new isolated profile."""
        now = time.time()
        cursor = self.connection.execute(
            """
            SELECT profile_id, channel_id, external_user_id, display_name, memory_scope, created_at, last_active_at, preferences_json
            FROM user_profiles
            WHERE channel_id = ? AND external_user_id = ?;
            """,
            (channel_id, external_user_id),
        )
        row = cursor.fetchone()
        if row is not None:
            # Update last_active_at and display_name if changed
            profile_id = row["profile_id"]
            memory_scope = row["memory_scope"]
            created_at = row["created_at"]
            pref_dict = json.loads(row["preferences_json"] or "{}")
            if preferences:
                pref_dict.update(preferences)

            with self.connection:
                self.connection.execute(
                    """
                    UPDATE user_profiles
                    SET last_active_at = ?, display_name = ?, preferences_json = ?
                    WHERE profile_id = ?;
                    """,
                    (now, display_name, json.dumps(pref_dict), profile_id),
                )

            return UserProfile(
                profile_id=profile_id,
                channel_id=channel_id,
                external_user_id=external_user_id,
                display_name=display_name,
                memory_scope=memory_scope,
                created_at=created_at,
                last_active_at=now,
                preferences=pref_dict,
            )

        # Create new profile
        raw_uuid = uuid.uuid4().hex[:12]
        profile_id = f"usr_{raw_uuid}"
        memory_scope = f"user_{profile_id}"
        pref_json = json.dumps(preferences or {})

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO user_profiles (profile_id, channel_id, external_user_id, display_name, memory_scope, created_at, last_active_at, preferences_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    profile_id,
                    channel_id,
                    external_user_id,
                    display_name,
                    memory_scope,
                    now,
                    now,
                    pref_json,
                ),
            )

        return UserProfile(
            profile_id=profile_id,
            channel_id=channel_id,
            external_user_id=external_user_id,
            display_name=display_name,
            memory_scope=memory_scope,
            created_at=now,
            last_active_at=now,
            preferences=preferences or {},
        )

    def get_profile_by_scope(self, memory_scope: str) -> Optional[UserProfile]:
        """Fetch profile by its isolated memory scope."""
        cursor = self.connection.execute(
            """
            SELECT profile_id, channel_id, external_user_id, display_name, memory_scope, created_at, last_active_at, preferences_json
            FROM user_profiles
            WHERE memory_scope = ?;
            """,
            (memory_scope,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return UserProfile(
            profile_id=row["profile_id"],
            channel_id=row["channel_id"],
            external_user_id=row["external_user_id"],
            display_name=row["display_name"],
            memory_scope=row["memory_scope"],
            created_at=row["created_at"],
            last_active_at=row["last_active_at"],
            preferences=json.loads(row["preferences_json"] or "{}"),
        )

    def list_profiles(self) -> List[UserProfile]:
        """List all registered user profiles in the store."""
        cursor = self.connection.execute(
            """
            SELECT profile_id, channel_id, external_user_id, display_name, memory_scope, created_at, last_active_at, preferences_json
            FROM user_profiles
            ORDER BY last_active_at DESC;
            """
        )
        return [
            UserProfile(
                profile_id=row["profile_id"],
                channel_id=row["channel_id"],
                external_user_id=row["external_user_id"],
                display_name=row["display_name"],
                memory_scope=row["memory_scope"],
                created_at=row["created_at"],
                last_active_at=row["last_active_at"],
                preferences=json.loads(row["preferences_json"] or "{}"),
            )
            for row in cursor.fetchall()
        ]

    def close(self) -> None:
        """Close SQLite database connection."""
        self.connection.close()
