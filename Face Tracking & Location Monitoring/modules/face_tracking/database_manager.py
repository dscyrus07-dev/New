"""
Database Manager for Face Tracking & Location Monitoring

This module handles database operations for storing tracking data,
zone transitions, and movement logs.
"""

import sqlite3
import json
import pickle
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from contextlib import contextmanager
import threading
from dataclasses import asdict

from .tracker import Track
from .zone_monitor import ZoneTransition

logger = logging.getLogger(__name__)

class TrackingDatabase:
    """
    Database manager for storing face tracking and zone monitoring data.
    Handles tracks, zone transitions, movement patterns, and statistics.
    """
    
    def __init__(self, db_path: str = "data/tracking.db"):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_database()
        
        logger.info(f"TrackingDatabase initialized: {db_path}")
    
    def _init_database(self):
        """Initialize database tables"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Tracks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    bbox_x INTEGER NOT NULL,
                    bbox_y INTEGER NOT NULL,
                    bbox_w INTEGER NOT NULL,
                    bbox_h INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    identity TEXT,
                    identity_confidence REAL,
                    encoding BLOB,
                    center_x INTEGER,
                    center_y INTEGER,
                    session_id TEXT
                )
            ''')
            
            # Zone transitions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS zone_transitions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL,
                    from_zone TEXT,
                    to_zone TEXT,
                    timestamp DATETIME NOT NULL,
                    confidence REAL NOT NULL,
                    identity TEXT,
                    session_id TEXT
                )
            ''')
            
            # Movement patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS movement_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL,
                    identity TEXT,
                    total_transitions INTEGER,
                    zone_dwell_times TEXT,  -- JSON
                    zone_visit_counts TEXT, -- JSON
                    first_seen DATETIME,
                    last_seen DATETIME,
                    most_visited_zone TEXT,
                    analysis_date DATETIME,
                    session_id TEXT
                )
            ''')
            
            # Zone statistics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS zone_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    zone_name TEXT NOT NULL,
                    date DATE NOT NULL,
                    entries INTEGER DEFAULT 0,
                    exits INTEGER DEFAULT 0,
                    max_occupancy INTEGER DEFAULT 0,
                    avg_dwell_time REAL DEFAULT 0,
                    total_time REAL DEFAULT 0,
                    session_id TEXT,
                    UNIQUE(zone_name, date, session_id)
                )
            ''')
            
            # System events table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    event_data TEXT,  -- JSON
                    timestamp DATETIME NOT NULL,
                    session_id TEXT
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_track_id ON tracks(track_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_timestamp ON tracks(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tracks_identity ON tracks(identity)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transitions_track_id ON zone_transitions(track_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transitions_timestamp ON zone_transitions(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_zone_stats_date ON zone_statistics(date)')
            
            conn.commit()
            logger.info("Database tables initialized successfully")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper locking"""
        with self.lock:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()
    
    def save_track(self, track: Track, session_id: str = None) -> bool:
        """
        Save track data to database.
        
        Args:
            track: Track object to save
            session_id: Optional session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Serialize face encoding if present
                encoding_blob = None
                if track.encoding is not None:
                    encoding_blob = pickle.dumps(track.encoding)
                
                # Calculate center point
                center_x = track.bbox[0] + track.bbox[2] // 2
                center_y = track.bbox[1] + track.bbox[3] // 2
                
                cursor.execute('''
                    INSERT INTO tracks (
                        track_id, timestamp, bbox_x, bbox_y, bbox_w, bbox_h,
                        confidence, identity, identity_confidence, encoding,
                        center_x, center_y, session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    track.track_id,
                    datetime.now(),
                    track.bbox[0], track.bbox[1], track.bbox[2], track.bbox[3],
                    track.confidence,
                    track.identity,
                    track.identity_confidence,
                    encoding_blob,
                    center_x, center_y,
                    session_id
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to save track {track.track_id}: {e}")
            return False
    
    def save_zone_transition(self, transition: ZoneTransition, session_id: str = None) -> bool:
        """
        Save zone transition to database.
        
        Args:
            transition: ZoneTransition object to save
            session_id: Optional session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO zone_transitions (
                        track_id, from_zone, to_zone, timestamp,
                        confidence, identity, session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    transition.track_id,
                    transition.from_zone,
                    transition.to_zone,
                    transition.timestamp,
                    transition.confidence,
                    transition.identity,
                    session_id
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to save zone transition: {e}")
            return False
    
    def save_movement_pattern(self, track_id: int, pattern_data: Dict, session_id: str = None) -> bool:
        """
        Save movement pattern analysis.
        
        Args:
            track_id: Track ID
            pattern_data: Movement pattern analysis data
            session_id: Optional session identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO movement_patterns (
                        track_id, identity, total_transitions,
                        zone_dwell_times, zone_visit_counts,
                        first_seen, last_seen, most_visited_zone,
                        analysis_date, session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    track_id,
                    pattern_data.get('identity'),
                    pattern_data.get('total_transitions', 0),
                    json.dumps(pattern_data.get('zone_dwell_times', {})),
                    json.dumps(pattern_data.get('zone_visit_counts', {})),
                    pattern_data.get('first_seen'),
                    pattern_data.get('last_seen'),
                    pattern_data.get('most_visited_zone'),
                    datetime.now(),
                    session_id
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to save movement pattern for track {track_id}: {e}")
            return False
    
    def update_zone_statistics(self, zone_stats: Dict[str, Dict], session_id: str = None):
        """
        Update zone statistics in database.
        
        Args:
            zone_stats: Dictionary of zone statistics
            session_id: Optional session identifier
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                today = datetime.now().date()
                
                for zone_name, stats in zone_stats.items():
                    cursor.execute('''
                        INSERT OR REPLACE INTO zone_statistics (
                            zone_name, date, entries, exits, max_occupancy,
                            avg_dwell_time, total_time, session_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        zone_name,
                        today,
                        stats.get('entries', 0),
                        stats.get('exits', 0),
                        stats.get('max_occupancy', 0),
                        stats.get('avg_dwell_time', 0.0),
                        stats.get('total_time', 0.0),
                        session_id
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to update zone statistics: {e}")
    
    def log_system_event(self, event_type: str, event_data: Dict = None, session_id: str = None):
        """
        Log system event.
        
        Args:
            event_type: Type of event
            event_data: Additional event data
            session_id: Optional session identifier
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO system_events (event_type, event_data, timestamp, session_id)
                    VALUES (?, ?, ?, ?)
                ''', (
                    event_type,
                    json.dumps(event_data) if event_data else None,
                    datetime.now(),
                    session_id
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to log system event: {e}")
    
    def get_tracks_by_timerange(self, start_time: datetime, end_time: datetime, 
                               identity: str = None) -> List[Dict]:
        """
        Get tracks within time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            identity: Optional identity filter
            
        Returns:
            List of track records
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT * FROM tracks 
                    WHERE timestamp BETWEEN ? AND ?
                '''
                params = [start_time, end_time]
                
                if identity:
                    query += ' AND identity = ?'
                    params.append(identity)
                
                query += ' ORDER BY timestamp'
                
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get tracks by time range: {e}")
            return []
    
    def get_zone_transitions_by_timerange(self, start_time: datetime, end_time: datetime,
                                        zone_name: str = None) -> List[Dict]:
        """
        Get zone transitions within time range.
        
        Args:
            start_time: Start of time range
            end_time: End of time range
            zone_name: Optional zone filter
            
        Returns:
            List of transition records
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT * FROM zone_transitions 
                    WHERE timestamp BETWEEN ? AND ?
                '''
                params = [start_time, end_time]
                
                if zone_name:
                    query += ' AND (from_zone = ? OR to_zone = ?)'
                    params.extend([zone_name, zone_name])
                
                query += ' ORDER BY timestamp'
                
                cursor.execute(query, params)
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get zone transitions by time range: {e}")
            return []
    
    def get_daily_zone_statistics(self, date: datetime.date = None) -> List[Dict]:
        """
        Get daily zone statistics.
        
        Args:
            date: Date to get statistics for (default: today)
            
        Returns:
            List of zone statistics
        """
        if date is None:
            date = datetime.now().date()
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM zone_statistics 
                    WHERE date = ?
                    ORDER BY zone_name
                ''', (date,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get daily zone statistics: {e}")
            return []
    
    def get_track_movement_pattern(self, track_id: int) -> Optional[Dict]:
        """
        Get movement pattern for a specific track.
        
        Args:
            track_id: Track ID
            
        Returns:
            Movement pattern data or None
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM movement_patterns 
                    WHERE track_id = ?
                    ORDER BY analysis_date DESC
                    LIMIT 1
                ''', (track_id,))
                
                row = cursor.fetchone()
                if row:
                    data = dict(row)
                    # Parse JSON fields
                    data['zone_dwell_times'] = json.loads(data['zone_dwell_times'] or '{}')
                    data['zone_visit_counts'] = json.loads(data['zone_visit_counts'] or '{}')
                    return data
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get movement pattern for track {track_id}: {e}")
            return None
    
    def get_identity_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get statistics for all identities.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary of identity statistics
        """
        start_time = datetime.now() - timedelta(hours=hours)
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get identity visit counts
                cursor.execute('''
                    SELECT identity, COUNT(DISTINCT track_id) as track_count,
                           COUNT(*) as total_detections,
                           MIN(timestamp) as first_seen,
                           MAX(timestamp) as last_seen
                    FROM tracks 
                    WHERE timestamp >= ? AND identity IS NOT NULL
                    GROUP BY identity
                    ORDER BY track_count DESC
                ''', (start_time,))
                
                identity_stats = {}
                for row in cursor.fetchall():
                    identity_stats[row['identity']] = {
                        'track_count': row['track_count'],
                        'total_detections': row['total_detections'],
                        'first_seen': row['first_seen'],
                        'last_seen': row['last_seen']
                    }
                
                # Get zone transition counts by identity
                cursor.execute('''
                    SELECT identity, COUNT(*) as transition_count
                    FROM zone_transitions 
                    WHERE timestamp >= ? AND identity IS NOT NULL
                    GROUP BY identity
                ''', (start_time,))
                
                for row in cursor.fetchall():
                    if row['identity'] in identity_stats:
                        identity_stats[row['identity']]['zone_transitions'] = row['transition_count']
                
                return identity_stats
                
        except Exception as e:
            logger.error(f"Failed to get identity statistics: {e}")
            return {}
    
    def cleanup_old_data(self, days: int = 30):
        """
        Remove old data from database.
        
        Args:
            days: Keep data newer than this many days
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Clean up old tracks
                cursor.execute('DELETE FROM tracks WHERE timestamp < ?', (cutoff_date,))
                tracks_deleted = cursor.rowcount
                
                # Clean up old transitions
                cursor.execute('DELETE FROM zone_transitions WHERE timestamp < ?', (cutoff_date,))
                transitions_deleted = cursor.rowcount
                
                # Clean up old movement patterns
                cursor.execute('DELETE FROM movement_patterns WHERE analysis_date < ?', (cutoff_date,))
                patterns_deleted = cursor.rowcount
                
                # Clean up old zone statistics
                cursor.execute('DELETE FROM zone_statistics WHERE date < ?', (cutoff_date.date(),))
                stats_deleted = cursor.rowcount
                
                # Clean up old system events
                cursor.execute('DELETE FROM system_events WHERE timestamp < ?', (cutoff_date,))
                events_deleted = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleaned up old data: {tracks_deleted} tracks, "
                          f"{transitions_deleted} transitions, {patterns_deleted} patterns, "
                          f"{stats_deleted} stats, {events_deleted} events")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    def export_data_to_csv(self, output_dir: str, hours: int = 24):
        """
        Export data to CSV files.
        
        Args:
            output_dir: Output directory for CSV files
            hours: Number of hours of data to export
        """
        import pandas as pd
        import os
        
        start_time = datetime.now() - timedelta(hours=hours)
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            with self._get_connection() as conn:
                # Export tracks
                tracks_df = pd.read_sql_query('''
                    SELECT track_id, timestamp, bbox_x, bbox_y, bbox_w, bbox_h,
                           confidence, identity, identity_confidence, center_x, center_y
                    FROM tracks WHERE timestamp >= ?
                    ORDER BY timestamp
                ''', conn, params=(start_time,))
                tracks_df.to_csv(os.path.join(output_dir, 'tracks.csv'), index=False)
                
                # Export zone transitions
                transitions_df = pd.read_sql_query('''
                    SELECT track_id, from_zone, to_zone, timestamp, confidence, identity
                    FROM zone_transitions WHERE timestamp >= ?
                    ORDER BY timestamp
                ''', conn, params=(start_time,))
                transitions_df.to_csv(os.path.join(output_dir, 'zone_transitions.csv'), index=False)
                
                # Export zone statistics
                stats_df = pd.read_sql_query('''
                    SELECT zone_name, date, entries, exits, max_occupancy,
                           avg_dwell_time, total_time
                    FROM zone_statistics 
                    WHERE date >= ?
                    ORDER BY date, zone_name
                ''', conn, params=(start_time.date(),))
                stats_df.to_csv(os.path.join(output_dir, 'zone_statistics.csv'), index=False)
                
                logger.info(f"Data exported to {output_dir}")
                
        except Exception as e:
            logger.error(f"Failed to export data to CSV: {e}")
    
    def get_database_stats(self) -> Dict[str, int]:
        """Get database statistics"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Count records in each table
                tables = ['tracks', 'zone_transitions', 'movement_patterns', 
                         'zone_statistics', 'system_events']
                
                for table in tables:
                    cursor.execute(f'SELECT COUNT(*) FROM {table}')
                    stats[f'{table}_count'] = cursor.fetchone()[0]
                
                # Get database size
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                stats['database_size_bytes'] = cursor.fetchone()[0]
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}