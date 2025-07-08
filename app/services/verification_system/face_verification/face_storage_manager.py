import logging
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class FaceStorageManager:
    """In-memory storage manager for face tokens"""
    
    def __init__(self):
        # Dictionary to store face tokens grouped by upload session
        # Format: {session_id: {"face_tokens": [tokens], "created_at": datetime, "count": int}}
        self.face_data: Dict[str, Dict] = {}
        self.MAX_TOKENS_PER_SESSION = 1000
        
    def save_face_token(self, face_token: str, session_id: str = "default") -> bool:
        """Save a face token to the in-memory storage
        
        Args:
            face_token: The face token to save
            session_id: Session identifier (default: "default")
        """
        try:
            if session_id not in self.face_data:
                self.face_data[session_id] = {
                    "face_tokens": [],
                    "created_at": datetime.now(),
                    "count": 0
                }
            
            session_data = self.face_data[session_id]
            
            # Check if we've reached the maximum capacity
            if session_data["count"] >= self.MAX_TOKENS_PER_SESSION:
                logger.warning(f"Session {session_id} has reached maximum capacity")
                return False
                
            # Add the face token
            session_data["face_tokens"].append(face_token)
            session_data["count"] += 1
            
            logger.info(f"Successfully saved face token to session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving face token: {str(e)}")
            return False

    def get_all_stored_faces(self) -> Dict[str, List[str]]:
        """Get all stored face tokens grouped by session"""
        try:
            result = {}
            for session_id, session_data in self.face_data.items():
                result[session_id] = session_data.get("face_tokens", [])
            return result
        except Exception as e:
            logger.error(f"Error getting stored faces: {str(e)}")
            return {}

    def get_session_metadata(self) -> Dict[str, Dict]:
        """Get metadata for all sessions"""
        try:
            result = {}
            for session_id, session_data in self.face_data.items():
                result[session_id] = {
                    "count": session_data.get("count", 0),
                    "created_at": session_data.get("created_at")
                }
            return result
        except Exception as e:
            logger.error(f"Error getting session metadata: {str(e)}")
            return {}

    def clear_session_data(self, session_id: str = None) -> bool:
        """Clear stored face data for a specific session or all sessions"""
        try:
            if session_id:
                if session_id in self.face_data:
                    del self.face_data[session_id]
                    logger.info(f"Cleared data for session {session_id}")
                    return True
                else:
                    logger.warning(f"Session {session_id} not found")
                    return False
            else:
                # Clear all sessions
                self.face_data.clear()
                logger.info("Cleared all session data")
                return True
                
        except Exception as e:
            logger.error(f"Error clearing session data: {str(e)}")
            return False

    def get_total_face_count(self) -> int:
        """Get total number of face tokens stored"""
        try:
            total = sum(session_data.get("count", 0) for session_data in self.face_data.values())
            return total
        except Exception as e:
            logger.error(f"Error getting total face count: {str(e)}")
            return 0
