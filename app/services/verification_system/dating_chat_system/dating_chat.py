import logging

from app.services.verification_system.api_manager.dating_chat_manager import ChatService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatingChatSystem:
    def __init__(self):
        self.chat_service = ChatService()

    async def get_chat_response(self, user_message: str, user_id: str) -> str:
        """Get response from the dating chat assistant"""
        logger.info(f"User {user_id} sent message: {user_message}")
        response = await self.chat_service.get_chat_response(user_message, user_id)
        logger.info(f"Assistant response for user {user_id}: {response}")
        return response

    def get_chat_history(self, user_id: str):
        """Get chat history for a specific user"""
        history = self.chat_service.get_history(user_id)
        logger.info(f"Chat history for user {user_id}: {history}")
        return history

    def clear_chat_history(self, user_id: str):
        """Clear chat history for a specific user"""
        self.chat_service.clear_history(user_id)
        logger.info(f"Chat history cleared for user {user_id}")