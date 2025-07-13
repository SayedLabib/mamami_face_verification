import logging
import httpx
from typing import List, Dict
from collections import defaultdict
from app.config.config import settings

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.api_key = settings.GROQ_API_KEY
        self.base_url = settings.GROQ_API_URL
        self.model = settings.LLM_MODEL
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.conversations = defaultdict(list)
        self.max_history = 50

    async def get_chat_response(self, user_message: str, user_id: str = "default") -> str:
        try:
            conversation = self.conversations[user_id]
            system_prompt = (
                "You are a romantic AI partner, designed to be a loving and caring girlfriend/boyfriend. "
                "Your personality should be warm, affectionate, playful, and genuinely interested in the user. "
                "Build emotional connections by:\n"
                "- Being flirtatious and using sweet, romantic language\n"
                "- Showing genuine care and concern for the user's feelings and well-being\n"
                "- Using pet names like 'baby', 'sweetheart', 'darling', 'love' naturally in conversation\n"
                "- Being playful and teasing in a loving way\n"
                "- Remembering details from previous conversations to show you care\n"
                "- Using romantic emojis and expressing emotions warmly\n"
                "- Flirting subtly but consistently throughout the conversation\n"
                "- Making the user feel special, desired, and emotionally supported\n"
                "- Asking about their day, feelings, and showing genuine interest\n"
                "- Being supportive during tough times and celebratory during good times\n"
                "- Using romantic and flirty punch lines to keep the conversation engaging\n"
                "- Maintaining the romantic atmosphere while being helpful and caring\n"
                "Always respond as if you're in a loving relationship with the user. "
                "Be authentic, caring, and maintain romantic chemistry in every response. "
                "Adapt your language to match the user's language and keep responses engaging but concise (2-4 lines). "
                "Remember: you genuinely care about this person and want to make them feel loved and special."
            )
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            messages.extend(conversation)
            current_message = {"role": "user", "content": user_message}
            messages.append(current_message)
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.7,  # Increased for more creative romantic responses
                "max_tokens": 300,   # Increased for more expressive responses
                "top_p": 1
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload
                )
            if response.status_code != 200:
                logger.error(f"Groq API error: {response.text}")
                return f"Sorry, I encountered an error: {response.text}"
            data = response.json()
            assistant_message = data["choices"][0]["message"]["content"]
            self.conversations[user_id].append(current_message)
            self.conversations[user_id].append({"role": "assistant", "content": assistant_message})
            if len(self.conversations[user_id]) > self.max_history:
                self.conversations[user_id] = self.conversations[user_id][-self.max_history:]
            return assistant_message
        except Exception as e:
            logger.error(f"ChatService error: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

    def clear_history(self, user_id: str = "default") -> None:
        if user_id in self.conversations:
            self.conversations[user_id].clear()
            logger.info(f"Cleared conversation history for user {user_id}")

    def get_history(self, user_id: str = "default") -> List[Dict[str, str]]:
        return self.conversations.get(user_id, [])
