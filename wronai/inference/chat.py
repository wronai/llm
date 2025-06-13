"""
Chat functionality for WronAI models.
"""

import json
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable

from .engine import InferenceEngine, InferenceConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ChatMessage:
    """Represents a single chat message."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float
    message_id: str = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.message_id is None:
            self.message_id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class ConversationContext:
    """Context information for a conversation."""

    conversation_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    created_at: float = None
    last_updated: float = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.last_updated is None:
            self.last_updated = self.created_at
        if self.metadata is None:
            self.metadata = {}

class ConversationManager:
    """
    Manages conversation history and context.
    """

    def __init__(
        self,
        max_history_length: int = 20,
        max_context_tokens: int = 2048,
        save_conversations: bool = False,
        storage_path: Optional[str] = None
    ):
        self.max_history_length = max_history_length
        self.max_context_tokens = max_context_tokens
        self.save_conversations = save_conversations
        self.storage_path = storage_path

        # In-memory storage
        self.conversations: Dict[str, List[ChatMessage]] = {}
        self.conversation_contexts: Dict[str, ConversationContext] = {}

        # Statistics
        self.conversation_stats = {
            "total_conversations": 0,
            "total_messages": 0,
            "average_conversation_length": 0.0
        }

    def create_conversation(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Create a new conversation.

        Args:
            user_id: Optional user identifier
            session_id: Optional session identifier
            system_prompt: Optional system prompt

        Returns:
            Conversation ID
        """
        conversation_id = str(uuid.uuid4())

        # Create context
        context = ConversationContext(
            conversation_id=conversation_id,
            user_id=user_id,
            session_id=session_id
        )

        self.conversation_contexts[conversation_id] = context
        self.conversations[conversation_id] = []

        # Add system prompt if provided
        if system_prompt:
            system_message = ChatMessage(
                role="system",
                content=system_prompt,
                timestamp=time.time()
            )
            self.conversations[conversation_id].append(system_message)

        self.conversation_stats["total_conversations"] += 1

        logger.info(f"Created conversation: {conversation_id}")
        return conversation_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """
        Add a message to conversation.

        Args:
            conversation_id: Conversation identifier
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional metadata

        Returns:
            Created message
        """
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")

        message = ChatMessage(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata or {}
        )

        self.conversations[conversation_id].append(message)

        # Update conversation context
        if conversation_id in self.conversation_contexts:
            self.conversation_contexts[conversation_id].last_updated = time.time()

        # Trim history if too long
        if len(self.conversations[conversation_id]) > self.max_history_length:
            # Keep system messages and trim from oldest user/assistant messages
            system_messages = [msg for msg in self.conversations[conversation_id] if msg.role == "system"]
            other_messages = [msg for msg in self.conversations[conversation_id] if msg.role != "system"]

            # Keep most recent messages
            trimmed_messages = other_messages[-(self.max_history_length - len(system_messages)):]
            self.conversations[conversation_id] = system_messages + trimmed_messages

        self.conversation_stats["total_messages"] += 1
        self._update_conversation_stats()

        # Save if enabled
        if self.save_conversations:
            self._save_conversation(conversation_id)

        return message

    def get_conversation_history(
        self,
        conversation_id: str,
        max_messages: Optional[int] = None,
        include_system: bool = True
    ) -> List[ChatMessage]:
        """
        Get conversation history.

        Args:
            conversation_id: Conversation identifier
            max_messages: Maximum number of messages to return
            include_system: Whether to include system messages

        Returns:
            List of messages
        """
        if conversation_id not in self.conversations:
            return []

        messages = self.conversations[conversation_id]

        if not include_system:
            messages = [msg for msg in messages if msg.role != "system"]

        if max_messages:
            messages = messages[-max_messages:]

        return messages

    def get_conversation_context(self, conversation_id: str) -> Optional[ConversationContext]:
        """Get conversation context."""
        return self.conversation_contexts.get(conversation_id)

    def format_conversation_for_model(
        self,
        conversation_id: str,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Format conversation history for model input.

        Args:
            conversation_id: Conversation identifier
            max_tokens: Maximum tokens to include

        Returns:
            Formatted conversation string
        """
        messages = self.get_conversation_history(conversation_id)

        formatted_parts = []
        total_length = 0
        max_length = max_tokens or self.max_context_tokens

        # Start from most recent and work backwards
        for message in reversed(messages):
            if message.role == "system":
                formatted = f"System: {message.content}"
            elif message.role == "user":
                formatted = f"Użytkownik: {message.content}"
            elif message.role == "assistant":
                formatted = f"Asystent: {message.content}"
            else:
                formatted = f"{message.role}: {message.content}"

            # Rough token estimation (1 token ≈ 4 characters for Polish)
            estimated_tokens = len(formatted) // 4

            if total_length + estimated_tokens > max_length and formatted_parts:
                break

            formatted_parts.insert(0, formatted)
            total_length += estimated_tokens

        return "\n\n".join(formatted_parts)

    def clear_conversation(self, conversation_id: str):
        """Clear conversation history."""
        if conversation_id in self.conversations:
            # Keep system messages
            system_messages = [msg for msg in self.conversations[conversation_id] if msg.role == "system"]
            self.conversations[conversation_id] = system_messages

            logger.info(f"Cleared conversation: {conversation_id}")

    def delete_conversation(self, conversation_id: str):
        """Delete conversation completely."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
        if conversation_id in self.conversation_contexts:
            del self.conversation_contexts[conversation_id]

        logger.info(f"Deleted conversation: {conversation_id}")

    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation summary statistics."""
        if conversation_id not in self.conversations:
            return {}

        messages = self.conversations[conversation_id]
        context = self.conversation_contexts.get(conversation_id)

        user_messages = [msg for msg in messages if msg.role == "user"]
        assistant_messages = [msg for msg in messages if msg.role == "assistant"]

        duration = 0
        if context and messages:
            duration = messages[-1].timestamp - context.created_at

        return {
            "conversation_id": conversation_id,
            "total_messages": len(messages),
            "user_messages": len(user_messages),
            "assistant_messages": len(assistant_messages),
            "duration_seconds": duration,
            "created_at": context.created_at if context else None,
            "last_updated": context.last_updated if context else None,
            "avg_message_length": sum(len(msg.content) for msg in messages) / max(len(messages), 1)
        }

    def _update_conversation_stats(self):
        """Update global conversation statistics."""
        if self.conversations:
            total_messages = sum(len(messages) for messages in self.conversations.values())
            avg_length = total_messages / len(self.conversations)

            self.conversation_stats.update({
                "total_messages": total_messages,
                "average_conversation_length": avg_length
            })

    def _save_conversation(self, conversation_id: str):
        """Save conversation to storage."""
        if not self.storage_path:
            return

        try:
            import os
            os.makedirs(self.storage_path, exist_ok=True)

            conversation_data = {
                "context": asdict(self.conversation_contexts.get(conversation_id, {})),
                "messages": [asdict(msg) for msg in self.conversations[conversation_id]]
            }

            file_path = os.path.join(self.storage_path, f"{conversation_id}.json")
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Failed to save conversation {conversation_id}: {e}")

class ChatBot:
    """
    High-level chatbot interface for WronAI models.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        conversation_manager: Optional[ConversationManager] = None,
        system_prompt: Optional[str] = None,
        personality: str = "helpful"
    ):
        self.inference_engine = inference_engine
        self.conversation_manager = conversation_manager or ConversationManager()
        self.default_system_prompt = system_prompt or self._get_default_system_prompt(personality)
        self.personality = personality

        # Chat statistics
        self.chat_stats = {
            "total_responses": 0,
            "total_response_time": 0.0,
            "average_response_time": 0.0,
            "error_count": 0
        }

    def _get_default_system_prompt(self, personality: str) -> str:
        """Get default system prompt based on personality."""
        prompts = {
            "helpful": "Jesteś pomocnym asystentem AI. Odpowiadaj w języku polskim, bądź uprzejmy i rzeczowy.",
            "friendly": "Jesteś przyjaznym asystentem AI. Używaj ciepłego, przyjacielskiego tonu w języku polskim.",
            "professional": "Jesteś profesjonalnym asystentem AI. Używaj formalnego języka polskiego i precyzyjnych odpowiedzi.",
            "creative": "Jesteś kreatywnym asystentem AI. Bądź twórczy i inspirujący w swoich odpowiedziach po polsku.",
            "educational": "Jesteś asystentem edukacyjnym. Wyjaśniaj pojęcia w sposób zrozumiały, używając przykładów."
        }

        return prompts.get(personality, prompts["helpful"])

    def start_conversation(
        self,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        custom_system_prompt: Optional[str] = None
    ) -> str:
        """
        Start a new conversation.

        Args:
            user_id: Optional user identifier
            session_id: Optional session identifier
            custom_system_prompt: Optional custom system prompt

        Returns:
            Conversation ID
        """
        system_prompt = custom_system_prompt or self.default_system_prompt

        conversation_id = self.conversation_manager.create_conversation(
            user_id=user_id,
            session_id=session_id,
            system_prompt=system_prompt
        )

        return conversation_id

    def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        generation_config: Optional[InferenceConfig] = None
    ) -> Dict[str, Any]:
        """
        Send a message and get response.

        Args:
            message: User message
            conversation_id: Optional conversation ID
            user_id: Optional user ID
            generation_config: Optional generation configuration

        Returns:
            Response data including message and metadata
        """
        start_time = time.time()

        try:
            # Create conversation if needed
            if conversation_id is None:
                conversation_id = self.start_conversation(user_id=user_id)

            # Add user message
            user_message = self.conversation_manager.add_message(
                conversation_id=conversation_id,
                role="user",
                content=message
            )

            # Format conversation for model
            conversation_context = self.conversation_manager.format_conversation_for_model(
                conversation_id=conversation_id
            )

            # Add current message
            full_context = f"{conversation_context}\n\nUżytkownik: {message}\nAsystent:"

            # Generate response
            if generation_config:
                original_config = self.inference_engine.config
                self.inference_engine.config = generation_config

            try:
                response_text = self.inference_engine.generate(full_context)
            finally:
                if generation_config:
                    self.inference_engine.config = original_config

            # Clean response
            response_text = self._clean_response(response_text)

            # Add assistant message
            assistant_message = self.conversation_manager.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content=response_text,
                metadata={
                    "generation_time": time.time() - start_time,
                    "user_message_id": user_message.message_id
                }
            )

            # Update statistics
            response_time = time.time() - start_time
            self._update_chat_stats(response_time)

            return {
                "response": response_text,
                "conversation_id": conversation_id,
                "message_id": assistant_message.message_id,
                "response_time": response_time,
                "metadata": {
                    "user_message_id": user_message.message_id,
                    "timestamp": assistant_message.timestamp,
                    "conversation_length": len(self.conversation_manager.get_conversation_history(conversation_id))
                }
            }

        except Exception as e:
            self.chat_stats["error_count"] += 1
            logger.error(f"Chat error: {e}")

            return {
                "response": "Przepraszam, wystąpił błąd podczas generowania odpowiedzi.",
                "conversation_id": conversation_id,
                "error": str(e),
                "response_time": time.time() - start_time
            }

    def _clean_response(self, response: str) -> str:
        """Clean generated response."""
        # Remove potential conversation context
        if "Użytkownik:" in response:
            response = response.split("Użytkownik:")[0]
        if "Asystent:" in response:
            response = response.split("Asystent:")[-1]

        # Clean up formatting
        response = response.strip()

        # Remove incomplete sentences at the end
        sentences = response.split('.')
        if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
            response = '.'.join(sentences[:-1]) + '.'

        return response

    def _update_chat_stats(self, response_time: float):
        """Update chat statistics."""
        self.chat_stats["total_responses"] += 1
        self.chat_stats["total_response_time"] += response_time
        self.chat_stats["average_response_time"] = (
            self.chat_stats["total_response_time"] / self.chat_stats["total_responses"]
        )

    def get_conversation_history(
        self,
        conversation_id: str,
        format_for_display: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history formatted for display.

        Args:
            conversation_id: Conversation identifier
            format_for_display: Whether to format for display

        Returns:
            Formatted conversation history
        """
        messages = self.conversation_manager.get_conversation_history(
            conversation_id=conversation_id,
            include_system=False
        )

        if not format_for_display:
            return [asdict(msg) for msg in messages]

        formatted_history = []
        for msg in messages:
            formatted_msg = {
                "role": msg.role,
                "content": msg.content,
                "timestamp": datetime.fromtimestamp(msg.timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                "message_id": msg.message_id
            }

            if msg.role == "user":
                formatted_msg["display_name"] = "Ty"
            elif msg.role == "assistant":
                formatted_msg["display_name"] = "🐦‍⬛ WronAI"

            formatted_history.append(formatted_msg)

        return formatted_history

    def clear_conversation(self, conversation_id: str):
        """Clear conversation history."""
        self.conversation_manager.clear_conversation(conversation_id)

    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """Get conversation summary."""
        summary = self.conversation_manager.get_conversation_summary(conversation_id)

        # Add chat-specific metrics
        messages = self.conversation_manager.get_conversation_history(conversation_id)
        assistant_messages = [msg for msg in messages if msg.role == "assistant"]

        if assistant_messages:
            generation_times = [
                msg.metadata.get("generation_time", 0)
                for msg in assistant_messages
                if msg.metadata
            ]

            if generation_times:
                summary["avg_generation_time"] = sum(generation_times) / len(generation_times)
                summary["total_generation_time"] = sum(generation_times)

        return summary

    def set_personality(self, personality: str, conversation_id: Optional[str] = None):
        """
        Change chatbot personality.

        Args:
            personality: New personality type
            conversation_id: Optional conversation to update
        """
        self.personality = personality
        new_system_prompt = self._get_default_system_prompt(personality)

        if conversation_id:
            # Add new system message to existing conversation
            self.conversation_manager.add_message(
                conversation_id=conversation_id,
                role="system",
                content=new_system_prompt
            )
        else:
            # Update default for new conversations
            self.default_system_prompt = new_system_prompt

    def get_chat_statistics(self) -> Dict[str, Any]:
        """Get chatbot statistics."""
        return {
            **self.chat_stats,
            **self.conversation_manager.conversation_stats,
            "personality": self.personality,
            "active_conversations": len(self.conversation_manager.conversations)
        }

    def export_conversation(
        self,
        conversation_id: str,
        format: str = "json"
    ) -> Union[str, Dict[str, Any]]:
        """
        Export conversation in specified format.

        Args:
            conversation_id: Conversation to export
            format: Export format (json, txt, md)

        Returns:
            Exported conversation data
        """
        messages = self.conversation_manager.get_conversation_history(conversation_id)
        context = self.conversation_manager.get_conversation_context(conversation_id)

        if format == "json":
            return {
                "conversation_id": conversation_id,
                "context": asdict(context) if context else {},
                "messages": [asdict(msg) for msg in messages],
                "summary": self.get_conversation_summary(conversation_id)
            }

        elif format == "txt":
            lines = [f"Konwersacja: {conversation_id}"]
            if context:
                lines.append(f"Utworzona: {datetime.fromtimestamp(context.created_at)}")
            lines.append("-" * 50)

            for msg in messages:
                timestamp = datetime.fromtimestamp(msg.timestamp).strftime("%H:%M:%S")
                if msg.role == "user":
                    lines.append(f"[{timestamp}] Ty: {msg.content}")
                elif msg.role == "assistant":
                    lines.append(f"[{timestamp}] WronAI: {msg.content}")
                elif msg.role == "system":
                    lines.append(f"[{timestamp}] System: {msg.content}")
                lines.append("")

            return "\n".join(lines)

        elif format == "md":
            lines = [f"# Konwersacja {conversation_id}"]
            if context:
                lines.append(f"**Utworzona:** {datetime.fromtimestamp(context.created_at)}")
            lines.append("")

            for msg in messages:
                timestamp = datetime.fromtimestamp(msg.timestamp).strftime("%H:%M:%S")
                if msg.role == "user":
                    lines.append(f"**[{timestamp}] Ty:**")
                    lines.append(msg.content)
                elif msg.role == "assistant":
                    lines.append(f"**[{timestamp}] 🐦‍⬛ WronAI:**")
                    lines.append(msg.content)
                elif msg.role == "system":
                    lines.append(f"*[{timestamp}] System: {msg.content}*")
                lines.append("")

            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported export format: {format}")

class AdvancedChatBot(ChatBot):
    """
    Advanced chatbot with additional features.
    """

    def __init__(
        self,
        inference_engine: InferenceEngine,
        conversation_manager: Optional[ConversationManager] = None,
        system_prompt: Optional[str] = None,
        personality: str = "helpful",
        enable_memory: bool = True,
        enable_context_awareness: bool = True
    ):
        super().__init__(inference_engine, conversation_manager, system_prompt, personality)

        self.enable_memory = enable_memory
        self.enable_context_awareness = enable_context_awareness

        # Advanced features
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        self.conversation_topics: Dict[str, List[str]] = {}
        self.response_filters: List[Callable[[str], str]] = []

    def add_response_filter(self, filter_func: Callable[[str], str]):
        """Add a response filter function."""
        self.response_filters.append(filter_func)

    def chat_with_context_awareness(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        context_hints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Chat with enhanced context awareness.

        Args:
            message: User message
            conversation_id: Optional conversation ID
            user_id: Optional user ID
            context_hints: Optional context hints

        Returns:
            Enhanced response with context
        """
        # Extract context if enabled
        if self.enable_context_awareness and conversation_id:
            context = self._extract_conversation_context(conversation_id)
            if context:
                # Enhance message with context
                enhanced_message = self._enhance_message_with_context(message, context)
                message = enhanced_message

        # Use base chat functionality
        response_data = self.chat(
            message=message,
            conversation_id=conversation_id,
            user_id=user_id
        )

        # Apply response filters
        if self.response_filters:
            filtered_response = response_data["response"]
            for filter_func in self.response_filters:
                filtered_response = filter_func(filtered_response)
            response_data["response"] = filtered_response

        # Update topic tracking
        if conversation_id:
            self._update_conversation_topics(conversation_id, message, response_data["response"])

        return response_data

    def _extract_conversation_context(self, conversation_id: str) -> Dict[str, Any]:
        """Extract context from conversation history."""
        messages = self.conversation_manager.get_conversation_history(conversation_id)

        context = {
            "topics": [],
            "user_preferences": {},
            "conversation_style": "neutral",
            "recent_topics": []
        }

        # Extract topics from recent messages
        recent_messages = messages[-10:]  # Last 10 messages
        for msg in recent_messages:
            if msg.role in ["user", "assistant"]:
                # Simple keyword extraction (could be enhanced with NLP)
                words = msg.content.lower().split()
                topics = [word for word in words if len(word) > 5 and word.isalpha()]
                context["recent_topics"].extend(topics[:3])  # Top 3 topics per message

        return context

    def _enhance_message_with_context(self, message: str, context: Dict[str, Any]) -> str:
        """Enhance message with conversation context."""
        # This is a simplified implementation
        # In practice, you might use more sophisticated context injection

        if context.get("recent_topics"):
            # Add subtle context hints
            recent_topics = context["recent_topics"][-3:]  # Last 3 topics
            context_hint = f"Kontekst rozmowy: {', '.join(recent_topics)}. "
            return context_hint + message

        return message

    def _update_conversation_topics(self, conversation_id: str, user_message: str, assistant_response: str):
        """Update topic tracking for conversation."""
        if conversation_id not in self.conversation_topics:
            self.conversation_topics[conversation_id] = []

        # Simple topic extraction (could be enhanced)
        combined_text = f"{user_message} {assistant_response}".lower()
        words = combined_text.split()
        topics = [word for word in words if len(word) > 5 and word.isalpha()]

        # Add unique topics
        existing_topics = set(self.conversation_topics[conversation_id])
        new_topics = [topic for topic in topics if topic not in existing_topics]

        self.conversation_topics[conversation_id].extend(new_topics[:5])  # Max 5 new topics

        # Keep only recent topics
        if len(self.conversation_topics[conversation_id]) > 20:
            self.conversation_topics[conversation_id] = self.conversation_topics[conversation_id][-20:]

    def get_conversation_insights(self, conversation_id: str) -> Dict[str, Any]:
        """Get insights about the conversation."""
        summary = self.get_conversation_summary(conversation_id)
        topics = self.conversation_topics.get(conversation_id, [])

        insights = {
            **summary,
            "topics": topics,
            "topic_count": len(set(topics)),
            "conversation_depth": len(topics) / max(summary.get("total_messages", 1), 1)
        }

        return insights

    def suggest_follow_up_questions(self, conversation_id: str) -> List[str]:
        """Suggest follow-up questions based on conversation."""
        topics = self.conversation_topics.get(conversation_id, [])

        if not topics:
            return [
                "Czy mogę w czymś jeszcze pomóc?",
                "Masz jakieś pytania?",
                "O czym chciałbyś porozmawiać?"
            ]

        # Generate topic-based suggestions
        recent_topics = topics[-3:]
        suggestions = []

        for topic in recent_topics:
            suggestions.extend([
                f"Chcesz dowiedzieć się więcej o {topic}?",
                f"Jak {topic} wpływa na codzienne życie?",
                f"Jakie są najnowsze trendy związane z {topic}?"
            ])

        return suggestions[:5]  # Return top 5 suggestions