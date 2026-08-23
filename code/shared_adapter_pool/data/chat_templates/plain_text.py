"""Pass-through schema transform for plain-text (language-modelling) training.

(migrated from src/llm_pipeline/chat_templates/plain_text.py)
"""
from dataclasses import dataclass
from typing import ClassVar, override

from shared_adapter_pool.data.schema_transforms import SchemaTransform
from shared_adapter_pool.data.schemas import PlainTextMessages, PreparedText


@dataclass
class PlainTextChatTemplate(SchemaTransform[PlainTextMessages, PreparedText]):
    """Identity formatter that copies the document text into a ``PreparedText`` row."""

    transformation: ClassVar[str] = "plain_text_format"

    add_eos: bool = False

    @override
    def apply(self, messages: PlainTextMessages) -> PreparedText:
        return {"text": messages["text"], "add_eos": self.add_eos}

    @override
    def params(self) -> dict:
        return {"add_eos": self.add_eos}
