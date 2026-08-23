"""Schema transform for instruction-following tasks in Alpaca format.

(migrated from src/llm_pipeline/chat_templates/alpaca.py)

Produces ``PreparedCompletion`` rows in the standard Alpaca layout:

    Below is an instruction that describes a task...

    ### Instruction:
    {instruction}

    ### Input:       <- optional
    {input}

    ### Response:
    {output}        <- completion (empty when generating a prompt only)
"""

from dataclasses import dataclass
from typing import ClassVar, override

from shared_adapter_pool.data.schema_transforms import SchemaTransform
from shared_adapter_pool.data.schemas import AlpacaMessages, PreparedCompletion

_ALPACA_SYSTEM = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request."
)


@dataclass
class AlpacaChatTemplate(SchemaTransform[AlpacaMessages, PreparedCompletion]):
    """Formats Alpaca-style instruction samples as prompt/completion pairs.

    When ``output`` is ``None`` the completion is the empty string — useful
    for prompt-only generation flows; completion-only loss masking then sees
    no completion tokens to score.

    Parameters
    ----------
    system_header:
        The preamble printed before the instruction block.
    response_marker:
        The string that precedes the completion.  Defaults to "### Response:".
    """

    transformation: ClassVar[str] = "alpaca_format"

    system_header: str = _ALPACA_SYSTEM
    response_marker: str = "### Response:"

    @override
    def apply(self, messages: AlpacaMessages, /) -> PreparedCompletion:
        instruction_block = f"### Instruction:\n{messages['instruction'].strip()}"

        if messages.get("input"):
            assert messages["input"] is not None
            input_block = f"### Input:\n{messages['input'].strip()}"
            prompt = (
                f"{self.system_header}\n\n"
                f"{instruction_block}\n\n"
                f"{input_block}\n\n"
                f"{self.response_marker}"
            )
        else:
            prompt = (
                f"{self.system_header}\n\n"
                f"{instruction_block}\n\n"
                f"{self.response_marker}"
            )

        output = messages.get("output")
        completion = f"\n{output.strip()}" if output is not None else ""

        return {"prompt": prompt, "completion": completion, "add_eos": False}

    @override
    def params(self) -> dict:
        return {
            "system_header": self.system_header,
            "response_marker": self.response_marker,
        }
