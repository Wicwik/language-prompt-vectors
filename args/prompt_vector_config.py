from dataclasses import field, dataclass
from typing import List

from peft import PromptTuningConfig

@dataclass
class PromptVectorConfig(PromptTuningConfig):
    init_prompts: List[str] = field(
        default=None, metadata={"help": "Initialization prompt paths."}
    )
