import torch
from typing import Self


class PromptVector:
    task_name: str = None
    task_weights: torch.Tensor = (None,)
    init_weigts: torch.Tensor = (None,)
    prompt: torch.Tensor = None
    tasks: set = None
    device: str = None

    def __init__(
        self,
        task_name: str,
        task_weights: torch.Tensor = None,
        init_weigts: torch.Tensor = None,
        prompt: torch.Tensor = None,
        device: str = "cuda",
    ):
        if "+" not in task_name and "-" not in task_name:
            self.task_name = f"+ {task_name}"
        else:
            self.task_name = task_name

        if isinstance(prompt, torch.Tensor):
            self.prompt = prompt
        else:
            assert isinstance(task_weights, torch.Tensor) and isinstance(
                init_weigts, torch.Tensor
            )

            self.prompt = task_weights.to(device) - init_weigts.to(device)

        self.tasks = set(task_name.replace("+ ", "").replace("- ", "").split(" "))

    def __add__(self, other: Self) -> Self:
        # print(type(other))
        assert isinstance(other, self.__class__)

        new_prompt = self.prompt + other.prompt
        new_task_name = f"{self.task_name} {other.task_name}"

        return PromptVector(new_task_name, prompt=new_prompt)

    def __radd__(self, other: Self) -> Self:
        if other is None or isinstance(other, int):
            return self

        return self.__add__(other)

    def __sub__(self, other: Self) -> Self:
        assert isinstance(other, self.__class__)
        return self + -other

    def __neg__(self) -> Self:
        if self.task_name[0] == "-":
            new_task_name = f"+{self.task_name[1:]}"
        else:
            new_task_name = f"-{self.task_name[1:]}"

        return PromptVector(new_task_name, prompt=-self.prompt)

    def __str__(self) -> str:
        return f"{self.task_name} {self.prompt}"

    def apply(self, init_weights: torch.Tensor, coef: int = 1) -> torch.nn.Parameter:
        return torch.nn.Parameter(init_weights + coef * self.prompt)
