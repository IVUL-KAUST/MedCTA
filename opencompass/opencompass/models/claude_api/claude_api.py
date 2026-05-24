from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

from opencompass.registry import MODELS
from opencompass.utils import PromptList

from ..base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class Claude(BaseAPIModel):
    """Model wrapper around Claude API using Anthropic Messages API.

    This version fixes:
    1. deprecated /v1/complete usage
    2. invalid old Claude prompt format
    3. assistant-prefill final-turn error
    4. Claude 3/4 style messages API compatibility
    """

    def __init__(
        self,
        key: str,
        path: str = 'claude-3-5-sonnet-latest',
        query_per_second: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 2,
        temperature: float = 0.0,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            meta_template=meta_template,
            retry=retry,
        )
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                'Import anthropic failed. Please install it with '
                '"pip install anthropic" and try again.'
            )

        self.anthropic = Anthropic(api_key=key)
        self.model = path
        self.temperature = temperature

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
    ) -> List[str]:
        """Generate results given a list of inputs."""
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs, [max_out_len] * len(inputs))
            )
        return results

    def _flatten_to_single_user_message(
        self,
        system_prompt: Optional[str],
        messages: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """Claude rejects assistant-prefill as final turn.
        Convert full history into one user message when needed.
        """
        merged = []
        if system_prompt:
            merged.append(f'SYSTEM:\n{system_prompt}')

        for msg in messages:
            role = msg['role'].upper()
            merged.append(f'{role}:\n{msg["content"]}')

        merged.append('Please provide the next response.')
        return [{'role': 'user', 'content': '\n\n'.join(merged)}]

    def _generate(
        self,
        input: PromptType,
        max_out_len: int = 512,
    ) -> str:
        """Generate result for one input."""
        assert isinstance(input, (str, PromptList))

        system_prompt: Optional[str] = None
        messages: List[Dict[str, str]] = []

        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        else:
            for item in input:
                role = item.get('role')
                prompt = item.get('prompt', '')

                if role == 'SYSTEM':
                    if system_prompt is None:
                        system_prompt = prompt
                    else:
                        system_prompt += '\n' + prompt
                elif role == 'HUMAN':
                    messages.append({'role': 'user', 'content': prompt})
                elif role == 'BOT':
                    messages.append({'role': 'assistant', 'content': prompt})

        if not messages:
            messages = [{'role': 'user', 'content': ''}]

        # Claude Messages API requires the conversation to end with a user turn.
        # If OpenCompass/Lagent gives assistant as the final turn, flatten it.
        if messages[-1]['role'] == 'assistant':
            messages = self._flatten_to_single_user_message(system_prompt, messages)
            system_prompt = None

        num_retries = 0
        while num_retries < self.retry:
            self.wait()
            try:
                kwargs = dict(
                    model=self.model,
                    max_tokens=max_out_len,
                    temperature=self.temperature,
                    messages=messages,
                )
                if system_prompt:
                    kwargs['system'] = system_prompt

                completion = self.anthropic.messages.create(**kwargs)

                text_blocks = []
                for block in completion.content:
                    if getattr(block, 'type', None) == 'text':
                        text_blocks.append(block.text)

                return ''.join(text_blocks).strip()

            except Exception as e:
                self.logger.error('%s', e)

            num_retries += 1

        raise RuntimeError(
            'Calling Claude API failed after retrying for '
            f'{self.retry} times. Check the logs for details.'
        )