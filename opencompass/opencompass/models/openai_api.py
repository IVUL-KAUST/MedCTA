import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union

import jieba
import requests

from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

PromptType = Union[PromptList, str]
OPENAI_API_BASE = 'https://api.openai.com/v1/chat/completions'


@MODELS.register_module()
class OpenAI(BaseAPIModel):
    is_api: bool = True

    def __init__(self,
                 path: str = 'gpt-3.5-turbo',
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 key: Union[str, List[str]] = 'ENV',
                 org: Optional[Union[str, List[str]]] = None,
                 meta_template: Optional[Dict] = None,
                 openai_api_base: str = OPENAI_API_BASE,
                 mode: str = 'none',
                 temperature: Optional[float] = None,
                 **gen_params):

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         retry=retry)
        import tiktoken
        self.tiktoken = tiktoken
        self.temperature = temperature
        assert mode in ['none', 'front', 'mid', 'rear']
        self.mode = mode
        self.gen_params = gen_params

        if isinstance(key, str):
            key = os.getenv('OPENAI_API_KEY') if key == 'ENV' else key
            self.keys = [os.getenv('ALLES_API_KEY') if key == 'alles' else key]
        else:
            self.keys = key

        self.invalid_keys = set()
        self.key_ctr = 0

        if isinstance(org, str):
            self.orgs = [org]
        else:
            self.orgs = org
        self.org_ctr = 0

        self.url = openai_api_base
        self.path = path
        self._key_lock = Lock()
        self._org_lock = Lock()

    def _uses_max_completion_tokens(self) -> bool:
        model_name = (self.path or '').lower()
        # Safe default for newer GPT-5 family.
        return model_name.startswith('gpt-5')

    def generate(
        self,
        inputs: List[str or PromptList],
        max_out_len: int = 512,
        temperature: float = 0.7,
    ) -> List[str]:
        if self.temperature is not None:
            temperature = self.temperature

        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [max_out_len] * len(inputs),
                             [temperature] * len(inputs)))
        return results

    def _generate(self, input: str or PromptList, max_out_len: int,
                  temperature: float) -> str:
        assert isinstance(input, (str, PromptList))

        context_window = 4096
        if '32k' in self.path:
            context_window = 32768
        elif '16k' in self.path:
            context_window = 16384
        elif 'gpt-4' in self.path:
            context_window = 8192
        elif 'gpt-5' in self.path:
            context_window = self.max_seq_len

        if isinstance(input, str) and self.mode != 'none':
            context_window = self.max_seq_len
            input = self.bin_trim(input, context_window - 100 - max_out_len)

        if isinstance(input, str):
            messages = [{'role': 'user', 'content': input}]
        else:
            messages = []
            for item in input:
                msg = {'content': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                messages.append(msg)

        max_out_len = min(
            max_out_len, context_window - self.get_token_len(str(input)) - 100)
        if max_out_len <= 0:
            return ''

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.wait()

            with self._key_lock:
                if len(self.invalid_keys) == len(self.keys):
                    raise RuntimeError('All keys have insufficient quota.')

                while True:
                    self.key_ctr = (self.key_ctr + 1) % len(self.keys)
                    if self.keys[self.key_ctr] not in self.invalid_keys:
                        break

                key = self.keys[self.key_ctr]

            headers = {
                'Authorization': f'Bearer {key}',
                'Content-Type': 'application/json',
            }

            if self.orgs:
                with self._org_lock:
                    self.org_ctr = (self.org_ctr + 1) % len(self.orgs)
                    org = self.orgs[self.org_ctr]
                headers['OpenAI-Organization'] = org

            try:
                data = {
                    'model': self.path,
                    'messages': messages,
                    'n': 1,
                    'stop': None,
                    'temperature': temperature,
                }

                if self._uses_max_completion_tokens():
                    data['max_completion_tokens'] = max_out_len
                else:
                    data['max_tokens'] = max_out_len

                data = {**data, **self.gen_params}

                raw_response = requests.post(
                    self.url,
                    headers=headers,
                    data=json.dumps(data),
                    timeout=120,
                )
            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                max_num_retries += 1
                continue

            try:
                response = raw_response.json()
            except requests.JSONDecodeError:
                self.logger.error('JSON decode error, got %s',
                                  raw_response.text[:1000])
                max_num_retries += 1
                continue

            response = response.get('data', response)

            if 'choices' in response:
                message = response['choices'][0].get('message', {})
                content = message.get('content', '')

                if isinstance(content, str):
                    return content.strip()

                # Some APIs may return structured content parts.
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            text_parts.append(part.get('text', ''))
                    return ''.join(text_parts).strip()

                return str(content).strip()

            if 'error' in response:
                err = response['error']
                code = err.get('code')
                msg = err.get('message', '')

                if code == 'rate_limit_exceeded':
                    time.sleep(1)
                    self.logger.warning('Rate limit exceeded, retrying...')
                    max_num_retries += 1
                    continue

                if code == 'insufficient_quota':
                    self.invalid_keys.add(key)
                    self.logger.warning('insufficient_quota key: %s', key)
                    max_num_retries += 1
                    continue

                # Fallback: if model rejected max_tokens, retry once with max_completion_tokens
                if ('max_tokens' in msg and 'max_completion_tokens' in msg
                        and 'max_tokens' in data):
                    self.logger.warning(
                        'Model rejected max_tokens; retrying with max_completion_tokens.')
                    self.gen_params.pop('max_tokens', None)
                    max_num_retries += 1
                    time.sleep(1)
                    continue

                self.logger.error('Error response from OpenAI API: %s', err)
                max_num_retries += 1
                continue

            self.logger.error('Unexpected response format: %s', response)
            max_num_retries += 1

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for details.')

    def get_token_len(self, prompt: str) -> int:
        if self.path in self.tiktoken.model.MODEL_TO_ENCODING:
            enc = self.tiktoken.encoding_for_model(self.path)
        else:
            enc = self.tiktoken.encoding_for_model('gpt-4')
        return len(enc.encode(prompt))

    def bin_trim(self, prompt: str, num_token: int) -> str:
        token_len = self.get_token_len(prompt)
        if token_len <= num_token:
            return prompt
        pattern = re.compile(r'[\u4e00-\u9fa5]')
        if pattern.search(prompt):
            words = list(jieba.cut(prompt, cut_all=False))
            sep = ''
        else:
            words = prompt.split(' ')
            sep = ' '

        l, r = 1, len(words)
        while l + 2 < r:
            mid = (l + r) // 2
            if self.mode == 'front':
                cur_prompt = sep.join(words[-mid:])
            elif self.mode == 'mid':
                cur_prompt = sep.join(words[:mid]) + sep.join(words[-mid:])
            elif self.mode == 'rear':
                cur_prompt = sep.join(words[:mid])

            if self.get_token_len(cur_prompt) <= num_token:
                l = mid
            else:
                r = mid

        if self.mode == 'front':
            prompt = sep.join(words[-l:])
        elif self.mode == 'mid':
            prompt = sep.join(words[:l]) + sep.join(words[-l:])
        elif self.mode == 'rear':
            prompt = sep.join(words[:l])
        return prompt