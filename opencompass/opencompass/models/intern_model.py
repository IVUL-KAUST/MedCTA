from __future__ import annotations

import time
from typing import Dict, List, Optional

import requests

from opencompass.models.base import BaseModel, LMTemplateParser
from opencompass.registry import MODELS


@MODELS.register_module()
class InternLM(BaseModel):
    is_api: bool = True

    def __init__(self,
                 path: str,
                 openai_api_base: str,
                 key: str = 'EMPTY',
                 query_per_second: float = 1,
                 max_seq_len: int = 4096,
                 meta_template: Optional[Dict] = None,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 timeout: int = 300,
                 max_out_len: int = 1024):
        self.path = path
        self.key = key
        self.query_per_second = max(query_per_second, 1e-6)
        self.max_seq_len = max_seq_len
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.default_max_out_len = max_out_len

        self.template_parser = LMTemplateParser(meta_template)
        self.eos_token_id = None
        if meta_template and 'eos_token_id' in meta_template:
            self.eos_token_id = meta_template['eos_token_id']

        base = openai_api_base.rstrip('/')
        self.api_base = base
        self.chat_url = self.api_base

        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.key}',
        }

        self._min_interval = 1.0 / self.query_per_second
        self._last_request_time = 0.0

    def _throttle(self):
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _post(self, url: str, payload: Dict) -> Dict:
        self._throttle()
        resp = requests.post(
            url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
        )
        try:
            resp.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(
                f'InternLM API request failed: {resp.status_code} {resp.text}'
            ) from e

        try:
            return resp.json()
        except Exception as e:
            raise RuntimeError(
                f'InternLM API returned non-JSON response: {resp.text}'
            ) from e

    def _build_messages(self, prompt: str) -> List[Dict[str, str]]:
        return [{'role': 'user', 'content': prompt}]

    def get_token_len(self, prompt: str) -> int:
        return max(1, len(prompt.split()))

    def generate(self,
                 inputs: List[str],
                 max_out_len: Optional[int] = None) -> List[str]:
        outputs = []
        max_tokens = max_out_len if max_out_len is not None else self.default_max_out_len

        for prompt in inputs:
            payload = {
                'model': self.path,
                'messages': self._build_messages(prompt),
                'max_tokens': max_tokens,
                'temperature': self.temperature,
                'top_p': self.top_p,
            }

            data = self._post(self.chat_url, payload)

            try:
                text = data['choices'][0]['message']['content']
            except (KeyError, IndexError, TypeError) as e:
                raise RuntimeError(
                    f'Unexpected response format from InternLM API: {data}'
                ) from e

            outputs.append(text)

        return outputs

    def get_ppl(self,
                input_texts: List[str],
                mask_length: Optional[List[int]] = None) -> List[float]:
        raise NotImplementedError(
            'get_ppl is not supported by this API-backed InternLM wrapper.'
        )