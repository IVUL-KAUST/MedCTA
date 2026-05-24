# flake8: noqa: E501
import json
import time
from typing import Dict, List, Optional, Tuple, Union

import requests

from opencompass.utils.prompt import PromptList
from .base_api import BaseAPIModel

PromptType = Union[PromptList, str, float]


class Gemini(BaseAPIModel):
    """Model wrapper around Gemini generateContent API for OpenCompass + Lagent."""

    def __init__(
        self,
        key: str,
        path: str,
        query_per_second: int = 1,
        max_seq_len: int = 2048,
        meta_template: Optional[Dict] = None,
        retry: int = 4,
        temperature: float = 0.0,
        top_p: float = 0.8,
        top_k: int = 10,
        timeout: int = 90,
        thinking_level: str = 'high',
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            query_per_second=query_per_second,
            meta_template=meta_template,
            retry=retry,
        )
        self.api_key = key
        self.url = (
            f'https://generativelanguage.googleapis.com/v1beta/'
            f'models/{path}:generateContent?key={key}'
        )
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.timeout = timeout
        self.thinking_level = thinking_level
        self.headers = {'content-type': 'application/json'}
        self.session = requests.Session()

    def generate(
        self,
        inputs: List[PromptType],
        max_out_len: int = 512,
    ) -> List[str]:
        results = []
        for x in inputs:
            results.append(self._generate(x, max_out_len))
        self.flush()
        return results

    def _build_messages(
        self,
        input: PromptType,
    ) -> Tuple[Optional[str], List[Dict]]:
        assert isinstance(input, (str, PromptList))

        if isinstance(input, str):
            return None, [{'role': 'user', 'parts': [{'text': input}]}]

        messages: List[Dict] = []
        system_prompt = None

        for item in input:
            if item['role'] == 'SYSTEM':
                system_prompt = item['prompt']
                break

        for item in input:
            role = item['role']
            if role == 'SYSTEM':
                continue

            text = item['prompt']
            msg = {'parts': [{'text': text}]}

            if role == 'HUMAN':
                msg['role'] = 'user'
            elif role == 'BOT':
                msg['role'] = 'model'
            else:
                continue

            messages.append(msg)

        return system_prompt, messages

    def _make_payload(
        self,
        messages: List[Dict],
        max_out_len: int,
        system_prompt: Optional[str] = None,
    ) -> Dict:
        payload = {
            'contents': messages,
            'safetySettings': [
                {
                    'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                    'threshold': 'BLOCK_NONE'
                },
                {
                    'category': 'HARM_CATEGORY_HATE_SPEECH',
                    'threshold': 'BLOCK_NONE'
                },
                {
                    'category': 'HARM_CATEGORY_HARASSMENT',
                    'threshold': 'BLOCK_NONE'
                },
                {
                    'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    'threshold': 'BLOCK_NONE'
                },
            ],
            'generationConfig': {
                'candidateCount': 1,
                'temperature': self.temperature,
                'maxOutputTokens': max_out_len,
                'topP': self.top_p,
                'topK': self.top_k,
                'thinkingConfig': {
                    'thinkingLevel': self.thinking_level
                }
            }
        }

        if system_prompt:
            payload['systemInstruction'] = {
                'parts': [{'text': system_prompt}]
            }

        return payload

    def _extract_text(self, response: Dict) -> Optional[str]:
        candidates = response.get('candidates', [])
        if not candidates:
            return None

        content = candidates[0].get('content', {})
        parts = content.get('parts', [])
        texts = [p.get('text') for p in parts if 'text' in p]

        if not texts:
            return None

        return '\n'.join(t for t in texts if t is not None).strip()

    def _get_first_candidate(self, response: Dict) -> Dict:
        candidates = response.get('candidates', [])
        if not candidates:
            return {}
        return candidates[0]

    def _get_finish_reason(self, response: Dict) -> str:
        cand = self._get_first_candidate(response)
        return cand.get('finishReason', 'UNKNOWN')

    def _recover_malformed_function_call(self, response: Dict) -> Optional[str]:
        """Recover textual tool call from Gemini MALFORMED_FUNCTION_CALL."""
        cand = self._get_first_candidate(response)
        finish_reason = cand.get('finishReason')
        finish_message = cand.get('finishMessage', '')

        if finish_reason != 'MALFORMED_FUNCTION_CALL':
            return None

        if not finish_message:
            return None

        prefix = 'Malformed function call:'
        if finish_message.startswith(prefix):
            recovered = finish_message[len(prefix):].strip()
        else:
            recovered = finish_message.strip()

        if recovered:
            return recovered

        return None

    def _generate(
        self,
        input: PromptType,
        max_out_len: int = 512,
    ) -> str:
        system_prompt, messages = self._build_messages(input)
        data = self._make_payload(messages, max_out_len, system_prompt)

        last_error = None
        server_error_count = 0

        for attempt in range(self.retry):
            self.wait()

            try:
                raw_response = self.session.post(
                    self.url,
                    headers=self.headers,
                    json=data,
                    timeout=(10, self.timeout),
                )
            except requests.Timeout as e:
                last_error = f'timeout: {e}'
                self.logger.error(last_error)
                time.sleep(min(15, 2 ** attempt))
                continue
            except requests.RequestException as e:
                last_error = f'request failed: {e}'
                self.logger.error(last_error)
                time.sleep(min(10, 2 ** attempt))
                continue

            try:
                response = raw_response.json()
            except ValueError:
                last_error = f'non-json response: {raw_response.text[:1000]}'
                self.logger.error(last_error)
                time.sleep(min(10, 2 ** attempt))
                continue

            if raw_response.status_code == 200:
                text = self._extract_text(response)
                if text is not None and text != '':
                    return text

                recovered_call = self._recover_malformed_function_call(response)
                if recovered_call is not None:
                    self.logger.warning(
                        'Recovered MALFORMED_FUNCTION_CALL as text: %s',
                        recovered_call[:1000]
                    )
                    return recovered_call

                prompt_feedback = response.get('promptFeedback')
                if prompt_feedback:
                    block_reason = prompt_feedback.get('blockReason', 'UNKNOWN')
                    return f'[Gemini blocked prompt: {block_reason}]'

                finish_reason = self._get_finish_reason(response)
                last_error = (
                    f'empty Gemini response with finishReason={finish_reason}: '
                    f'{json.dumps(response)[:2000]}'
                )
                self.logger.warning(last_error)

                if attempt < 1:
                    time.sleep(3)
                    continue

                return f'[Gemini empty response: finishReason={finish_reason}]'

            error_obj = response.get('error', {})
            msg = error_obj.get('message', str(response))
            code = error_obj.get('code', raw_response.status_code)
            status = error_obj.get('status', 'UNKNOWN')
            last_error = f'Gemini API error {code} {status}: {msg}'
            self.logger.error(last_error)

            if code in (500, 502, 503, 504):
                server_error_count += 1
                if server_error_count >= 3:
                    self.logger.error(
                        'Too many backend server errors, skipping this sample: %s',
                        last_error
                    )
                    return f'[Gemini backend failure: {last_error}]'
                time.sleep(min(8, 2 ** attempt))
                continue

            if code == 429:
                time.sleep(min(20, 2 ** attempt))
                continue

            return f'[Gemini non-retryable error: {last_error}]'

        self.logger.error('Giving up after retries: %s', last_error)
        return f'[Gemini request failed after retries: {last_error}]'

    def __del__(self):
        try:
            self.session.close()
        except Exception:
            pass