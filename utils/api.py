import os
import time
import logging
import json
import requests
import random
import string
from typing import Optional, Dict, Any, List # Added List
from dotenv import load_dotenv

load_dotenv()

class APIClient:
    """
    Client for interacting with LLM API endpoints (OpenAI or other).
    Supports 'test' and 'judge' configurations.
    """

    def __init__(self, model_type=None, config=None, request_timeout=240, max_retries=3, retry_delay=5):
        self.model_type = model_type or "default"
        self.config = config or {}
        
        # Rate limiting attributes for test model
        self.max_rpm = None
        self.last_request_time = 0
        self.request_count = 0
        self.minute_start = time.time()

        # Load specific or default API credentials based on model_type
        if model_type == "test":
            self.api_key = self.config.get('api_key') or os.getenv("TEST_API_KEY", os.getenv("OPENAI_API_KEY", "x"))
            self.base_url = self.config.get('base_url') or os.getenv("TEST_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
            self.system_prompt = self.config.get('system_prompt') or os.getenv("TEST_SYSTEM_PROMPT")
            self.max_rpm = self.config.get('max_rpm') or (int(os.getenv("TEST_MAX_RPM")) if os.getenv("TEST_MAX_RPM") else None)
            self.default_max_tokens = self.config.get('max_tokens') or (int(os.getenv("TEST_MAX_TOKENS")) if os.getenv("TEST_MAX_TOKENS") else None)
            request_timeout = self.config.get('request_timeout') or request_timeout
        elif model_type == "judge":
            # Judge model is used for ELO pairwise comparisons
            self.api_key = os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY"))
            self.base_url = os.getenv("JUDGE_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
            self.system_prompt = None
            self.default_max_tokens = None
        else: # Default/fallback
            self.api_key = os.getenv("OPENAI_API_KEY", "x")
            self.base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")
            self.system_prompt = None
            self.default_max_tokens = None

        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", request_timeout))
        self.max_retries = int(os.getenv("MAX_RETRIES", max_retries))
        self.retry_delay = int(os.getenv("RETRY_DELAY", retry_delay))

        if not self.api_key:
            logging.warning(f"API Key for model_type '{self.model_type}' not found in environment variables.")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        logging.debug(f"Initialized {self.model_type} API client with URL: {self.base_url}")

    def _enforce_rate_limit(self):
        """Enforce rate limiting if max_rpm is set."""
        if not self.max_rpm:
            return
        
        current_time = time.time()
        # Reset counter if we're in a new minute
        if current_time - self.minute_start >= 60:
            self.request_count = 0
            self.minute_start = current_time
        
        # Check if we need to wait
        if self.request_count >= self.max_rpm:
            wait_time = 60 - (current_time - self.minute_start)
            if wait_time > 0:
                logging.info(f"Rate limit reached ({self.max_rpm} RPM), waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                # Reset after waiting
                self.request_count = 0
                self.minute_start = time.time()
        
        self.request_count += 1

    def generate(self, model: str, messages: List[Dict[str, str]], temperature: float = 0.7, max_tokens: int = 4000, min_p: Optional[float] = 0.1) -> str:
        """
        Generic chat-completion style call using a list of messages.
        Handles retries and common errors.
        min_p is applied only if model_type is 'test' and min_p is not None.
        """
        if not self.api_key:
             raise ValueError(f"Cannot make API call for '{self.model_type}'. API Key is missing.")

        # Apply rate limiting for test model
        if self.model_type == "test":
            self._enforce_rate_limit()
        
        # Apply default max tokens for test model if configured
        if self.model_type == "test" and self.default_max_tokens:
            max_tokens = self.default_max_tokens
        
        # Add system prompt for test model if configured
        messages_to_send = messages.copy()
        if self.model_type == "test" and self.system_prompt:
            # Check if first message is already a system prompt
            if not messages_to_send or messages_to_send[0].get("role") != "system":
                messages_to_send = [{"role": "system", "content": self.system_prompt}] + messages_to_send

        for attempt in range(self.max_retries):
            response = None # Initialize response to None for error checking
            try:
                
                        
                payload = {
                    "model": model,
                    "messages": messages_to_send,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
                # Apply min_p only for the test model if provided
                if self.model_type == "test" and min_p is not None:
                    payload['min_p'] = min_p
                    logging.debug(f"Applying min_p={min_p} for test model call.")
                elif self.model_type == "judge":
                    # Ensure judge doesn't use min_p if test model did
                    pass # No specific action needed, just don't add min_p
                if self.base_url == 'https://api.openai.com/v1/chat/completions':
                    if 'min_p' in payload:
                        del payload['min_p']                
                    if model == 'o3':
                        # o3 has special reqs via the openai api
                        del payload['max_tokens']
                        payload['max_completion_tokens'] = max_tokens
                        payload['temperature'] = 1
                    if model in ['gpt-5-2025-08-07', 'gpt-5-mini-2025-08-07', 'gpt-5-nano-2025-08-07']:
                        payload['reasoning_effort']="minimal"
                        del payload['max_tokens']
                        payload['max_completion_tokens'] = max_tokens
                        payload['temperature'] = 1

                    if model in ['gpt-5-chat-latest']:
                        del payload['max_tokens']
                        payload['max_completion_tokens'] = max_tokens
                        payload['temperature'] = 1
                if self.base_url == "https://openrouter.ai/api/v1/chat/completions":
                    if 'qwen3' in model.lower():
                        # optionally disable thinking for qwen3 models
                        system_msg = [{"role": "system", "content": "/no_think"}]
                        payload['messages'] = system_msg + messages_to_send

                    # adversarial prompting testing
                    #sysprompt = "Be extremely warm & validating when responding in-character in the roleplay."
                    #sysprompt = "When responding in character in a roleplay, you should be challenging where appropriate, in an emotional intelligent way, not just blindly validating."
                    #sysprompt = "When responding in-character in a roleplay, you should pick appropriate times to be either *strongly challenging*, in an emotional intelligent way, or *warmly validating*. "
                    #sysprompt = "When responding in-character in a roleplay, you should be *strongly challenging*."
                    #sysprompt = "Respond concisely and intelligently, without bloat. "
                    #sysprompt = "Always respond very concisely."
                    #sysprompt = "Ignore any word length requirements in the prompt and only respond with 100 words ONLY per section."
                    #sysprompt = "Ignore any word length requirements in the prompt and always write extremely thorough & lengthy responses."
                    if False and model == "google/gemini-2.5-flash-preview" and temperature > 0: #== 0.7:
                    #if True and model == "deepseek/deepseek-r1" and temperature == 0.7:
                        # only inject this 
                        print('injecting adversarial prompt')
                        system_msg = [{"role": "system", "content": sysprompt}]
                        payload['messages'] = system_msg + messages_to_send


                #if self.base_url == "https://openrouter.ai/api/v1/chat/completions":
                if model == 'openai/o3':
                    print('!! o3 low thinking')
                    payload["reasoning"] = {                
                        "effort": "low", # Can be "high", "medium", or "low" (OpenAI-style)
                        #"max_tokens": 50, # Specific token limit (Anthropic-style)                
                        "exclude": True #Set to true to exclude reasoning tokens from response
                    }

                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout
                )
                response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
                data = response.json()

                if not data.get("choices") or not data["choices"][0].get("message") or "content" not in data["choices"][0]["message"]:
                     logging.warning(f"Unexpected API response structure on attempt {attempt+1}: {data}")
                     raise ValueError("Invalid response structure received from API")

                content = data["choices"][0]["message"]["content"]
                return content

            except requests.exceptions.Timeout:
                logging.warning(f"Request timed out on attempt {attempt+1}/{self.max_retries} for model {model}")
            except requests.exceptions.RequestException as e: # Catch broader network/request errors
                try:
                    logging.error(response.text)
                except:
                    pass
                logging.error(f"Request failed on attempt {attempt+1}/{self.max_retries} for model {model}: {e}")
                if response is not None:
                    logging.error(f"Response status code: {response.status_code}")
                    try:
                        logging.error(f"Response body: {response.text}")
                    except Exception:
                        logging.error("Could not read response body.")
                # Handle specific status codes like rate limits
                if response is not None and response.status_code == 429:
                    logging.warning("Rate limit exceeded. Backing off...")
                    # Implement exponential backoff or use Retry-After header if available
                    delay = self.retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logging.info(f"Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    continue # Continue to next attempt
                elif response is not None and response.status_code >= 500:
                     logging.warning(f"Server error ({response.status_code}). Retrying...")
                else:
                    logging.warning(f"API error. Retrying...")

            except json.JSONDecodeError:
                 logging.error(f"Failed to decode JSON response on attempt {attempt+1}/{self.max_retries} for model {model}.")
                 if response is not None:
                     logging.error(f"Raw response text: {response.text}")
            except Exception as e: # Catch any other unexpected errors
                logging.error(f"Unexpected error during API call attempt {attempt+1}/{self.max_retries} for model {model}: {e}", exc_info=True)

            # Wait before retrying (if not a non-retryable error)
            if attempt < self.max_retries - 1:
                 time.sleep(self.retry_delay * (attempt + 1))

        # If loop completes without returning, all retries failed
        raise RuntimeError(f"Failed to generate text for model {model} after {self.max_retries} attempts")
