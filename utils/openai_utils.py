# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any
import re
import json
import time
import os
import base64

from openai import OpenAI  # type: ignore
from dotenv import load_dotenv  # type: ignore

dot_file = os.path.join(os.path.dirname(__file__), "../.env")
load_dotenv(dot_file)

openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_text_base_url = os.environ.get("OPENAI_TEXT_BASE_URL")
openai_image_base_url = os.environ.get("OPENAI_IMAGE_BASE_URL") or openai_text_base_url


def _create_openai_client(base_url: str = None):
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY must be set.")

    client_args = {"api_key": openai_api_key}
    if base_url:
        client_args["base_url"] = base_url
    return OpenAI(**client_args)


openai_client = _create_openai_client(openai_text_base_url) if openai_api_key else None
openai_image_client = (
    _create_openai_client(openai_image_base_url) if openai_api_key else None
)


def _require_openai_client(client, client_name: str):
    if client is None:
        raise ValueError(f"OPENAI_API_KEY must be set to use {client_name}.")
    return client


def _aspect_ratio_to_openai_size(aspect_ratio: str) -> str:
    if not aspect_ratio:
        return "auto"

    try:
        width, height = aspect_ratio.split(":", 1)
        width_value = float(width)
        height_value = float(height)
    except (TypeError, ValueError):
        return "auto"

    if width_value <= 0 or height_value <= 0:
        return "auto"
    if width_value == height_value:
        return "1024x1024"
    if width_value > height_value:
        return "1536x1024"
    return "1024x1536"


def parse_openai_json_results(response: str):
    if not response or not isinstance(response, str):
        return None

    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, response, re.DOTALL)

    if not matches:
        json_pattern = r"(\{.*\}|\[.*\])"
        matches = re.findall(json_pattern, response, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            try:
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                json_string_clean = re.sub(r",\s*([\]}])", r"\1", json_string_clean)
                json_string_clean = re.sub(
                    r'\\(?![\\"/bfnrtu])', r"\\\\", json_string_clean
                )

                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue

    try:
        start_idx = response.find("{")
        end_idx = response.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_string = response[start_idx : end_idx + 1]
            json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
            json_string_clean = re.sub(r",\s*([\]}])", r"\1", json_string_clean)
            json_string_clean = re.sub(
                r'\\(?![\\"/bfnrtu])', r"\\\\", json_string_clean
            )
            return json.loads(json_string_clean)
    except Exception:
        pass

    try:
        start_idx = response.find("[")
        end_idx = response.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_string = response[start_idx : end_idx + 1]
            json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
            json_string_clean = re.sub(r",\s*([\]}])", r"\1", json_string_clean)
            json_string_clean = re.sub(
                r'\\(?![\\"/bfnrtu])', r"\\\\", json_string_clean
            )
            return json.loads(json_string_clean)
    except Exception:
        pass

    return None  # No valid JSON found


def call_openai_models_with_content(
    content: list[Any],
    model_name: str,
    max_retries: int = 5,
    base_interval_sec: int = 5,
    result_parsing_func: callable = parse_openai_json_results,
    generation_configs: dict = {},
    check_parsed_response_not_none: bool = True,
    system_prompt: str = None,
):
    raw_response = None
    parsed_response = None

    for attempt in range(max_retries):
        try:
            client = _require_openai_client(openai_client, "OpenAI text models")
            messages = []
            if system_prompt:
                role = "developer" if model_name.startswith(("o1", "o3")) else "system"
                messages.append({"role": role, "content": system_prompt})

            messages.append({"role": "user", "content": content})

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                **generation_configs,
            )
            raw_response = response.choices[0].message.content
            parsed_response = result_parsing_func(raw_response)

            if check_parsed_response_not_none:
                assert (
                    parsed_response is not None
                ), f"Parsed response is None, Here is the raw response:\n{raw_response}"

            return {
                "raw_response": raw_response,
                "parsed_response": parsed_response,
            }

        except Exception as e:
            print(
                f"Warning: OpenAI API call (model={model_name}) failed on attempt {attempt + 1} of {max_retries}. Error: {e}",
            )

            if attempt + 1 == max_retries:
                print(f"Error: All {max_retries} retry attempts failed.")
                raise e

            delay = base_interval_sec * (2 ** (attempt - 1)) if attempt > 0 else 0
            print(f"Retrying in {delay} seconds...")
            if delay > 0:
                time.sleep(delay)

    print("Error: Exited retry loop unexpectedly.")
    return {
        "raw_response": raw_response,
        "parsed_response": parsed_response,
    }


def call_openai_models_with_text_prompt(
    prompt: str,
    model_name: str,
    result_parsing_func: callable = parse_openai_json_results,
    generation_configs: dict = {},
    check_parsed_response_not_none: bool = True,
    max_retries: int = 5,
    base_interval_sec: int = 5,
    system_prompt: str = None,
):
    return call_openai_models_with_content(
        content=[{"type": "text", "text": prompt}],
        model_name=model_name,
        result_parsing_func=result_parsing_func,
        generation_configs=generation_configs,
        check_parsed_response_not_none=check_parsed_response_not_none,
        max_retries=max_retries,
        base_interval_sec=base_interval_sec,
        system_prompt=system_prompt,
    )


def get_openai_image_part_from_path(image_path: str):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    image_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
            },
        }
    ]
    return image_content


def generate_image_with_openai(
    model_name: str,
    prompt: str,
    aspect_ratio: str = "16:9",
    generation_configs: dict = None,
    max_retries: int = 5,
    base_interval_sec: int = 5,
    save_path: str = None,
) -> str:
    """
    Calls the OpenAI Images API for image generation and returns base64.
    """
    if generation_configs is None:
        generation_configs = {}

    gemini_only_config_keys = {
        "system_instruction",
        "response_modalities",
        "image_config",
        "candidate_count",
    }
    image_configs = {
        key: value
        for key, value in generation_configs.items()
        if key not in gemini_only_config_keys
    }
    image_configs.setdefault("size", _aspect_ratio_to_openai_size(aspect_ratio))
    image_configs.setdefault("quality", "auto")
    if model_name.lower().startswith("dall-e"):
        image_configs.setdefault("response_format", "b64_json")

    for attempt in range(max_retries):
        try:
            client = _require_openai_client(openai_image_client, "OpenAI image models")
            response = client.images.generate(
                model=model_name,
                prompt=prompt,
                n=1,
                **image_configs,
            )

            if not response.data:
                raise ValueError("Incomplete response from OpenAI Images API.")

            first_image = response.data[0]
            img_base64 = getattr(first_image, "b64_json", None)
            if not img_base64:
                image_url = getattr(first_image, "url", None)
                if image_url:
                    raise ValueError(
                        "OpenAI-compatible image endpoint returned a URL instead of "
                        "base64 image data. Configure the endpoint to support "
                        'response_format="b64_json" or return b64_json data.'
                    )
                raise ValueError(
                    "OpenAI Images API response did not include b64_json image data."
                )

            if save_path:
                try:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    with open(save_path, "wb") as fh:
                        fh.write(base64.b64decode(img_base64))
                    print(f"Image for prompt '{prompt}' saved to {save_path}")
                except Exception as e:
                    print(f"Warning: Failed to save image to {save_path}: {e}")

            return img_base64
        except Exception as e:
            print(
                f"Warning: OpenAI API image generation (model={model_name}) failed "
                f"on attempt {attempt + 1} of {max_retries}. Error: {e}",
            )

            if attempt + 1 == max_retries:
                print(
                    "Error: All retry attempts failed for OpenAI image generation."
                )
                raise e

            delay = base_interval_sec * (2 ** (attempt - 1)) if attempt > 0 else 0
            print(f"Retrying in {delay} seconds...")
            if delay > 0:
                time.sleep(delay)

    return None
