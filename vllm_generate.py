"""
vLLM-backed generate_fn for beam search.

Creates a generate_fn(prefix: str, K: int) -> list[str] that calls vLLM's
OpenAI-compatible API to generate K HTML element continuations from a given
prefix. Each "element" is one segment of HTML split at '<' boundaries.

The generate_fn handles the prompt formatting internally (image + instruction)
so the beam search module only needs to pass the current HTML prefix.

Usage:
    from utils.vllm_generate import create_vllm_generate_fn

    generate_fn = create_vllm_generate_fn(
        base_url="http://localhost:8888",
        model_name="tcz/qwen3-vl-8b-box-layouts-inline-sft-900",
        image_b64=image_base64_string,
        instruction="Generate the HTML markup with inline styles...",
    )
    continuations = generate_fn("", 3)      # first step: 3 branches
    continuations = generate_fn(html, 3)    # subsequent steps
"""

import base64
import io
import re
from typing import Callable

from PIL import Image


def pil_to_base64(img: Image.Image) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


def extract_next_segment(text: str) -> str:
    """Extract the first HTML segment from generated text.

    A segment is everything up to (but not including) the second '<'.
    This naturally corresponds to one HTML tag boundary:
      '<div style="...">' → full segment
      '</div>' → full segment
      '<div style="...">text' → includes trailing text

    Args:
        text: Raw generated text from the model.

    Returns:
        The first segment, or empty string if text is empty.
    """
    if not text:
        return ""
    first_lt = text.find('<')
    if first_lt < 0:
        return text
    second_lt = text.find('<', first_lt + 1)
    if second_lt < 0:
        return text
    return text[:second_lt]


def _extract_first_element_from_logprobs(logprobs_content) -> str:
    """Extract the first HTML element from token-level logprobs.

    Walks through tokens, accumulating text. When a token contains '<'
    and the accumulated text already has a '<' (i.e., we've started an
    element), that token starts the NEXT element — discard it and stop.

    If '<' appears mid-token, the entire token is discarded to avoid
    injecting partial tokens the model didn't intend.

    Args:
        logprobs_content: List of token logprob objects from the API response.
            Each has a .token attribute with the decoded token string.

    Returns:
        The first HTML element, or empty string if none found.
    """
    if not logprobs_content:
        return ""

    accumulated = ""
    for token_info in logprobs_content:
        token_text = token_info.token
        if "<" in token_text and "<" in accumulated:
            # Already have our element, this token starts the next one
            break
        accumulated += token_text

    return accumulated


def create_vllm_generate_fn(
    base_url: str,
    model_name: str,
    image_b64: str,
    instruction: str,
    temperature: float = 1.0,
    max_tokens: int = 256,
) -> Callable[[str, int], list[str]]:
    """Create a generate_fn for beam search backed by vLLM.

    The returned function has signature (prefix: str, K: int) -> list[str]:
        - prefix: Current HTML generated so far (empty for first step)
        - K: Number of continuations to generate
        - Returns: K continuation strings. Empty string = terminal (EOS).

    Generation strategy:
        - First step (prefix=""): Generate freely, extract first element
        - Subsequent steps: Continue from prefix using continue_final_message.
          No tokens are injected — the model continues from the natural token
          boundary at the end of the prefix. Logprobs are used to walk tokens
          and stop at the first token containing '<' after the element has
          started, ensuring clean token boundaries.

    Args:
        base_url: vLLM server URL (e.g. "http://localhost:8888")
        model_name: Model name served by vLLM
        image_b64: Base64-encoded reference image (PNG)
        instruction: Text instruction for the model
        temperature: Sampling temperature (higher = more diverse branches)
        max_tokens: Max tokens per generation step
    """
    from openai import OpenAI

    client = OpenAI(base_url=f"{base_url}/v1", api_key="unused")

    user_message = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": instruction},
        ],
    }

    def generate(prefix: str, K: int) -> list[str]:
        if not prefix:
            # First step: generate freely and extract the first element
            response = client.chat.completions.create(
                model=model_name,
                messages=[user_message],
                n=K,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            results = []
            for choice in response.choices:
                text = choice.message.content or ""
                segment = extract_next_segment(text)
                results.append(segment if segment.strip() else "")
            return results

        else:
            # Subsequent steps: continue from prefix naturally (no injection)
            messages = [
                user_message,
                {"role": "assistant", "content": prefix},
            ]

            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                n=K,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=True,
                extra_body={
                    "continue_final_message": True,
                    "add_generation_prompt": False,
                },
            )

            results = []
            for choice in response.choices:
                if choice.logprobs and choice.logprobs.content:
                    segment = _extract_first_element_from_logprobs(
                        choice.logprobs.content
                    )
                else:
                    # Fallback: no logprobs, use text-level extraction
                    text = choice.message.content or ""
                    if text.startswith(prefix):
                        text = text[len(prefix):]
                    segment = extract_next_segment(text)

                if not segment.strip():
                    results.append("")  # terminal (EOS or whitespace only)
                else:
                    results.append(segment)
            return results

    return generate


def create_vllm_generate_fn_from_pil(
    base_url: str,
    model_name: str,
    image: Image.Image,
    instruction: str = "Generate the HTML markup with inline styles that produces this webpage layout.",
    **kwargs,
) -> Callable[[str, int], list[str]]:
    """Convenience wrapper that accepts a PIL Image instead of base64 string."""
    image_b64 = pil_to_base64(image)
    return create_vllm_generate_fn(
        base_url=base_url,
        model_name=model_name,
        image_b64=image_b64,
        instruction=instruction,
        **kwargs,
    )
