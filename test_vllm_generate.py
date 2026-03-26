"""Tests for vllm_generate.py — uses mocks (no GPU needed)."""

import unittest
from unittest.mock import MagicMock, patch

from utils.vllm_generate import (
    extract_next_segment,
    _extract_first_element_from_logprobs,
    create_vllm_generate_fn,
)


class TestExtractNextSegment(unittest.TestCase):
    """Test HTML segment extraction."""

    def test_empty_string(self):
        self.assertEqual(extract_next_segment(""), "")

    def test_single_element(self):
        self.assertEqual(
            extract_next_segment('<div style="background:red;">'),
            '<div style="background:red;">'
        )

    def test_two_elements(self):
        self.assertEqual(
            extract_next_segment('<div style="display:flex;"><div style="background:red;">'),
            '<div style="display:flex;">'
        )

    def test_closing_tag(self):
        self.assertEqual(
            extract_next_segment('</div><div>'),
            '</div>'
        )

    def test_self_closing_with_content(self):
        self.assertEqual(
            extract_next_segment('<div style="background:red;"></div><div>more</div>'),
            '<div style="background:red;">'
        )

    def test_no_angle_brackets(self):
        self.assertEqual(extract_next_segment("just text"), "just text")

    def test_text_before_element(self):
        # Only one '<', so it's one segment
        self.assertEqual(extract_next_segment("text<div>"), "text<div>")

    def test_closing_tags_chain(self):
        self.assertEqual(
            extract_next_segment('</div></div></div>'),
            '</div>'
        )


def _make_token(text):
    """Create a mock token logprob object."""
    t = MagicMock()
    t.token = text
    return t


class TestExtractFirstElementFromLogprobs(unittest.TestCase):
    """Test token-level element extraction."""

    def test_empty_logprobs(self):
        self.assertEqual(_extract_first_element_from_logprobs([]), "")
        self.assertEqual(_extract_first_element_from_logprobs(None), "")

    def test_single_element_natural_tokens(self):
        # Tokens: <div  style="  background:red;  ">  \n
        tokens = [_make_token(t) for t in ['<div', ' style="', 'background:red;', '">', '\n']]
        result = _extract_first_element_from_logprobs(tokens)
        self.assertEqual(result, '<div style="background:red;">\n')

    def test_two_elements_stops_at_second(self):
        # First element: <div style="flex;">  then second: <div style="bg:red;">
        tokens = [_make_token(t) for t in [
            '<div', ' style="', 'flex;', '">\n',  # first element
            '<div', ' style="', 'bg:red;', '">',  # second element — discard from '<div'
        ]]
        result = _extract_first_element_from_logprobs(tokens)
        self.assertEqual(result, '<div style="flex;">\n')

    def test_closing_tag(self):
        tokens = [_make_token(t) for t in ['</div', '>\n']]
        result = _extract_first_element_from_logprobs(tokens)
        self.assertEqual(result, '</div>\n')

    def test_closing_then_opening(self):
        tokens = [_make_token(t) for t in ['</div', '>\n', '<div', ' style="bg:blue;">']]
        result = _extract_first_element_from_logprobs(tokens)
        self.assertEqual(result, '</div>\n')

    def test_whitespace_before_element(self):
        tokens = [_make_token(t) for t in ['\n', '<div', ' style="x"', '>\n', '<span']]
        result = _extract_first_element_from_logprobs(tokens)
        self.assertEqual(result, '\n<div style="x">\n')

    def test_lt_mid_token_discarded(self):
        # Token ">\n<" has '<' but accumulated already has '<' from first token
        tokens = [_make_token(t) for t in ['<div', ' style="x"', '>\n<', 'span>']]
        result = _extract_first_element_from_logprobs(tokens)
        # ">\n<" token is discarded entirely because accumulated already has '<'
        self.assertEqual(result, '<div style="x"')

    def test_no_lt_at_all(self):
        # EOS or pure text without any '<'
        tokens = [_make_token(t) for t in ['hello', ' world']]
        result = _extract_first_element_from_logprobs(tokens)
        self.assertEqual(result, 'hello world')

    def test_single_token_with_full_element(self):
        # Rare: entire element in one token
        tokens = [_make_token(t) for t in ['<div>', '<span>']]
        result = _extract_first_element_from_logprobs(tokens)
        self.assertEqual(result, '<div>')


class TestCreateVllmGenerateFn(unittest.TestCase):
    """Test vLLM generate_fn with mocked OpenAI client."""

    def _mock_choice(self, content, finish_reason="stop", logprobs_tokens=None):
        choice = MagicMock()
        choice.message.content = content
        choice.finish_reason = finish_reason
        if logprobs_tokens is not None:
            choice.logprobs.content = [_make_token(t) for t in logprobs_tokens]
        else:
            choice.logprobs = None
        return choice

    def _mock_response(self, contents, finish_reasons=None, logprobs_list=None):
        if finish_reasons is None:
            finish_reasons = ["stop"] * len(contents)
        if logprobs_list is None:
            logprobs_list = [None] * len(contents)
        response = MagicMock()
        response.choices = [
            self._mock_choice(c, fr, lp)
            for c, fr, lp in zip(contents, finish_reasons, logprobs_list)
        ]
        return response

    @patch('openai.OpenAI')
    def test_first_step_extracts_first_element(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        # Model generates full HTML, we should extract first element
        mock_client.chat.completions.create.return_value = self._mock_response([
            '<div style="display:flex;"><div style="bg:red;"></div></div>',
            '<div style="display:flex;gap:10px;"><div>child</div></div>',
            '<div style="flex-direction:column;"><div></div></div>',
        ])

        generate_fn = create_vllm_generate_fn(
            base_url="http://localhost:8888",
            model_name="test-model",
            image_b64="AAAA",
            instruction="Generate HTML",
        )

        results = generate_fn("", 3)
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0], '<div style="display:flex;">')
        self.assertEqual(results[1], '<div style="display:flex;gap:10px;">')
        self.assertEqual(results[2], '<div style="flex-direction:column;">')

        # Verify no stop token was used for first call
        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertNotIn('stop', call_kwargs.kwargs)

    @patch('openai.OpenAI')
    def test_subsequent_step_uses_logprobs(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        # Model continues from prefix, logprobs give us token-level data
        mock_client.chat.completions.create.return_value = self._mock_response(
            contents=["ignored"],  # content ignored when logprobs available
            logprobs_list=[
                ['<div', ' style="', 'background:red;', 'width:100px;', '">', '\n', '<div'],
            ],
        )

        generate_fn = create_vllm_generate_fn(
            base_url="http://localhost:8888",
            model_name="test-model",
            image_b64="AAAA",
            instruction="Generate HTML",
        )

        results = generate_fn('<div style="display:flex;">', 1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], '<div style="background:red;width:100px;">\n')

        # Verify continue_final_message is used, no stop token, logprobs requested
        call_kwargs = mock_client.chat.completions.create.call_args
        self.assertNotIn('stop', call_kwargs.kwargs)
        self.assertTrue(call_kwargs.kwargs.get('logprobs'))
        self.assertTrue(call_kwargs.kwargs['extra_body']['continue_final_message'])

    @patch('openai.OpenAI')
    def test_subsequent_step_no_injection(self, MockOpenAI):
        """Verify prefix is sent as-is, no '<' injected."""
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        prefix = '<div style="display:flex;">'
        mock_client.chat.completions.create.return_value = self._mock_response(
            contents=["ignored"],
            logprobs_list=[['</div', '>\n']],
        )

        generate_fn = create_vllm_generate_fn(
            base_url="http://localhost:8888",
            model_name="test-model",
            image_b64="AAAA",
            instruction="Generate HTML",
        )
        generate_fn(prefix, 1)

        # Check the assistant message content is exactly the prefix (no '<' appended)
        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs.get('messages', call_kwargs.args[0] if call_kwargs.args else None)
        assistant_msg = messages[-1]
        self.assertEqual(assistant_msg["content"], prefix)

    @patch('openai.OpenAI')
    def test_empty_output_is_terminal(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        mock_client.chat.completions.create.return_value = self._mock_response(
            contents=["", "  ", "ignored"],
            logprobs_list=[
                [],             # empty logprobs = terminal
                [_make_token(" ").token for _ in []],  # empty
                ['</div', '>\n'],
            ],
        )
        # Fix: logprobs_list items should be lists of token strings
        mock_client.chat.completions.create.return_value = self._mock_response(
            contents=["", "  ", "ignored"],
            logprobs_list=[
                [],          # empty = terminal
                [' ', ' '],  # whitespace only = terminal
                ['</div', '>\n'],  # valid closing tag
            ],
        )

        generate_fn = create_vllm_generate_fn(
            base_url="http://localhost:8888",
            model_name="test-model",
            image_b64="AAAA",
            instruction="Generate HTML",
        )

        results = generate_fn('<div style="display:flex;">', 3)
        self.assertEqual(results[0], "")   # terminal
        self.assertEqual(results[1], "")   # terminal
        self.assertEqual(results[2], '</div>\n')  # valid

    @patch('openai.OpenAI')
    def test_closing_tag_generation(self, MockOpenAI):
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        mock_client.chat.completions.create.return_value = self._mock_response(
            contents=["ignored", "ignored"],
            logprobs_list=[
                ['</div', '>\n', '<div'],   # closing tag, then next element
                ['</div', '>\n'],            # just a closing tag
            ],
        )

        generate_fn = create_vllm_generate_fn(
            base_url="http://localhost:8888",
            model_name="test-model",
            image_b64="AAAA",
            instruction="Generate HTML",
        )

        results = generate_fn('<div><div style="bg:red;"></div>', 2)
        self.assertEqual(results[0], '</div>\n')
        self.assertEqual(results[1], '</div>\n')

    @patch('openai.OpenAI')
    def test_fallback_without_logprobs(self, MockOpenAI):
        """When logprobs are unavailable, falls back to text-level extraction."""
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        prefix = '<div style="display:flex;">'
        mock_client.chat.completions.create.return_value = self._mock_response(
            contents=[prefix + '<div style="bg:red;"><span>text</span>'],
            logprobs_list=[None],  # no logprobs
        )

        generate_fn = create_vllm_generate_fn(
            base_url="http://localhost:8888",
            model_name="test-model",
            image_b64="AAAA",
            instruction="Generate HTML",
        )

        results = generate_fn(prefix, 1)
        self.assertEqual(results[0], '<div style="bg:red;">')


if __name__ == '__main__':
    unittest.main()
