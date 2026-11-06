"""DeepSeek V4.1 DSML detector.

LOCAL OVERLAY (James Meadlock's lab, 2026-11-06): stock file + "structural closers"
for the V4.1 detector only (V3.2/V4 detectors untouched).

DSML has no escaping, so a tool argument containing a DSML closer (a file *about*
DSML) used to be cut at that closer. Idea credited to Hector Isaza's patch 0052 in
MiaAI-Lab/DeepSeek-V4.1-Flash-1017-4x-DGX-Sparks-TensorFold (issue #14, reported by
ecohash-co); independent implementation for SGLang.

Rules:
  * ``</ parameter>`` ends a value only when followed by whitespace and a new
    ``< parameter name=`` head, or by the end of the invoke body.
  * ``</ invoke>`` ends a call only when its body is closed (ends with a closer, holds
    no parameter, or is a complete JSON object). An unclosed body's ``</ invoke>`` is
    structure when ``</ calls>``, a new ``< invoke`` or the end of the reply follows
    (the open parameter is then closed, not dropped) and literal text otherwise.
  * ``</ calls>`` ends the block only outside an invoke.
  * Every closer kept as text is logged.

Kill switch: ``SGLANG_DSV41_STRUCTURAL_DSML=0`` restores stock behaviour.
"""

import json
import logging
import os
import re
from typing import List, Literal, Optional, Union

from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import StructuralTag
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.utils import _find_common_prefix, _partial_json_loads

logger = logging.getLogger(__name__)

OVERLAY_ID = "dsv41-structural-dsml-sglang-v1"
_ENABLED = os.environ.get("SGLANG_DSV41_STRUCTURAL_DSML", "1") != "0"


class DeepSeekV41Detector(DeepSeekV32Detector):
    """DeepSeek V4.1 DSML detector.

    Tag names have a leading space: " calls", " invoke", and " parameter".
    """

    tool_calls_block_name = " calls"
    invoke_tag_name = " invoke"
    parameter_tag_name = " parameter"

    # The encoder joins an assistant turn's content and its calls block with a
    # blank line, and renders it even when there is no content.
    tool_calls_prefix = "\n\n"
    think_end_token = "</think>"

    def __init__(self):
        super().__init__()
        self._sd_final = False
        self._sd_param_open = f"<{self.dsml_token}{self.parameter_tag_name}"
        self._sd_param_close = f"</{self.dsml_token}{self.parameter_tag_name}>"
        po = re.escape(self._sd_param_open)
        self._sd_head_re = re.compile(rf'{po}\s+name="([^"]+)"\s+string="([^"]+)"\s*>')
        self._sd_next_head_re = re.compile(rf"{po}\s+name=")
        self._sd_head_prog_re = re.compile(rf"{po}(?:\s[^>]*)?$")
        self._sd_invoke_head_re = re.compile(
            rf'{re.escape(self.invoke_start_token)}\s+name="(?P<name>[^"]+)"\s*'
            r"(?:(?P<self_close>/>)|>)"
        )

    # ── structural helpers ────────────────────────────────────────────────

    def _sd_closer_structural(self, text: str, after: int, partial: bool) -> bool:
        follower = text[after:].lstrip()
        if not follower:
            return True
        if self._sd_next_head_re.match(follower):
            return True
        if partial and (
            self._sd_param_open.startswith(follower)
            or self._sd_head_prog_re.match(follower)
            # No lexer here: a streaming tick can end inside the next closer.
            or self.invoke_end_token.startswith(follower)
            or self.eot_token.startswith(follower)
        ):
            return True
        return False

    def _sd_value_end(self, content: str, vstart: int, partial: bool):
        pos = vstart
        while True:
            c = content.find(self._sd_param_close, pos)
            if c < 0:
                return None
            after = c + len(self._sd_param_close)
            if self._sd_closer_structural(content, after, partial):
                return c, after
            pos = after

    def _sd_body_closed(self, body: str) -> bool:
        s = body.strip()
        if not s:
            return True
        if s.startswith("{"):
            try:
                json.loads(s)
                return True
            except (json.JSONDecodeError, ValueError):
                return False
        return (self._sd_param_open not in s) or s.endswith(self._sd_param_close)

    def _sd_body_open_param(self, body: str) -> bool:
        i = body.rfind(self._sd_param_open)
        if i < 0:
            return False
        m = self._sd_head_re.match(body, i)
        return bool(m) and self._sd_value_end(body, m.end(), partial=False) is None

    def _sd_log(self, closer: str):
        logger.info("%s: DSML %r inside a parameter value kept as text", OVERLAY_ID, closer)

    def _sd_match_invoke(self, text: str, final: bool):
        """Return (start, name, body, complete, end) for the first invoke, or None.

        ``end`` is where parsing resumes; a block closer that ends an invoke whose
        own closer is missing is not consumed.
        """
        m = self._sd_invoke_head_re.search(text)
        if not m:
            return None
        name = m.group("name").strip()
        if m.group("self_close"):
            return m.start(), name, "", True, m.end()
        bs = m.end()
        pos = bs
        while True:
            ci = text.find(self.invoke_end_token, pos)
            ce = text.find(self.eot_token, pos)
            if ci < 0 and ce < 0:
                return m.start(), name, text[bs:], False, len(text)
            if ci >= 0 and (ce < 0 or ci <= ce):
                c, clen, closer = ci, len(self.invoke_end_token), self.invoke_end_token
            else:
                c, clen, closer = ce, 0, self.eot_token  # missing </invoke>: stop before </calls>
            body = text[bs:c]
            after = c + clen
            if self._sd_body_closed(body):
                return m.start(), name, body, True, after
            follower = text[c + len(closer):].lstrip()
            structural_next = (
                (closer == self.invoke_end_token and follower.startswith(self.eot_token))
                or follower.startswith(self.invoke_start_token)
                or (final and not follower)
            )
            if structural_next:
                if self._sd_body_open_param(body):
                    body = body + self._sd_param_close
                return m.start(), name, body, True, after
            if not final and (
                not follower
                or self.eot_token.startswith(follower)
                or self.invoke_start_token.startswith(follower)
            ):
                return m.start(), name, body, False, len(text)  # undecided, wait
            self._sd_log(closer)
            pos = c + len(closer)

    # ── parameter parsing ─────────────────────────────────────────────────

    def _parse_parameters_from_xml(self, invoke_content: str, allow_partial: bool = False) -> str:
        if not _ENABLED:
            return super()._parse_parameters_from_xml(invoke_content, allow_partial)
        invoke_content_stripped = invoke_content.strip()
        if invoke_content_stripped.startswith("{"):
            if allow_partial:
                for token in reversed(self.prefix_invoke_end_call):
                    invoke_content_stripped = invoke_content_stripped.rstrip(token)
                return invoke_content_stripped
            elif invoke_content_stripped.endswith("}"):
                return invoke_content_stripped

        parameters = {}
        last_match_end = 0
        pos = 0
        while True:
            m = self._sd_head_re.search(invoke_content, pos)
            if not m:
                break
            found = self._sd_value_end(invoke_content, m.end(), allow_partial)
            if found is None:
                break
            vend, pos = found
            last_match_end = pos
            param_name, param_type = m.group(1), m.group(2)
            param_value = invoke_content[m.end():vend]
            if param_type == "true":
                parameters[param_name] = param_value.strip()
            else:
                try:
                    parameters[param_name] = json.loads(param_value.strip())
                except (json.JSONDecodeError, ValueError):
                    parameters[param_name] = param_value.strip()

        if allow_partial:
            remaining_content = invoke_content[last_match_end:]
            for token in reversed(self.prefix_parameter_end_call):
                remaining_content = remaining_content.rstrip(token)
            partial_match = re.search(self.partial_parameter_regex, remaining_content, re.DOTALL)
            if partial_match and (param_value := partial_match.group(3)):
                param_name = partial_match.group(1)
                if partial_match.group(2) == "true":
                    parameters[param_name] = param_value.strip()
                else:
                    try:
                        parameters[param_name] = _partial_json_loads(param_value, Allow.ALL)[0]
                    except (json.JSONDecodeError, MalformedJSON, ValueError):
                        parameters[param_name] = param_value.strip()

        return json.dumps(parameters, ensure_ascii=False)

    # ── one-shot parsing ──────────────────────────────────────────────────

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        if not _ENABLED:
            return super().detect_and_parse(text, tools)
        idx = text.find(self.bot_token)
        normal_text = text[:idx].removesuffix("\n\n") if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        calls = []
        try:
            pos = idx
            while True:
                b = text.find(self.bot_token, pos)
                if b < 0:
                    break
                seg = text[b + len(self.bot_token):]
                cur, section, closed = 0, [], False
                while True:
                    rest = seg[cur:]
                    mi = self._sd_match_invoke(rest, final=True)
                    e = rest.find(self.eot_token)
                    if mi is None or (e >= 0 and e < mi[0]):
                        if e >= 0:
                            closed = True
                            cur += e + len(self.eot_token)
                        break
                    section.append((mi[1], mi[2]))
                    cur += mi[4]
                if not closed:
                    break  # stock: a block that never closes yields no calls
                for func_name, invoke_content in section:
                    func_args = self._parse_parameters_from_xml(invoke_content)
                    calls.extend(
                        self.parse_base_json(
                            {"name": func_name, "parameters": json.loads(func_args)}, tools
                        )
                    )
                pos = b + len(self.bot_token) + cur
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    # ── streaming ─────────────────────────────────────────────────────────

    def parse_streaming_increment(self, new_text: str, tools: list[Tool]) -> StreamingParseResult:
        if not _ENABLED:
            return super().parse_streaming_increment(new_text, tools)
        self._buffer += new_text
        current_text = self._buffer

        dsml_markers = ["｜DSML｜", "<｜", "</｜"]
        potentially_dsml = any(marker in current_text for marker in dsml_markers)
        dsml_prefixes = ["<", "<｜", "</", "</｜"]
        ends_with_prefix = any(current_text.rstrip().endswith(p) for p in dsml_prefixes)

        if not self.has_tool_call(current_text) and not potentially_dsml and not ends_with_prefix:
            self._buffer = ""
            for e_token in [self.eot_token, self.invoke_end_token]:
                if e_token in current_text:
                    current_text = current_text.replace(e_token, "")
            return StreamingParseResult(normal_text=current_text)

        all_calls: list[ToolCallItem] = []
        preamble = ""
        try:
            while True:
                mi = self._sd_match_invoke(current_text, final=self._sd_final)
                if mi is None:
                    break
                call_start, func_name, invoke_content, is_tool_end, call_end = mi

                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]
                    bot_pos = current_text.rfind(self.bot_token, 0, call_start)
                    if bot_pos != -1:
                        call_start = bot_pos
                    preamble = current_text[:call_start].removesuffix("\n\n")

                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                while len(self.streamed_args_for_tool) <= self.current_tool_id:
                    self.streamed_args_for_tool.append("")

                if not self.current_tool_name_sent:
                    all_calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, name=func_name, parameters="")
                    )
                    self.current_tool_name_sent = True

                current_params = self._parse_parameters_from_xml(
                    invoke_content, allow_partial=not is_tool_end
                )
                sent_len = len(self.streamed_args_for_tool[self.current_tool_id])
                prev_params = self.prev_tool_call_arr[self.current_tool_id].get("arguments")
                argument_diff = None
                if is_tool_end:
                    argument_diff = current_params[sent_len:]
                elif prev_params is not None:
                    if current_params != prev_params:
                        prefix = _find_common_prefix(current_params, prev_params)
                        if len(prefix) > sent_len:
                            argument_diff = prefix[sent_len:]
                if argument_diff:
                    all_calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, name=None, parameters=argument_diff)
                    )
                    self.streamed_args_for_tool[self.current_tool_id] += argument_diff
                self.prev_tool_call_arr[self.current_tool_id] = {
                    "name": func_name,
                    "arguments": current_params,
                }
                if is_tool_end:
                    self._buffer = current_text[call_end:]
                    current_text = self._buffer
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False
                    continue
                break
            return StreamingParseResult(normal_text=preamble, calls=all_calls)
        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            self._buffer = ""
            if not current_text.startswith(preamble):
                current_text = preamble + current_text
            return StreamingParseResult(normal_text=current_text)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        # A held </ invoke> at the very end of the reply becomes structure now.
        if _ENABLED and self.current_tool_name_sent and self.invoke_start_token in self._buffer:
            self._sd_final = True
            try:
                return self.parse_streaming_increment("", tools)
            finally:
                self._sd_final = False
        return super().finish(tools)

    def get_structural_tag_name(self) -> Optional[str]:
        # xgrammar's builtin "deepseek_v4" tag hardcodes the unspaced names,
        # so the V4.1 tag is assembled in get_structural_tag instead.
        return None

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
        parallel_tool_calls: bool = True,
    ) -> Optional[StructuralTag]:
        """The builtin "deepseek_v4" shape with the spaced tag names.

        Bodies are JSON: xgrammar's "deepseek_xml" body style also hardcodes
        the unspaced " parameter" name, and the V3.2-lineage parser accepts a
        JSON body inside an invoke.
        """
        try:
            from xgrammar.structural_tag import (
                AnyTextFormat,
                ConstStringFormat,
                JSONSchemaFormat,
                OrFormat,
                SequenceFormat,
                StructuralTag,
                TagFormat,
                TagsWithSeparatorFormat,
                TriggeredTagsFormat,
            )
        except ImportError:
            return None

        tools = list(tools or [])
        if isinstance(tool_choice, ToolChoice):
            tools = [
                tool
                for tool in tools
                if tool.function.name == tool_choice.function.name
            ]
            if len(tools) != 1:
                return None
        if not tools:
            return None

        def invoke_tag(tool: Tool) -> TagFormat:
            function = tool.function
            schema = function.parameters if function.strict else True
            if schema is None:
                schema = True
            return TagFormat(
                begin=f'{self.invoke_start_token} name="{function.name}">',
                content=JSONSchemaFormat(json_schema=schema),
                end=f"{self.invoke_end_token}\n",
            )

        tags = [invoke_tag(tool) for tool in tools]
        if isinstance(tool_choice, ToolChoice):
            calls = tags[0]
        elif parallel_tool_calls:
            calls = TagsWithSeparatorFormat(tags=tags, separator="", at_least_one=True)
        else:
            calls = OrFormat(elements=tags)
        block_begin = f"{self.bot_token}\n"

        if tool_choice == "auto":
            body = TriggeredTagsFormat(
                triggers=[self.bot_token],
                tags=[TagFormat(begin=block_begin, content=calls, end=self.eot_token)],
                excludes=["<think>", self.think_end_token],
            )
        else:
            body = SequenceFormat(
                elements=[
                    ConstStringFormat(value=self.tool_calls_prefix + block_begin),
                    calls,
                    ConstStringFormat(value=self.eot_token),
                ]
            )
        if not thinking_mode:
            return StructuralTag(format=body)
        reasoning = TagFormat(
            begin="", content=AnyTextFormat(), end=self.think_end_token
        )
        return StructuralTag(format=SequenceFormat(elements=[reasoning, body]))


if _ENABLED:
    logger.info("%s: structural DSML closers active", OVERLAY_ID)
