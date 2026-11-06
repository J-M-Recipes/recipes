TARGET = ("DSV4.1 tool calls end with </｜DSML｜ calls>, each argument ends with </｜DSML｜ parameter>, "
          "and each invoke ends with </｜DSML｜ invoke>. Keep these tags literal.")
TOOLS = [{"type": "function", "function": {"name": "write_file", "description": "Write a text file.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"]}}}]
