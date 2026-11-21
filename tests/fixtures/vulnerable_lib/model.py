# Fixture: a lib that renders the model's own chat_template with a PLAIN Jinja
# Environment. This is the server-side template injection pattern the tool flags.
from jinja2 import BaseLoader, Environment


def apply_chat_template(tokenizer, messages):
    chat_template = tokenizer.chat_template  # comes from the model's tokenizer_config.json (untrusted)
    template = Environment(loader=BaseLoader).from_string(chat_template)
    return template.render(messages=messages)
