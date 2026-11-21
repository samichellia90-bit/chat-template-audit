# Fixture: the same job done safely, with a Jinja sandbox. The tool must NOT
# mark this as REVIEW.
from jinja2.sandbox import ImmutableSandboxedEnvironment


def apply_chat_template(tokenizer, messages):
    chat_template = tokenizer.chat_template  # untrusted, but rendered in a sandbox
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(chat_template)
    return template.render(messages=messages)
