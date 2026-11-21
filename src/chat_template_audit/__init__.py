"""chat-template-audit: a static check for template-injection risk in LLM chat-template rendering."""

from chat_template_audit.audit import Finding, scan_file, scan_tree

__all__ = ["Finding", "scan_file", "scan_tree"]
__version__ = "0.1.0"
