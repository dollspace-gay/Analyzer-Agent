"""
Checksum Tool - Generate SHA-256 checksums for report integrity verification
"""

import hashlib
from typing import Dict, Any


def generate_report_checksum(report_body: str) -> str:
    """
    Generate SHA-256 checksum for report body.

    Args:
        report_body: Full report text (excluding terminal lines)

    Returns:
        64-character hex digest
    """
    # Ensure UTF-8 encoding with LF line breaks
    normalized_body = report_body.replace('\r\n', '\n').replace('\r', '\n')

    # Generate SHA-256 hash
    hash_obj = hashlib.sha256(normalized_body.encode('utf-8'))
    return hash_obj.hexdigest()


def verify_report_checksum(report_body: str, expected_checksum: str) -> bool:
    """
    Verify report checksum matches expected value.

    Args:
        report_body: Full report text
        expected_checksum: Expected 64-character hex digest

    Returns:
        True if checksum matches, False otherwise
    """
    actual_checksum = generate_report_checksum(report_body)
    return actual_checksum.lower() == expected_checksum.lower()


def generate_system_integrity_checksum(modules: list, principles: list) -> str:
    """
    Generate system integrity checksum based on loaded modules and principles.

    Args:
        modules: List of module objects
        principles: List of principle strings

    Returns:
        Formatted checksum like "CS[CSM-v1.0:a1b2c3d4]"
    """
    canonical_string = "CSM-v1.0"

    # Sort and hash module names
    module_names = sorted([m.name for m in modules])
    module_concat = '|'.join(module_names)
    module_hash = hashlib.sha256(module_concat.encode('utf-8')).hexdigest()
    canonical_string += f"|MODULE_HASH:{module_hash}"

    # Sort and hash principles
    principles_sorted = sorted(principles)
    principle_concat = '|'.join(principles_sorted)
    principle_hash = hashlib.sha256(principle_concat.encode('utf-8')).hexdigest()
    canonical_string += f"|PRINCIPLE_HASH:{principle_hash}"

    # Generate final checksum
    final_hash = hashlib.sha256(canonical_string.encode('utf-8')).hexdigest()

    # Return first 8 characters
    return f"CS[CSM-v1.0:{final_hash[:8]}]"


# Tool registration for protocol_ai
def get_tool_definition() -> Dict[str, Any]:
    """Return tool definition for registration."""
    return {
        "name": "generate_checksum",
        "description": "Generate SHA-256 checksum for report integrity verification",
        "function": generate_report_checksum,
        "parameters": {
            "report_body": {
                "type": "string",
                "description": "Full report text to hash",
                "required": True
            }
        }
    }
