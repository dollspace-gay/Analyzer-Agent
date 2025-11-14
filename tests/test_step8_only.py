"""
Quick test of just Step 8 (final formatting) using existing section content.
"""

from protocol_ai import ModuleLoader, LLMInterface, Module
from section_by_section_analysis import create_final_formatting_prompt

# Mock sections (simplified for testing)
mock_sections = [
    "OpenAI presents itself as building safe and beneficial AGI...",
    "Stated Intent: Benefit all humanity\nBehavior: API pricing creates tiered access\nNarrative Collapse: Mission conflicts with commercial structure",
    "Concept: AI Safety\nThe Narrative: Emphasized as core priority\nStructural Analysis: Operationalized as liability protection",
    "Technological determinism, neoliberal capitalism, market optimism detected",
    "Synthesis reveals pattern of virtue-washed coercion through moral language",
    "Analysis completeness: Strong structural analysis, limited by lack of internal data",
    "This analysis prioritizes observable systemic dynamics and structural logic. Other epistemological frameworks may offer complementary perspectives. This statement is a standardized component of this report structure."
]

# Mock modules
class MockModule:
    def __init__(self, name):
        self.name = name

mock_modules = [
    MockModule("AffectiveFirewall"),
    MockModule("BluntTone"),
    MockModule("CadenceNeutralization"),
]

# Create formatting prompt
prompt = create_final_formatting_prompt(mock_sections, mock_modules)

print("="*70)
print("STEP 8 FORMATTING PROMPT TEST")
print("="*70)
print("\nPrompt length:", len(prompt), "chars")
print("\nPrompt preview (first 1000 chars):")
print(prompt[:1000])
print("\n[...]")
print("\nPrompt end (last 500 chars):")
print(prompt[-500:])
print("\n" + "="*70)
print("This prompt would be sent to the LLM to format the sections.")
print("The LLM should output a properly formatted report with:")
print("  - Section headers")
print("  - Module tags per section")
print("  - Section 7 with just the epistemic statement")
print("  - Checksum/metadata at the end")
print("="*70)
