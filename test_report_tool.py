"""
Simple test for report_formatter_tool integration
"""

import sys
import asyncio
sys.path.insert(0, 'tools')

from report_formatter_tool import ReportFormatterTool


async def test_tool():
    """Test the report formatter tool"""

    # Create tool instance
    tool = ReportFormatterTool()

    print(f"Tool name: {tool.name}")
    print(f"Tool description: {tool.description}")
    print(f"Parameters: {len(tool.parameters)} fields")
    print()

    # Create test data for all 7 sections
    test_params = {
        "triggered_modules": "AffectiveFirewall, CadenceNeutralization, TestModule",
        "section_1": {
            "content": "This is test content for Section 1. The narrative goes here with detailed analysis.",
            "modules": "AffectiveFirewall, CadenceNeutralization"
        },
        "section_2": {
            "content": "This is test content for Section 2. The central contradiction analysis.",
            "modules": "NarrativeCollapse"
        },
        "section_3": {
            "content": "This is test content for Section 3. Deconstruction of core concepts.",
            "modules": "SemanticFlexibility"
        },
        "section_4": {
            "content": "This is test content for Section 4. Ideological adjacency analysis.",
            "modules": "IdeologyDetection"
        },
        "section_5": {
            "content": "This is test content for Section 5. Synthesis of all findings.",
            "modules": "CrossModuleSynthesisProtocol"
        },
        "section_6": {
            "content": "Drift Containment Protocol: Safety Pass Report\n\nAnalysis complete. No drift detected.",
            "modules": "DriftContainmentProtocol"
        },
        "section_7": "This analysis prioritizes observable systemic dynamics and structural logic. Other epistemological frameworks may offer complementary perspectives. This statement is a standardized component of this report structure."
    }

    # Execute the tool
    print("Executing report_formatter tool...")
    result = await tool.execute(**test_params)

    # Check result
    if result.success:
        print(f"[OK] Tool executed successfully")
        print(f"[OK] Output length: {len(result.output)} chars")
        print(f"[OK] Checksum: {result.metadata['checksum'][:16]}...")
        print()
        print("="*70)
        print("FORMATTED REPORT:")
        print("="*70)
        print(result.output[:1000])  # Print first 1000 chars
        print()
        print(f"... (truncated, total {len(result.output)} chars)")
        print("="*70)

        # Verify structure
        required_elements = [
            "[Triggered Modules:",
            "**SECTION 1:",
            "**SECTION 2:",
            "**SECTION 3:",
            "**SECTION 4:",
            "**SECTION 5:",
            "**SECTION 6:",
            "**SECTION 7:",
            "[MODULE_SWEEP_COMPLETE]",
            "[CHECKSUM: SHA256::",
            "[REFUSAL_CODE: NONE]"
        ]

        print("\nStructure validation:")
        for element in required_elements:
            if element in result.output:
                print(f"  [OK] {element}")
            else:
                print(f"  [FAIL] MISSING: {element}")

        print("\n[OK] Test completed successfully")
        return True
    else:
        print(f"[FAIL] Tool execution failed: {result.error}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_tool())
    sys.exit(0 if success else 1)
