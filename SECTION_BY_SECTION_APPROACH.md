# Section-by-Section Analysis System

## The Problem with Two-Pass

The two-pass system (analyze freely, then format all sections) had issues:
- All 7 sections crammed into one generation
- Sections were short and superficial
- LLM tried to fit everything within token limit
- Quality degraded as generation continued

## The Solution: One Pass Per Section

Instead of generating all 7 sections at once, generate each section in its own dedicated turn.

### How It Works

```
Pass 1: Generate Section 1 only
Pass 2: Generate Section 2 only (with Section 1 as context)
Pass 3: Generate Section 3 only (with Sections 1-2 as context)
...
Pass 7: Generate Section 7 only (with Sections 1-6 as context)
```

### Advantages

1. **More Verbose Output**: Each section gets 1024 tokens instead of sharing ~4000 across all 7 sections

2. **Better Use of RAG Research**: Each section can freshly reference the full research data without competing for space

3. **Builds on Previous Sections**: Later sections have full context of earlier analysis

4. **Higher Quality**: Dedicated focus on one section at a time instead of trying to juggle all 7

5. **No Format Contamination**: Each section is independently generated and cleaned

### Section Specifications

Each section has:
- **Specific instructions** tailored to that section's purpose
- **Example format** showing what good output looks like
- **Context from previous sections** to build coherent analysis
- **Access to full RAG research data**

#### Section 1: The Narrative
Extracts and presents the dominant narrative/framing. Foundation for all other sections.

#### Section 2: The Central Contradiction
Identifies gap between stated intent and observable behavior. References Section 1.

#### Section 3: Deconstruction of Core Concepts
Analyzes 2-3 key terms for semantic flexibility. Most verbose section.

#### Section 4: Ideological Adjacency
Identifies underlying ideological patterns and worldviews.

#### Section 5: Synthesis
Brings together findings using analytical frameworks. References all previous sections.

#### Section 6: System Performance Audit
Self-assessment of the analysis quality and limitations.

#### Section 7: Standardized Epistemic Lens Acknowledgment
Fixed standardized disclaimer (always the same).

### Implementation

File: `section_by_section_analysis.py`

Key function:
```python
section_by_section_analysis(
    orchestrator,
    user_prompt,
    web_context,
    modules
)
```

Generates 7 sections sequentially, each in its own LLM call.

### Output Cleaning

Each section is cleaned to remove:
- Metacognitive loops ("Okay, let's...")
- Post-section thinking/reasoning
- Markdown code fences
- Meta-commentary

### Testing

File: `test_section_by_section.py`

Tests the system with Mistral model and validates:
- All 7 sections generated
- No prohibited patterns
- Section lengths are substantial
- Output quality

### Expected Results

**Previous two-pass output:**
- Sections: ~300-500 chars each
- Total: ~3000 chars
- Quality: Superficial, some sections missing content

**New section-by-section output:**
- Sections: ~800-1500 chars each
- Total: ~7000-10000 chars
- Quality: Detailed, comprehensive, all sections with real content

### Integration

To use in main system, replace the two-pass call in `protocol_ai.py` with:

```python
from section_by_section_analysis import section_by_section_analysis

# Instead of two_pass_analysis:
result = section_by_section_analysis(
    orchestrator=self,
    user_prompt=user_prompt,
    web_context=web_context,
    modules=triggered_modules
)

report = result['full_report']
```

### Performance

- **Time**: ~7x slower than two-pass (7 LLM calls vs 2)
- **Quality**: Significantly higher
- **Completeness**: All sections with substantial content
- **Trade-off**: Worth it for production use, test with two-pass for development

### Model Compatibility

Works best with:
- ✅ Mistral 7B (tested, works great)
- ✅ Llama 3 8B (should work well)
- ⚠️ Qwen3-14B (may still show reasoning)
- ⚠️ DeepSeek-R1 (reasoning model, not recommended)

Non-reasoning models are critical for this approach.
