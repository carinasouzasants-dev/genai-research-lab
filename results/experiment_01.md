# Experiment 01: Zero-shot vs Few-shot Prompting

## Objective

Compare the behavior of zero-shot and few-shot prompting when asking a
Generative AI model to analyze an agricultural productivity problem.

## Research Question

Does providing examples in the prompt improve the structure, specificity,
and practical usefulness of the model's response?

## Hypothesis

Few-shot prompting may produce responses that are more structured,
specific and aligned with the desired output format.

---

## Experimental Setup

**Model:** Gemini  
**Language:** Python  
**SDK:** Google GenAI SDK  
**Domain:** Agriculture  
**Task:** Diagnose a significant decrease in agricultural productivity.

### Zero-shot

The model receives the problem without examples.

### Few-shot

The model receives examples showing how similar agricultural problems
should be analyzed before receiving the target problem.

---

## Results

| Metric            | Zero-shot |  Few-shot |
| ----------------- | --------: | --------: |
| Response time     |    86.14s |    15.71s |
| Clarity           |      Good | Very Good |
| Relevance         |      Good | Very Good |
| Structure         |      Good | Very Good |
| Specificity       |      Good | Very Good |
| Practical utility |      Good | Very Good |

### Response Time

Few-shot was approximately 81.8% faster in this execution.

However, this result should not be interpreted as evidence that few-shot
prompting is inherently faster. Response latency can vary depending on
model conditions, network, prompt size and API behavior.

---

## Qualitative Analysis

### Zero-shot

The zero-shot response provided a broad and detailed diagnosis.

It identified four major categories:

- Climate and environmental factors
- Soil and nutrition
- Pests and diseases
- Operational and management factors

It also proposed diagnostic, corrective and preventive actions.

### Few-shot

The few-shot response was more directly structured around the requested
analysis.

It organized the answer into:

1. Possible causes
2. Evidence to investigate
3. Recommended actions

The examples provided in the prompt helped establish the expected style
and level of specificity.

---

## Findings

The few-shot approach produced a more structured response for this task.

The experiment suggests that examples can help guide the model toward a
specific response format and domain-oriented reasoning.

The latency result was also lower in the few-shot execution, but more
experiments are required before drawing conclusions about performance.

---

## Limitations

This experiment has several limitations:

- Only one execution was performed for each prompting strategy.
- Response latency may vary between executions.
- The evaluation was primarily qualitative.
- No automated scoring system was used.
- The experiment used a single agricultural scenario.

---

## Next Experiment

The next experiment will investigate structured outputs.

The objective will be to require the model to return the analysis as
structured JSON instead of free-form text.

This will allow the response to be consumed programmatically by Python
and evaluated automatically.
