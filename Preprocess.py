
import json
from pathlib import Path
import os

def convert_to_plasma_format(input_data):
    formatted_data = []

    for entry in input_data:
        uri = entry.get("uri", "")
        question = entry.get("question", "")
        context = entry.get("context", "")
        answers = entry.get("answers", [])
        raw_text = entry.get("raw_text", "")

        labelled_spans = entry.get("labelled_answer_spans", {})
        summaries = entry.get("labelled_summaries", {})

        # Extract perspective keys present
        perspectives_present = list(labelled_spans.keys())
        summary_sources = [f"A{i}" for i in range(len(answers))]

        # Assemble final dictionary
        formatted_entry = {
            "uri": uri,
            "question": question,
            "context": context,
            "answers": answers,
            "labelled_answer_spans": labelled_spans,
            "labelled_summaries": summaries,
            "raw_text": raw_text,
            "meta": {
                "perspectives_present": perspectives_present,
                "summary_sources": summary_sources
            }
        }

        formatted_data.append(formatted_entry)

    return formatted_data



# Input and Output file paths
input_file = "test.json"
output_file = "plasma_formatted_test_data.json"

# Load raw data
with open(input_file, "r", encoding="utf-8") as f:
    raw_data = json.load(f)
    
# Convert
converted_data = convert_to_plasma_format(raw_data)

# Save output
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(converted_data, f, indent=4)
    
print(f"[✓] Converted {len(converted_data)} entries and saved to {output_file}")


# Perspective conditions as per PLASMA
perspective_conditions = {
    "INFORMATION": {
        "anchor": "For information purposes...",
        "tone": "Informative, Educational",
        "definition": "Defined as knowledge about diseases, disorders, and health-related facts, providing insights into symptoms and diagnosis."
    },
    "CAUSE": {
        "anchor": "Some of the causes...",
        "tone": "Explanatory, Causal",
        "definition": "Defined as reasons responsible for the occurrence of a particular medical condition, symptom, or disease."
    },
    "SUGGESTION": {
        "anchor": "It is suggested...",
        "tone": "Advisory, Recommending",
        "definition": "Defined as advice or recommendations to assist users in making informed medical decisions, solving problems, or improving health issues."
    },
    "EXPERIENCE": {
        "anchor": "In user’s experience...",
        "tone": "Personal, Narrative",
        "definition": "Defined as individual experiences, anecdotes, or firsthand insights related to health, treatments, medication usage, or coping strategies."
    },
    "QUESTION": {
        "anchor": "It is inquired...",
        "tone": "Seeking Understanding",
        "definition": "Defined as inquiry made for deeper understanding."
    }
}

def generate_prompts(data):
    prompt_target_pairs = []

    for entry in data:
        question = entry['question']
        answers = entry['answers']
        labelled_summaries = entry['labelled_summaries']

        for perspective_key, summary in labelled_summaries.items():
            # Strip suffix _SUMMARY to get base perspective
            perspective = perspective_key.replace("_SUMMARY", "")
            condition = perspective_conditions.get(perspective)

            if condition:
                prompt = f"""Task: Summarize the following content according to the perspective: {perspective}.

Perspective Definition: {condition['definition']}

Begin Summary With: {condition['anchor']}

Tone of Summary: {condition['tone']}

Associated Question: {question}
what was the architecture 
Content to Summarize:
""" + "\n".join(answers)

                prompt_target_pairs.append({
                    "input": prompt.strip(),
                    "target": summary.strip(),
                    "perspective": perspective,
                    "uri": entry["uri"]
                })

    return prompt_target_pairs


# Load your PLASMA-format data
with open("plasma_formatted_test_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

prompt_data = generate_prompts(data)

# Save to JSONL or JSON
with open("plasma_prompt_test_dataset.json", "w", encoding="utf-8") as f:
    json.dump(prompt_data, f, indent=2)

print(f"[✓] Generated {len(prompt_data)} prompt → summary pairs.")


# Input: your raw PUMA-style file
with open("plasma_formatted_data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

processed = []

for entry in raw_data:
    question = entry["question"]
    answers = entry["answers"]
    labelled_summaries = entry.get("labelled_summaries", {})
    perspectives = entry.get("meta", {}).get("perspectives_present", [])

    for perspective in perspectives:
        key = f"{perspective}_SUMMARY"
        if key not in labelled_summaries:
            continue

        summary = labelled_summaries[key]

        processed.append({
            "question": question,
            "answers": answers,
            "Perspective": perspective,
            "Summary": summary
        })

# Save processed data
Path("data").mkdir(parents=True, exist_ok=True)
with open("data/preprocessed_all_data.json", "w", encoding="utf-8") as f:
    json.dump(processed, f, indent=2)

print(f"Saved {len(processed)} processed examples.")

# Input: your raw PUMA-style file
with open("plasma_formatted_valid_data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

processed = []

for entry in raw_data:
    question = entry["question"]
    answers = entry["answers"]
    labelled_summaries = entry.get("labelled_summaries", {})
    perspectives = entry.get("meta", {}).get("perspectives_present", [])

    for perspective in perspectives:
        key = f"{perspective}_SUMMARY"
        if key not in labelled_summaries:
            continue

        summary = labelled_summaries[key]

        processed.append({
            "question": question,
            "answers": answers,
            "Perspective": perspective,
            "Summary": summary
        })

# Save processed data
Path("data").mkdir(parents=True, exist_ok=True)
with open("data/preprocessed_valid_data.json", "w", encoding="utf-8") as f:
    json.dump(processed, f, indent=2)

print(f" Saved {len(processed)} processed examples.")



# Input: your raw PUMA-style file
with open("plasma_formatted_test_data.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

processed = []

for entry in raw_data:
    question = entry["question"]
    answers = entry["answers"]
    labelled_summaries = entry.get("labelled_summaries", {})
    perspectives = entry.get("meta", {}).get("perspectives_present", [])

    for perspective in perspectives:
        key = f"{perspective}_SUMMARY"
        if key not in labelled_summaries:
            continue

        summary = labelled_summaries[key]

        processed.append({
            "question": question,
            "answers": answers,
            "Perspective": perspective,
            "Summary": summary
        })

# Save processed data
Path("data").mkdir(parents=True, exist_ok=True)
with open("data/preprocessed_test_data.json", "w", encoding="utf-8") as f:
    json.dump(processed, f, indent=2)

print(f" Saved {len(processed)} processed examples.")
