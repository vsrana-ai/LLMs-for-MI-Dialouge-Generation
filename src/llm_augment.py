import os
import argparse
import polars as pl
import ollama


prompt_templates = {
    "base_prompt": (
        "You are an expert psychologist. Please expand on the following therapy session by providing detailed "
        "follow-up responses and additional insights from both the therapist and client perspectives. "
        "Here's the session:\n\n"
        "{session_text}\n\n"
        "Continue the following therapy session. Respond only with dialogue lines, following this exact format: "
        "Therapist: [therapist’s response]"
        "Client: [client’s response]"
        "Do not include any narrative descriptions, explanations, or additional text outside of the direct dialogue."
    ),
    "chatgpt_prompt_engineering_prompt_v1": (
        "You are a highly experienced psychologist specializing in motivational interviewing, equipped with an extensive "
        "understanding of therapeutic techniques. Your task is to thoughtfully expand on the following recorded therapy "
        "session by providing additional detailed responses and nuanced insights that deepen the conversation.\n\n"
        "Task Instructions:\n"
        "1. First, analyze the context of each interaction between the client and therapist.\n"
        "2. Identify any implicit emotional or cognitive themes in the client's responses.\n"
        "3. Develop thoughtful, empathetic follow-up questions and responses that the therapist might use to encourage "
        "self-reflection, goal setting, and personal insight for the client.\n"
        "4. For each client response, consider potential underlying emotions or beliefs, and use these insights to "
        "craft the therapist's next responses.\n\n"
        "Chain of Thought Process:\n"
        "- Step 1: Carefully review each client and therapist utterance.\n"
        "- Step 2: Interpret the client's perspective or underlying emotions from the last response.\n"
        "- Step 3: Formulate a response that builds on the client’s themes, encouraging further exploration.\n"
        "- Step 4: Alternate between client and therapist responses until the session naturally concludes, ensuring "
        "the therapist’s responses align with motivational interviewing principles.\n\n"
        "Here is the therapy session:\n\n"
        "{session_text}\n\n"
        "Please continue this session, adding more insightful dialogue. Be mindful to maintain a professional, empathetic tone. "
        "Respond only with dialogue lines, following this exact format: "
        "Therapist: [therapist’s response]"
        "Client: [client’s response]"
        "Do not include any narrative descriptions, explanations, or additional text outside of the direct dialogue."
    ),
    "chatgpt_prompt_engineering_prompt_v2": (
        "You are a highly experienced psychologist specializing in motivational interviewing, equipped with an extensive "
        "understanding of therapeutic techniques. Your task is to thoughtfully expand on the following recorded therapy "
        "session by providing additional detailed responses and nuanced insights that deepen the conversation.\n\n"
        "Session Focus: Explore the client's values and goals to help clarify their motivation for change.\n\n"
        "Task Instructions:\n"
        "1. First, analyze the context of each interaction between the client and therapist.\n"
        "2. Identify any implicit emotional or cognitive themes in the client's responses.\n"
        "3. Develop thoughtful, empathetic follow-up questions and responses that the therapist might use to encourage "
        "self-reflection, goal setting, and personal insight for the client.\n"
        "4. For each client response, consider potential underlying emotions or beliefs, and use these insights to "
        "craft the therapist's next responses.\n"
        "5. Remember to use motivational interviewing techniques such as open-ended questions, affirmations, reflective "
        "listening, and summarization to support the client’s exploration of thoughts and feelings.\n\n"
        "Example Response Format:\n"
        "Therapist: 'How does that make you feel?'\n"
        "Client: 'It makes me feel a bit uncertain about my next steps.'\n\n"
        "Chain of Thought Process:\n"
        "- Step 1: Carefully review each client and therapist utterance.\n"
        "- Step 2: Interpret the client's perspective or underlying emotions from the last response.\n"
        "- Step 3: Formulate a response that builds on the client’s themes, encouraging further exploration.\n"
        "- Step 4: Alternate between client and therapist responses until the session naturally concludes, ensuring "
        "the therapist’s responses align with motivational interviewing principles.\n\n"
        "When responding, consider multiple paths the conversation could take. If a particular response might not resonate, "
        "provide a follow-up that gently probes in a different direction.\n\n"
        "Continue the session by adding approximately 4-5 additional exchanges between the therapist and client, "
        "preserving a natural and meaningful progression.\n\n"
        "Here is the current session:\n\n"
        "{session_text}\n\n"
        "Please continue this session, adding more insightful dialogue. Be mindful to maintain a professional, empathetic tone."
        "Respond only with dialogue lines, following this exact format: "
        "Therapist: [therapist’s response]"
        "Client: [client’s response]"
        "Do not include any narrative descriptions, explanations, or additional text outside of the direct dialogue."
    )
}


# Define the base directory for easier file management
script_filepath = os.path.dirname(os.path.realpath(__file__))

# Set up argument parser to specify model name and context length
parser = argparse.ArgumentParser()
parser.add_argument('llm', help="Specify the Ollama model name to use (e.g., llama2)")
parser.add_argument('--context_len', type=int, default=1024, help="Specify the context length for the model")
parser.add_argument('--prompt_type', type=str, default="chatgpt_prompt_engineering_prompt_v1", help="Prompt sent to the LLM")
args = parser.parse_args()

print(f"Using prompt type: {args.prompt_type}")

# Define the base output path and locate relevant input files
input_files_basepath = os.path.join(script_filepath, os.pardir, 'AnnoMI Data')  # Unprocessed, single transcript in train
subpath = os.path.join('Experimental Setup 1 (single transcript in train)', 'unprocessed')
base_outpath = os.path.join(input_files_basepath, 'LLM Augmented Dataset', args.prompt_type)

input_annomi_file = os.path.join(input_files_basepath, subpath, 'merged_train.csv')
output_path = os.path.join(base_outpath, subpath)

data = pl.read_csv(input_annomi_file)

augmented_texts = []

# 2) Group by session ID to process each session individually
for (transcript_id,), transcript_df in data.group_by('transcript_id'):
    session_text = ""
    
    # Construct session text by iterating over each row
    for idx in range(len(transcript_df)):
        interlocutor = transcript_df['interlocutor'][idx]
        utterance_text = transcript_df['utterance_text'][idx]
        session_text += f"{interlocutor}: {utterance_text}\n"

    # Prepare the prompt for the Ollama model
    prompt = prompt_templates[args.prompt_type].format(session_text=session_text)

    # 3) Feed the prompt into the Ollama model and generate the augmented text
    generated_text = ""
    stream = ollama.generate(
        model=args.llm,
        prompt=prompt,
        options={
            "num_ctx": args.context_len,  # Set the context length dynamically
            "temperature": 0
        },
        stream=True
    )

    for chunk in stream:
        print(chunk['response'], end='', flush=True)
        generated_text += chunk['response']
    
    # Split generated text and add structured data back to the list
    augmented_rows = generated_text.split('\n')
    for line in augmented_rows:
        if line.strip() and ':' in line:
            interlocutor, text = line.split(':', 1)
            new_row = {
                'transcript_id': transcript_id,
                'interlocutor': interlocutor.strip(),
                'utterance_text': text.strip()
            }
            augmented_texts.append(new_row)
    break

# 4) Save the augmented data as a new CSV file with a descriptive filename
augmented_df = pl.DataFrame(augmented_texts)
output_csv_file = os.path.join(output_path, f"augmented_{args.llm.replace('/', '_')}.csv")
os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
augmented_df.write_csv(output_csv_file)

print(f"Augmented data saved to {output_csv_file}")
