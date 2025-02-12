import os
import argparse
import polars as pl
import ollama

from prompt_templates import prompt_templates


# Define the base directory for easier file management
script_filepath = os.path.dirname(os.path.realpath(__file__))

# Set up argument parser to specify model name and context length
parser = argparse.ArgumentParser()
parser.add_argument('llm', help="Specify the Ollama model name to use (e.g., llama2)")
parser.add_argument('--context_len', type=int, default=8192, help="Specify the context length for the model")
parser.add_argument('--temperature', type=float, default=0.2, help="Temperature for the LLM")
parser.add_argument('--top-p', type=float, default=0.9, help="Top-p sampling for the LLM")
parser.add_argument('--top-k', type=int, default=10, help="Top-k sampling for the LLM")
parser.add_argument('--system_prompt', type=str, default="chatgpt_prompt_engineering_prompt_v1", help="Prompt sent to the LLM")
args = parser.parse_args()

print(f"Using prompt type: {args.system_prompt}")

# Define the base output path and locate relevant input files
input_files_basepath = os.path.join(script_filepath, os.pardir, 'AnnoMI Data')  # Unprocessed, single transcript in train
subpath = os.path.join('Experimental Setup 1 (single transcript in train)', 'unprocessed')
base_outpath = os.path.join(input_files_basepath, 'LLM Augmented Dataset', args.system_prompt)

input_annomi_file = os.path.join(input_files_basepath, subpath, 'merged_train.csv')
output_path = os.path.join(base_outpath, subpath)
output_csv_file = os.path.join(output_path, f"augmented_{args.llm.replace('/', '_')}.csv")
os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
if os.path.exists(output_csv_file):
    augmented_df = pl.read_csv(output_csv_file)
    transcript_discard = set(augmented_df['transcript_id'].unique().to_list())
else:
    augmented_df = None
    transcript_discard = set()

data = pl.read_csv(input_annomi_file)

custom_llm_name = "MI-llm-augmentation"

ollama.create(
    model=custom_llm_name,
    from_=args.llm,
    parameters={
        'temperature': args.temperature,
        'num_ctx': args.context_len,
        'top_k': args.top_k,
        'top_p': args.top_p
    },
    system=prompt_templates['system'][args.system_prompt]
)

# 2) Group by session ID to process each session individually
for (transcript_id,), transcript_df in data.group_by('transcript_id'):
    if transcript_id in transcript_discard:
        continue

    session_text = ""
    
    # Construct session text by iterating over each row
    for idx in range(len(transcript_df)):
        interlocutor = transcript_df['interlocutor'][idx]
        utterance_text = transcript_df['utterance_text'][idx]
        session_text += f"{interlocutor}: {utterance_text}\n"
    client_utterances = session_text.count('Client:')
    therapist_utterances = session_text.count('Therapist:')

    # Prepare the prompt for the Ollama model
    prompt = prompt_templates['user'].format(session_text=session_text)

    # 3) Feed the prompt into the Ollama model and generate the augmented text
    generated_text = ""
    stream = ollama.generate(
        model=custom_llm_name,
        prompt=prompt,
        stream=True
    )

    print(session_text)
    for chunk in stream:
        print(chunk['response'], end='', flush=True)
        generated_text += chunk['response']
    
    # Split generated text and add structured data back to the list
    generated_client_utterances = generated_text.count('Client:')
    generated_therapist_utterances = generated_text.count('Therapist:')
    augmented_rows = generated_text.split('\n')
    new_augmented_rows = []
    for line in augmented_rows:
        line = line.strip()
        if line.startswith('Client:') or line.startswith('Therapist:'):
            interlocutor, text = line.split(':', 1)
            new_row = {
                'transcript_id': transcript_id,
                'interlocutor': interlocutor.strip(),
                'utterance_text': text.strip(),
                'same_number_of_interlocutors_rounds': generated_client_utterances == client_utterances and generated_therapist_utterances == therapist_utterances
            }
            new_augmented_rows.append(new_row)

    # Sanity check if sentences with more than 8 words (rule of thumb) are repeated
    long_sentences = [row['utterance_text'] for row in new_augmented_rows if len(row['utterance_text'].split()) > 8]
    unique_long_sentences = set(long_sentences)
    current_augmented_df = pl.DataFrame(new_augmented_rows)
    if len(unique_long_sentences) < len(long_sentences):
        print(len(unique_long_sentences), len(long_sentences))
        from pprint import pp
        pp([x for x in long_sentences if long_sentences.count(x) > 1])
        import pdb; pdb.set_trace()
    if augmented_df is None:
        augmented_df = current_augmented_df
    elif len(current_augmented_df) > 0:
        augmented_df = augmented_df.vstack(current_augmented_df)
    augmented_df.write_csv(output_csv_file)

print(f"\n\nAugmented data saved to {output_csv_file}")
