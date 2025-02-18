import glob
import os
import polars as pl
import re


def parse_metrics_file(filepath):
    """Parses a classification_metrics.txt file and returns a dict of metric values.
    Expects each line to have the format: key: value.
    """

    with open(filepath, "r") as f:
        metrics_content = f.readlines()
    
    accuracy = re.sub(r'\s+', ' ', metrics_content[6]).strip().split(' ')[1]
    weighted_column = [m for m in re.sub(r'\s+', ' ', metrics_content[8]).strip().split(' ')[2:]]
    balanced_accuracy_val = float(metrics_content[10].strip().split(': ')[1])
    balanced_accuracy = f"{balanced_accuracy_val:.4f}"
    metrics = {
        'Acc': accuracy,
        'Prec': weighted_column[0],
        'Rec': weighted_column[1],
        'F1': weighted_column[2],
        'Supp': weighted_column[3],
        'BalAcc': balanced_accuracy,
    }

    if skip_support:
        del metrics['Supp']

    return metrics

def collect_results(hf_output_dir):
    """Walks through hf_output_dir looking for classification_metrics.txt files.
    Expects the directory structure to be:
        hf_output_dir/model_name/augmented_data_name/classification_metrics.txt
    Returns a list of tuples: (model_name, augmented_data_name, metrics_dict)
    """
    model_name_map = {
        'allenai-longformer-base-4096': 'Longformer',
        'google-bigbird-roberta-base': 'BigBird',
        'google-t5-t5-base': 'T5-Base',
        'answerdotai-ModernBERT-base': 'ModernBERT',
    }

    augmented_data_name_map = {
        'chatgpt4': 'ChatGPT (4.0)',
        'deepseek': 'DeepSeek (v3)',
        'gemini': 'Gemini-Flash',
        'llama3.1:70b': 'Llama3.1 (70B)',
        'llama3.2:3b': 'Llama3.2 (3B)',
        'llama3.3:70b': 'Llama3.3 (70B)',
        'mistral-nemo:12b': 'Mistral-Nemo (12B)',
        'nemotron-mini:4b': 'Nemotron-Mini (4B)',
        'phi4:14b': 'Phi4 (14B)',
        'qwen2.5:32b': 'Qwen2.5 (32B)',
        'qwen2.5:72b': 'Qwen2.5 (72B)',
        'smollm2:1.7b': 'Smollm2 (1.7B)',
    }

    results = []
    pattern = os.path.join(hf_output_dir, "*", "classification_metrics.txt")
    for metrics_file in glob.glob(pattern):
        # Extract model name and augmented data name from the path
        parts = os.path.normpath(metrics_file).split(os.sep)
        # Assuming last part is classification_metrics.txt, next one is augmented_data_name,
        # and next one is model_name.
        if 'unprocessed' in parts[-2]:
            model_name = parts[-2].split('_finetuned')[0]
            augmented_data_name = parts[-2].split(')_', maxsplit=1)[1]
            if '+' in augmented_data_name:
                augmented_data_name = re.sub(r'\(\d\)_', '', augmented_data_name)
                augmentation_type = 'Fused'
            else:
                augmentation_type = 'NLP'
        elif 'augmented' in parts[-2]:
            model_name, augmented_data_name = parts[-2].split('_finetuned_augmented_augmented_')
            if '+' in augmented_data_name:
                augmented_data_name = '+'.join([
                    augmented_data_name_map[augmented_data_name] for augmented_data_name in augmented_data_name.split('+')
                ])
                augmentation_type = 'Fused'
            else:
                augmented_data_name = augmented_data_name_map[augmented_data_name]
                augmentation_type = 'LLM'
        else:
            model_name, augmented_data_name = parts[-2].split('_finetuned_original')
            augmented_data_name = "AnnoMI (Base Data)"
            augmentation_type = 'Base'

        if 't5-t5' in model_name:
            continue

        try:
            metrics = parse_metrics_file(metrics_file)
        except Exception as e:
            print(f"Error parsing {metrics_file}: {e}")
            continue
            
        model_name = model_name_map[model_name]

        results.append((model_name, augmented_data_name, augmentation_type, metrics))
    return results

def generate_latex_table(results):
    """Generates a LaTeX table from the results.
    Rows: each combination of (model_name, augmented_data_name)
    Columns: Metric keys as reported in the classification_metrics.txt
    """
    # Gather all metric keys used across files
    metric_keys = ["Acc", "BalAcc", "F1", "Prec", "Rec"] + ([] if skip_support else ["Supp"])

    # Create header: two initial columns for model & augmented_data, then metrics
    header = ["Model", "Augmented Data", "Augmentation Type"] + metric_keys

    # Sort results by model name and then by augmented data name
    results = sorted(results, key=lambda x: (x[0], x[1]))

    # Build lines for the table content
    rows = []
    for model_name, augmented_data_name, augmentation_type, metrics in results:
        row = [model_name, augmented_data_name, augmentation_type]
        for key in metric_keys:
            row.append(metrics.get(key, "N/A"))
        rows.append(row)

    df = pl.DataFrame(rows, schema=header)
    atype_dfs = df.partition_by('Augmentation Type', as_dict=True)

    llm_dfs = pl.concat([atype_dfs[('LLM',)], atype_dfs[('Base',)]]).sort(['Model', 'Augmented Data'])
    nlp_dfs = pl.concat([atype_dfs[('NLP',)], atype_dfs[('Base',)]]).sort(['Model', 'Augmented Data'])
    fused_dfs = pl.concat([atype_dfs[('Fused',)], atype_dfs[('Base',)]]).sort(['Model', 'Augmented Data'])

    llm_dfs = llm_dfs.select(pl.all().exclude('Augmentation Type'))
    nlp_dfs = nlp_dfs.select(pl.all().exclude('Augmentation Type'))
    fused_dfs = fused_dfs.select(pl.all().exclude('Augmentation Type'))

    llm_dfs = llm_dfs.to_pandas().set_index(['Model', 'Augmented Data'])
    nlp_dfs = nlp_dfs.to_pandas().set_index(['Model', 'Augmented Data'])
    fused_dfs = fused_dfs.to_pandas().set_index(['Model', 'Augmented Data'])

    llm_latex_table = llm_dfs.to_latex(multirow=True, column_format='l|l|ccccc').replace(
        'multirow[t]', 'multirow[c]'
    ).replace(
        '\cline{1-7}', '\midrule'
    ).replace(
        ' &  & Acc & BalAcc & F1 & Prec & Rec \\', 'Model & Augmented Data & Acc & BalAcc & F1 & Prec & Rec \\'
    ).replace(
        'Model & Augmented Data &  &  &  &  &  \\\\\n', ''
    )
    nlp_latex_table = nlp_dfs.to_latex(multirow=True, column_format='l|l|ccccc').replace(
        'multirow[t]', 'multirow[c]'
    ).replace(
        '\cline{1-7}', '\midrule'
    ).replace(
        ' &  & Acc & BalAcc & F1 & Prec & Rec \\', 'Model & Augmented Data & Acc & BalAcc & F1 & Prec & Rec \\'
    ).replace(
        'Model & Augmented Data &  &  &  &  &  \\\\\n', ''
    )
    fused_latex_table = fused_dfs.to_latex(multirow=True, column_format='l|l|ccccc').replace(
        'multirow[t]', 'multirow[c]'
    ).replace(
        '\cline{1-7}', '\midrule'
    ).replace(
        ' &  & Acc & BalAcc & F1 & Prec & Rec \\', 'Model & Augmented Data & Acc & BalAcc & F1 & Prec & Rec \\'
    ).replace(
        'Model & Augmented Data &  &  &  &  &  \\\\\n', ''
    )

    head_table = """\\begin{table}
    """
    tail_table = """\\caption{CAPTION}
    \\label{tab:LABEL}
\\end{table}"""
    
    llm_latex_table = head_table + llm_latex_table.replace('\n', '\n    ') + tail_table.replace(
        "CAPTION", "Classification Metrics for different models and LLM-generated augmented data",
    ).replace(
        "LABEL", "llm_classification_metrics_table"
    )
    nlp_latex_table = head_table + nlp_latex_table.replace('\n', '\n    ') + tail_table.replace(
        "CAPTION", "Classification Metrics for different models and NLP-generated augmented data"
    ).replace(
        "LABEL", "nlp_classification_metrics_table"
    )
    fused_latex_table = head_table + fused_latex_table.replace('\n', '\n    ') + tail_table.replace(
        "CAPTION", "Classification Metrics for different models and fused augmented data"
    ).replace(
        "LABEL", "fused_classification_metrics_table"
    )
    
    return llm_latex_table, nlp_latex_table, fused_latex_table

def main():
    # Directory with HF output
    hf_output_dir = "/home/gmedda/projects/MI-LLMAugmentation/hf_output"
    
    # Collect all results
    results = collect_results(hf_output_dir)
    
    if not results:
        print("No classification_metrics.txt files found in", hf_output_dir)
        return

    # Generate latex table
    llm_latex_table, nlp_latex_table, fused_latex_table = generate_latex_table(results)
    
    # For simplicity, print the LaTeX table to stdout.
    # Redirect or write to file if desired.
    with open("../llm_classification_metrics_table.tex", "w") as f:
        print(llm_latex_table, file=f)

    with open("../nlp_classification_metrics_table.tex", "w") as f:
        print(nlp_latex_table, file=f)
    
    with open("../fused_classification_metrics_table.tex", "w") as f:
        print(fused_latex_table, file=f)

if __name__ == '__main__':
    skip_support = True
    main()