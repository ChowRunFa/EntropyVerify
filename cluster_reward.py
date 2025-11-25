import json
import pandas as pd
import numpy as np
from typing import List, Dict
from tqdm import tqdm

# ==============================================================================
# New, extensible scoring function
# ==============================================================================
def _calculate_cluster_score(group_df: pd.DataFrame) -> pd.Series:
    """
    Calculates cluster_score for a group of samples under the same prompt_idx.

    Args:
        group_df (pd.DataFrame): DataFrame containing all samples for a single prompt_idx.

    Returns:
        pd.Series: A Series containing the cluster_score for each sample, with an index matching the input.
        
    Scoring Logic:
    - Ranks samples by reward_score in descending order.
    - The top-ranked sample gets a score of 1.0.
    - The second-ranked gets 0.9, and so on, decreasing by 0.1 each time.
    - If there are more than 10 samples, the minimum score is 0.0.
    
    Extensibility:
    You can copy this function and modify the internal 'scores' generation logic
    (e.g., to use exponential decay, normalization, etc.), then call your new function 
    in the main process.
    """
    # 1. Sort by reward_score in descending order to determine rank
    sorted_group = group_df.sort_values(by="reward_score", ascending=False)
    
    # 2. Generate the list of scores [1.0, 0.9, 0.8, ...]
    num_samples = len(sorted_group)
    scores = [max(0.0, 1.0 - i * 0.1) for i in range(num_samples)]
    
    # 3. Create a Series of scores with an index corresponding to the sorted group
    score_series = pd.Series(scores, index=sorted_group.index)
    
    # 4. Sort the score Series by the original index to ensure correct alignment with the original DataFrame
    return score_series.sort_index()


# ==============================================================================
# Other helper functions (unchanged)
# ==============================================================================
def _calculate_cluster_pos_distribution(df: pd.DataFrame, num_bins: int = 10) -> List[float]:
    """Calculates the distribution ratio of high-entropy tokens in different position bins within the text sequence."""
    high_df = df[df['high_entropy'] == 1].copy()
    total_high_entropy_tokens = len(high_df)

    if total_high_entropy_tokens == 0:
        return [0.0] * num_bins

    total_len = len(df)
    high_df['pos_norm_internal'] = (high_df.index + 1) / total_len
    
    bins = np.linspace(0.0, 1.0, num_bins + 1)
    bin_assignments = pd.cut(high_df['pos_norm_internal'], bins=bins, labels=False, include_lowest=True, right=True)
    
    bin_counts = bin_assignments.value_counts(sort=False)
    proportions = bin_counts.reindex(range(num_bins), fill_value=0) / total_high_entropy_tokens
    
    return [round(p, 4) for p in proportions.tolist()]

# ==============================================================================
# Main processing function (refactored to support grouped scoring)
# ==============================================================================
def process_jsonl(input_files: List[str], output_file: str, top_percent: float = None, top_k: int = None):
    """
    Processes JSONL files, adding grouped ranking and scoring functionality.

    Workflow:
    1. Read all data into an in-memory DataFrame.
    2. Group by 'prompt_idx' and call _calculate_cluster_score to compute 'cluster_score'.
    3. Iterate through each record to calculate 'high_entropy_pos_norm' and 'cluster_entropy_pos'.
    4. Write the complete records, including all new fields, to the output file.
    """
    if top_k is None and top_percent is None:
        raise ValueError("Error: You must specify either 'top_k' or 'top_percent'.")

    print(f"Starting to process files...")

    # --- Step 1: Read all data into memory ---
    all_data = []
    for input_file in input_files:
        with open(input_file, "r", encoding="utf-8") as fin:
            for line in fin:
                all_data.append(json.loads(line.strip()))
    
    if not all_data:
        print("Input file(s) are empty. Processing finished.")
        return
        
    main_df = pd.DataFrame(all_data)

    # --- Step 2: Group and calculate cluster_score ---
    print("Grouping by prompt_idx and calculating cluster_score...")
    # Using groupby().apply() is a clear way to handle this custom scoring logic per group.
    scores = main_df.groupby('prompt_idx').apply(_calculate_cluster_score).reset_index(level=0, drop=True)
    main_df['cluster_score'] = scores
    main_df['cluster_score'] = main_df['cluster_score'].round(4) # Format the score

    # --- Step 3 & 4: Iterate through records, calculate other features, and write to file ---
    print("Calculating other features and writing to file...")
    with open(output_file, "w", encoding="utf-8") as fout:
        # Convert DataFrame to a list of dictionaries for easier processing
        for record in tqdm(main_df.to_dict('records')):
            entropy_list = record.get('entropy', [])
            df_row = pd.DataFrame({'entropy': entropy_list})
            total_len = len(df_row)

            # Calculate n (number of high-entropy tokens)
            n = 0
            if total_len > 0:
                if top_k is not None:
                    n = min(top_k, total_len)
                elif top_percent is not None:
                    n = max(1, int(total_len * top_percent))

            # Calculate high_entropy_pos_norm
            high_entropy_pos_norm_list = []
            if n > 0:
                high_entropy_indices = df_row.nlargest(n, 'entropy').index.sort_values()
                raw_list = ((high_entropy_indices + 1) / total_len).tolist()
                high_entropy_pos_norm_list = [round(p, 4) for p in raw_list]
                
                # Prepare for distribution calculation
                df_row['high_entropy'] = 0
                df_row.loc[high_entropy_indices, 'high_entropy'] = 1
            else:
                 df_row['high_entropy'] = 0

            # Calculate cluster_entropy_pos
            cluster_pos_distribution = _calculate_cluster_pos_distribution(df_row, num_bins=10)

            # Build the final output
            final_output = {
                "prompt_idx": record.get("prompt_idx"),
                "sample_idx": record.get("sample_idx"),
                "reward_score": record.get("reward_score"),
                "cluster_score": record.get("cluster_score"),  # Insert the newly calculated score
                "cluster_entropy_pos": cluster_pos_distribution,
                "high_entropy_pos_norm": high_entropy_pos_norm_list,
            }
            
            fout.write(json.dumps(final_output, ensure_ascii=False) + "\n")
             
    print(f"Processing complete. Results written to {output_file}")