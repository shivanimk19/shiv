import numpy as np
import pandas as pd
import tensorflow as tf
import tfomics
from deeplift import dinuc_shuffle

class Necessity:
    def __init__(self, model, class_index, num_iterations=20):
        self.model = model
        self.class_index = class_index
        self.num_iterations = num_iterations

    def necessity_test(self, X, scores_for_X, X_model_pred, prune_vals):
        # Create the DataFrame
        necessity_model_pred_df = pd.DataFrame({
            'Num Pruned-Lowest Score': prune_vals,
            'Pred WT': [[] for _ in range(len(prune_vals))],
            'Pred Modified': [[] for _ in range(len(prune_vals))],
            'Ratios': [[] for _ in range(len(prune_vals))],
        })

        for index in range(len(X)):
            print(f'{index=}')
            sequence = X[index]
            seq_scores = scores_for_X[index]

            mean_top_scores = calculate_mean_norm_factor(seq_scores, num_top_indices=10)
            seq_scores_normalized = normalize_scores(seq_scores, mean_top_scores)
            wt_pred = X_model_pred[index, self.class_index]

            avg_modified_preds = self.process_sequence(sequence, seq_scores_normalized, prune_vals)

            for prune_idx, prune_val in enumerate(prune_vals):
                avg_modified_pred = avg_modified_preds[prune_val]
                pred_ratio = avg_modified_pred / wt_pred

                necessity_model_pred_df.at[prune_idx, 'Pred WT'].append(wt_pred)
                necessity_model_pred_df.at[prune_idx, 'Pred Modified'].append(avg_modified_pred)
                necessity_model_pred_df.at[prune_idx, 'Ratios'].append(pred_ratio)

        return necessity_model_pred_df

    def process_sequence(self, sequence, seq_scores_normalized, prune_vals):
        all_modified_preds = {prune_val: [] for prune_val in prune_vals}

        for i in range(self.num_iterations):
            shuffled_sequence = self.create_shuffled_sequence(sequence, 20)

            for prune_val in prune_vals:
                modified_sequence = self.remove_salient_nucs(sequence, seq_scores_normalized, shuffled_sequence, prune_val)
                modified_pred = self.model.predict(np.array([modified_sequence]))[:, self.class_index]
                all_modified_preds[prune_val].append(modified_pred[0])

        avg_modified_preds = {prune_val: np.mean(all_modified_preds[prune_val]) for prune_val in prune_vals}

        return avg_modified_preds

    def create_shuffled_sequence(self, sequence, num_shuffles=20):
        shuffled_sequence = np.copy(sequence)
        for shuffle in range(num_shuffles):
            shuffled_sequence = dinuc_shuffle.dinuc_shuffle(shuffled_sequence)
        return shuffled_sequence

    def remove_salient_nucs(self, original_seq, scores, shuffled_seq, prune_val):
        scores_x_input = create_df(original_seq, scores)
        scores_x_input['Total_Score'] = scores_x_input.iloc[:, :4].sum(axis=1)

        # Get the indices of the highest-scoring prune_val sequences
        high_indices = scores_x_input['Total_Score'].abs().nlargest(prune_val).index.tolist()
        # Create a copy of the original sequence
        modified_sequence = np.copy(original_seq)
        # Replace the elements at the high-scoring indices with the shuffled sequence
        modified_sequence[high_indices] = shuffled_seq[high_indices]

        return modified_sequence
        
    def generate_prune_vals(self, seq_length, step):
        # Generate prune values from step to length_of_seq, inclusive, increasing by step size
        prune_vals = list(range(0, seq_length + 1, step))
        
        if prune_vals[-1] != seq_length:
            prune_vals.append(seq_length)
        
        return prune_vals

#------------------------------------------------------------------------
# Helper Functions
#------------------------------------------------------------------------

def calculate_mean_norm_factor(scores, num_top_indices=10):
    flattened_scores = scores.flatten()
    sorted_indices = np.argsort(flattened_scores)
    top_indices = sorted_indices[-num_top_indices:]
    mean_top_scores = np.mean(flattened_scores[top_indices])
    return mean_top_scores

def normalize_scores(scores, mean_top):
    return scores / mean_top

def create_df(sequence, scores):
    x_expanded = np.expand_dims(sequence, axis=0)
    scores_expanded = np.expand_dims(scores, axis=0)
    df = tfomics.impress.grad_times_input_to_df(x_expanded, scores_expanded)
    return df
