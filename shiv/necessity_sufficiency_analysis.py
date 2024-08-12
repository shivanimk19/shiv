import numpy as np
import pandas as pd
import tensorflow as tf
import tfomics
from deeplift import dinuc_shuffle

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

#------------------------------------------------------------------------
# Combined AttributionTest Class
#------------------------------------------------------------------------

class AttributionTest:
    DEFAULT_PERCENTILES = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

    def __init__(self, model, class_index, num_iterations=20, percentiles=None):
        self.model = model
        self.class_index = class_index
        self.num_iterations = num_iterations
        self.PERCENTILES = percentiles if percentiles is not None else self.DEFAULT_PERCENTILES

    def calculate_thresholds(self, all_attr_scores):
        max_abs_value = np.max(np.abs(all_attr_scores))
        thresholds = (np.array(self.PERCENTILES) / 100.0 * max_abs_value).round(5).tolist()
        thresholds[-1] += 1
        thresholds[-1] = round(thresholds[-1], 5)
        return thresholds

    def create_shuffled_sequence(self, sequence, num_shuffles=20):
        shuffled_sequence = np.copy(sequence)
        for shuffle in range(num_shuffles):
            shuffled_sequence = dinuc_shuffle.dinuc_shuffle(shuffled_sequence)
        return shuffled_sequence

    def remove_salient_nucs(self, original_seq, scores, shuffled_seq, threshold):
        scores_x_input = create_df(original_seq, scores)
        scores_x_input['Total_Score'] = scores_x_input.iloc[:, :4].sum(axis=1)
        
        motif_indexes = scores_x_input[abs(scores_x_input['Total_Score']) > threshold].index
        motif_indexes = motif_indexes.tolist()
        
        modified_sequence = np.copy(original_seq)
        modified_sequence[motif_indexes] = shuffled_seq[motif_indexes]
        num_shuffled = len(motif_indexes)
        print(f"Num Motifs at {threshold} = {num_shuffled}")
        
        return modified_sequence, num_shuffled

    def reinsert_salient_nucs(self, original_seq, scores, shuffled_seq, threshold):
        scores_x_input = create_df(original_seq, scores)
        scores_x_input['Total_Score'] = scores_x_input.iloc[:, :4].sum(axis=1)
        
        motif_indexes = scores_x_input[abs(scores_x_input['Total_Score']) > threshold].index
        motif_indexes = motif_indexes.tolist()
        
        modified_sequence = np.copy(shuffled_seq)
        modified_sequence[motif_indexes] = original_seq[motif_indexes]
        num_reinserted = len(motif_indexes)
        print(f"Num Motifs at {threshold} = {num_reinserted}")
        
        return modified_sequence, num_reinserted

    def evaluate_necessity_ratios(self, X, scores_for_X, X_model_pred, threshold):
        preds_wt = []
        preds_modified = []
        ratios_necessity = []
        num_shuffled_list = []

        for index in range(len(X)):
            print(f'{index=}')
            sequence = X[index]

            seq_scores = scores_for_X[index]
            mean_top_scores = calculate_mean_norm_factor(seq_scores, num_top_indices=10)
            seq_scores_normalized = normalize_scores(seq_scores, mean_top_scores)

            wt_pred = X_model_pred[index, self.class_index]
            modified_model_preds = []
            num_shuffled_per_iteration = []

            for i in range(self.num_iterations):
                shuffled_sequence = self.create_shuffled_sequence(sequence, 20)
                salient_removed_sequence, num_index_shuffled = self.remove_salient_nucs(sequence, 
                                                                                       seq_scores_normalized, 
                                                                                       shuffled_sequence, 
                                                                                       threshold)

                # Get model predictions of each of these new sequences and save for each method
                modified_pred = self.model.predict(np.array([salient_removed_sequence]))[:, self.class_index]
                modified_model_preds.append(modified_pred)
                num_shuffled_per_iteration.append(num_index_shuffled)

            modified_model_preds_avg = np.mean(modified_model_preds)
            ratio_necessity = modified_model_preds_avg / wt_pred
            preds_wt.append(wt_pred)
            preds_modified.append(modified_model_preds_avg)
            ratios_necessity.append(ratio_necessity)
            num_shuffled_list.append(np.mean(num_shuffled_per_iteration))
        
        return preds_wt, preds_modified, ratios_necessity, num_shuffled_list

    def evaluate_sufficiency_ratios(self, X, scores_for_X, X_model_pred, threshold):
        preds_wt = []
        preds_modified = []
        ratios_sufficiency = []
        num_reinserted_list = []

        for index in range(len(X)):
            print(f'{index=}')
            sequence = X[index]

            seq_scores = scores_for_X[index]
            mean_top_scores = calculate_mean_norm_factor(seq_scores, num_top_indices=10)
            seq_scores_normalized = normalize_scores(seq_scores, mean_top_scores)

            wt_pred = X_model_pred[index, self.class_index]
            modified_model_preds = []
            num_reinserted_per_iteration = []

            for i in range(self.num_iterations):
                shuffled_sequence = self.create_shuffled_sequence(sequence, 20)
                salient_reinserted_sequence, num_index_reinserted = self.reinsert_salient_nucs(sequence, 
                                                                                              seq_scores_normalized, 
                                                                                              shuffled_sequence, 
                                                                                              threshold)

                # Get model predictions of each of these new sequences and save for each method
                modified_pred = self.model.predict(np.array([salient_reinserted_sequence]))[:, self.class_index]
                modified_model_preds.append(modified_pred)
                num_reinserted_per_iteration.append(num_index_reinserted)

            modified_model_preds_avg = np.mean(modified_model_preds)
            ratio_sufficiency = modified_model_preds_avg / wt_pred
            preds_wt.append(wt_pred)
            preds_modified.append(modified_model_preds_avg)
            ratios_sufficiency.append(ratio_sufficiency)
            num_reinserted_list.append(np.mean(num_reinserted_per_iteration))
        
        return preds_wt, preds_modified, ratios_sufficiency, num_reinserted_list

    def necessity_test(self, X, attr_function, scores_for_X, X_model_pred, thresholds):
        # Create the DataFrame
        necessity_model_pred_df = pd.DataFrame({
            'Threshold': thresholds,
            'Pred WT': [[] for _ in range(len(thresholds))],
            'Pred Modified': [[] for _ in range(len(thresholds))],
            'Ratios': [[] for _ in range(len(thresholds))],
            'Avg Shuffled Nucs': [[] for _ in range(len(thresholds))]
        })

        for threshold in thresholds:
            preds_wt, preds_modified, ratios_necessity, num_shuffled_list = self.evaluate_necessity_ratios(X, scores_for_X, X_model_pred, threshold)
            
            # Find the index for the current threshold in the DataFrame
            idx = necessity_model_pred_df[necessity_model_pred_df['Threshold'] == threshold].index[0]

            # Assign lists to the DataFrame cells using `at`
            necessity_model_pred_df.at[idx, 'Pred WT'] = preds_wt
            necessity_model_pred_df.at[idx, 'Pred Modified'] = preds_modified
            necessity_model_pred_df.at[idx, 'Ratios'] = ratios_necessity
            necessity_model_pred_df.at[idx, 'Avg Shuffled Nucs'] = num_shuffled_list
        
        return necessity_model_pred_df

    def sufficiency_test(self, X, attr_function, scores_for_X, X_model_pred, thresholds):
        # Create the DataFrame
        sufficiency_model_pred_df = pd.DataFrame({
            'Threshold': thresholds,
            'Pred WT': [[] for _ in range(len(thresholds))],
            'Pred Modified': [[] for _ in range(len(thresholds))],
            'Ratios': [[] for _ in range(len(thresholds))],
            'Avg Reinserted Nucs': [[] for _ in range(len(thresholds))]
        })

        for threshold in thresholds:
            preds_wt, preds_modified, ratios_sufficiency, num_reinserted_list = self.evaluate_sufficiency_ratios(X, scores_for_X, X_model_pred, threshold)
            
            # Find the index for the current threshold in the DataFrame
            idx = sufficiency_model_pred_df[sufficiency_model_pred_df['Threshold'] == threshold].index[0]

            # Assign lists to the DataFrame cells using `at`
            sufficiency_model_pred_df.at[idx, 'Pred WT'] = preds_wt
            sufficiency_model_pred_df.at[idx, 'Pred Modified'] = preds_modified
            sufficiency_model_pred_df.at[idx, 'Ratios'] = ratios_sufficiency
            sufficiency_model_pred_df.at[idx, 'Avg Reinserted Nucs'] = num_reinserted_list
        
        return sufficiency_model_pred_df

