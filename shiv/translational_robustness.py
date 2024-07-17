import numpy as np
from tqdm import tqdm


#------------------------------------------------------------------------
# Main class
#------------------------------------------------------------------------

class TranslationalRobustness:
    def __init__(self, max_translation, num_translations):
        self.max_translation = max_translation
        self.num_translations = num_translations

    def random_translate(self, sequence):
        translation = int(np.random.uniform(-self.max_translation, self.max_translation))
        new_sequence = np.roll(sequence, translation, axis=0)
        return new_sequence, translation

    def calculate_variation_score(self, scores_realigned, normalized_original_scores):
        all_scores_realigned = scores_realigned.copy()
        all_scores_realigned.insert(0, np.copy(normalized_original_scores))

        for i in range(len(all_scores_realigned)):
            all_scores_realigned[i] = np.squeeze(all_scores_realigned[i])

        all_scores_realigned = np.array(all_scores_realigned)
        variation_score = np.sqrt(np.mean(np.var(all_scores_realigned, axis=0)))

        return variation_score

    def translate(self, sequence, attr_fun, mean_top_scores, scores_normalized):
        scores_realigned_list = []
        for _ in range(self.num_translations):
            translated_sequence = np.copy(sequence)
            translated_sequence, translation_num = self.random_translate(translated_sequence)
            translated_sequence = np.expand_dims(translated_sequence, axis=0)

            scores_translated = attr_fun(translated_sequence)
            scores_translated -= np.mean(scores_translated, axis=2, keepdims=True)
            scores_translated = normalize_scores(scores_translated, mean_top_scores)

            translated_sequence = np.squeeze(translated_sequence, axis=0)
            scores_translated = np.squeeze(scores_translated, axis=0)

            sequence_realigned = np.roll(translated_sequence, -translation_num, axis=0)
            scores_realigned = np.roll(scores_translated, -translation_num, axis=0)
            scores_realigned_list.append(scores_realigned)

        return self.calculate_variation_score(scores_realigned_list, scores_normalized)

    
    def test(self, X, scores_for_X, attr_function):
        variation_scores = []
    
        # Initialize tqdm with total length
        with tqdm(total=len(X), desc='Processing') as pbar:
            for index in range(len(X)):
                sequence = X[index]
                scores_sequence = scores_for_X[index]
    
                mean_top_scores = calculate_mean_norm_factor(scores_sequence, num_top_indices=10)
                scores_normalized = normalize_scores(scores_sequence, mean_top_scores)
    
                variation_score = self.translate(sequence, attr_function, mean_top_scores, scores_normalized)
    
                variation_scores.append(variation_score)
                pbar.update(1)  # Update progress bar
    
        return variation_scores



#------------------------------------------------------------------------
# Helper functions
#------------------------------------------------------------------------

def calculate_mean_norm_factor(scores, num_top_indices=10):
    flattened_scores = scores.flatten()
    top_indices = np.argsort(flattened_scores)[-num_top_indices:]
    mean_top_scores = np.mean(flattened_scores[top_indices])
    return mean_top_scores

def normalize_scores(scores, mean):
    return scores / mean

