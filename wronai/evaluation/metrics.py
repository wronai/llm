"""
Evaluation metrics for Polish language models.
"""

import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ..utils.logging import get_logger
from ..data.polish import load_polish_stopwords, POLISH_DIACRITICS_MAP

logger = get_logger(__name__)

class PolishEvaluationMetrics:
    """
    Comprehensive evaluation metrics for Polish language models.
    """

    def __init__(self):
        self.polish_stopwords = load_polish_stopwords()

        # Polish language specific patterns
        self.polish_patterns = {
            'question_words': ['co', 'jak', 'gdzie', 'kiedy', 'dlaczego', 'czy', 'kto', 'która', 'które', 'ile'],
            'formal_indicators': ['szanowny', 'uprzejmie', 'proszę', 'dziękuję', 'pozdrawiam'],
            'informal_indicators': ['cześć', 'siema', 'hej', 'pa', 'dzięki', 'spoko'],
            'polish_conjunctions': ['że', 'żeby', 'aby', 'ponieważ', 'dlatego', 'jednak', 'więc'],
            'polish_particles': ['się', 'nie', 'już', 'tylko', 'bardzo', 'też', 'również']
        }

    def perplexity(
        self,
        predictions: List[List[float]],
        targets: List[int],
        ignore_index: int = -100
    ) -> float:
        """
        Calculate perplexity score.

        Args:
            predictions: Model predictions (logits)
            targets: Target token IDs
            ignore_index: Index to ignore in calculation

        Returns:
            Perplexity score
        """
        total_loss = 0.0
        total_tokens = 0

        for pred_seq, target_seq in zip(predictions, targets):
            for pred, target in zip(pred_seq, target_seq):
                if target != ignore_index:
                    # Convert logits to probabilities
                    probs = np.exp(pred) / np.sum(np.exp(pred))

                    # Calculate negative log likelihood
                    if target < len(probs) and probs[target] > 0:
                        total_loss += -math.log(probs[target])
                        total_tokens += 1

        if total_tokens == 0:
            return float('inf')

        avg_loss = total_loss / total_tokens
        return math.exp(avg_loss)

    def bleu_score(
        self,
        predictions: List[str],
        references: List[str],
        n_grams: int = 4,
        smooth: bool = True
    ) -> Dict[str, float]:
        """
        Calculate BLEU score with Polish language considerations.

        Args:
            predictions: Generated texts
            references: Reference texts
            n_grams: Maximum n-gram order
            smooth: Whether to apply smoothing

        Returns:
            BLEU scores for different n-gram orders
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have same length")

        bleu_scores = {}

        for n in range(1, n_grams + 1):
            precision_scores = []

            for pred, ref in zip(predictions, references):
                pred_tokens = self._tokenize_polish(pred.lower())
                ref_tokens = self._tokenize_polish(ref.lower())

                pred_ngrams = self._get_ngrams(pred_tokens, n)
                ref_ngrams = self._get_ngrams(ref_tokens, n)

                if len(pred_ngrams) == 0:
                    precision_scores.append(0.0)
                    continue

                # Count matches
                matches = 0
                ref_counts = Counter(ref_ngrams)
                pred_counts = Counter(pred_ngrams)

                for ngram in pred_counts:
                    matches += min(pred_counts[ngram], ref_counts.get(ngram, 0))

                precision = matches / len(pred_ngrams)

                # Apply smoothing for zero scores
                if smooth and precision == 0.0 and n > 1:
                    precision = 1.0 / (2 ** n)

                precision_scores.append(precision)

            bleu_scores[f'bleu_{n}'] = sum(precision_scores) / len(precision_scores)

        # Calculate geometric mean for overall BLEU
        if all(score > 0 for score in bleu_scores.values()):
            geometric_mean = math.exp(sum(math.log(score) for score in bleu_scores.values()) / len(bleu_scores))
            bleu_scores['bleu_overall'] = geometric_mean
        else:
            bleu_scores['bleu_overall'] = 0.0

        return bleu_scores

    def rouge_score(
        self,
        predictions: List[str],
        references: List[str],
        rouge_types: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate ROUGE scores for Polish texts.

        Args:
            predictions: Generated texts
            references: Reference texts
            rouge_types: Types of ROUGE to calculate

        Returns:
            ROUGE scores
        """
        if rouge_types is None:
            rouge_types = ['rouge-1', 'rouge-2', 'rouge-l']

        rouge_scores = {}

        for rouge_type in rouge_types:
            scores = []

            for pred, ref in zip(predictions, references):
                pred_tokens = self._tokenize_polish(pred.lower())
                ref_tokens = self._tokenize_polish(ref.lower())

                if rouge_type == 'rouge-1':
                    score = self._rouge_n(pred_tokens, ref_tokens, 1)
                elif rouge_type == 'rouge-2':
                    score = self._rouge_n(pred_tokens, ref_tokens, 2)
                elif rouge_type == 'rouge-l':
                    score = self._rouge_l(pred_tokens, ref_tokens)
                else:
                    score = 0.0

                scores.append(score)

            rouge_scores[rouge_type] = sum(scores) / len(scores)

        return rouge_scores

    def polish_fluency_score(
        self,
        texts: List[str]
    ) -> Dict[str, float]:
        """
        Calculate fluency score specific to Polish language.

        Args:
            texts: Texts to evaluate

        Returns:
            Polish fluency metrics
        """
        scores = {
            'diacritic_usage': [],
            'morphology_richness': [],
            'syntax_complexity': [],
            'vocabulary_diversity': [],
            'formal_register': []
        }

        for text in texts:
            # Diacritic usage (should be present in Polish)
            polish_chars = sum(1 for char in text if char in 'ąćęłńóśźż')
            total_chars = len(re.sub(r'\s+', '', text))
            diacritic_ratio = polish_chars / max(total_chars, 1)
            scores['diacritic_usage'].append(min(diacritic_ratio * 10, 1.0))  # Normalize

            # Morphological richness (different word forms)
            words = self._tokenize_polish(text.lower())
            unique_words = set(words)
            morphology_score = len(unique_words) / max(len(words), 1)
            scores['morphology_richness'].append(morphology_score)

            # Syntax complexity (sentence length variation)
            sentences = re.split(r'[.!?]+', text)
            sentence_lengths = [len(self._tokenize_polish(s)) for s in sentences if s.strip()]
            if sentence_lengths:
                complexity = np.std(sentence_lengths) / max(np.mean(sentence_lengths), 1)
                scores['syntax_complexity'].append(min(complexity, 1.0))
            else:
                scores['syntax_complexity'].append(0.0)

            # Vocabulary diversity (TTR - Type Token Ratio)
            if words:
                ttr = len(set(words)) / len(words)
                scores['vocabulary_diversity'].append(ttr)
            else:
                scores['vocabulary_diversity'].append(0.0)

            # Formal register detection
            formal_indicators = sum(1 for word in words if word in self.polish_patterns['formal_indicators'])
            informal_indicators = sum(1 for word in words if word in self.polish_patterns['informal_indicators'])

            if formal_indicators + informal_indicators > 0:
                formal_ratio = formal_indicators / (formal_indicators + informal_indicators)
            else:
                formal_ratio = 0.5  # Neutral

            scores['formal_register'].append(formal_ratio)

        # Calculate averages
        return {key: sum(values) / len(values) for key, values in scores.items()}

    def polish_correctness_score(
        self,
        texts: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate Polish language correctness.

        Args:
            texts: Texts to evaluate

        Returns:
            Correctness metrics
        """
        scores = {
            'spelling_accuracy': [],
            'grammar_coherence': [],
            'punctuation_usage': [],
            'case_agreement': []
        }

        for text in texts:
            # Basic spelling check (diacritic consistency)
            spelling_errors = self._count_spelling_errors(text)
            total_words = len(self._tokenize_polish(text))
            spelling_accuracy = 1.0 - (spelling_errors / max(total_words, 1))
            scores['spelling_accuracy'].append(max(0.0, spelling_accuracy))

            # Grammar coherence (basic checks)
            grammar_score = self._evaluate_grammar_coherence(text)
            scores['grammar_coherence'].append(grammar_score)

            # Punctuation usage
            punct_score = self._evaluate_punctuation(text)
            scores['punctuation_usage'].append(punct_score)

            # Case agreement (basic noun-adjective agreement)
            case_score = self._evaluate_case_agreement(text)
            scores['case_agreement'].append(case_score)

        return {key: sum(values) / len(values) for key, values in scores.items()}

    def semantic_similarity(
        self,
        predictions: List[str],
        references: List[str],
        method: str = "jaccard"
    ) -> float:
        """
        Calculate semantic similarity between predictions and references.

        Args:
            predictions: Generated texts
            references: Reference texts
            method: Similarity method (jaccard, cosine)

        Returns:
            Average semantic similarity score
        """
        similarities = []

        for pred, ref in zip(predictions, references):
            pred_tokens = set(self._tokenize_polish(pred.lower()))
            ref_tokens = set(self._tokenize_polish(ref.lower()))

            if method == "jaccard":
                intersection = len(pred_tokens & ref_tokens)
                union = len(pred_tokens | ref_tokens)
                similarity = intersection / max(union, 1)
            elif method == "cosine":
                # Simple word-based cosine similarity
                all_tokens = pred_tokens | ref_tokens
                pred_vector = np.array([1 if token in pred_tokens else 0 for token in all_tokens])
                ref_vector = np.array([1 if token in ref_tokens else 0 for token in all_tokens])

                dot_product = np.dot(pred_vector, ref_vector)
                magnitude = np.linalg.norm(pred_vector) * np.linalg.norm(ref_vector)
                similarity = dot_product / max(magnitude, 1e-8)
            else:
                similarity = 0.0

            similarities.append(similarity)

        return sum(similarities) / len(similarities)

    def response_relevance(
        self,
        questions: List[str],
        answers: List[str]
    ) -> float:
        """
        Evaluate response relevance for Q&A pairs.

        Args:
            questions: Questions
            answers: Generated answers

        Returns:
            Average relevance score
        """
        relevance_scores = []

        for question, answer in zip(questions, answers):
            question_tokens = set(self._tokenize_polish(question.lower()))
            answer_tokens = set(self._tokenize_polish(answer.lower()))

            # Remove stopwords
            question_content = question_tokens - self.polish_stopwords
            answer_content = answer_tokens - self.polish_stopwords

            # Calculate overlap
            overlap = len(question_content & answer_content)
            total_content = len(question_content | answer_content)

            relevance = overlap / max(total_content, 1)

            # Bonus for question words being addressed
            question_words = [word for word in question_tokens if word in self.polish_patterns['question_words']]
            if question_words and any(self._is_answer_pattern(answer, qword) for qword in question_words):
                relevance += 0.2  # Bonus for addressing question words

            relevance_scores.append(min(relevance, 1.0))

        return sum(relevance_scores) / len(relevance_scores)

    def _tokenize_polish(self, text: str) -> List[str]:
        """Tokenize text for Polish language processing."""
        # Simple word tokenization preserving Polish characters
        words = re.findall(r'\b[\w\u0104\u0106\u0118\u0141\u0143\u00d3\u015a\u0179\u017b\u0105\u0107\u0119\u0142\u0144\u00f3\u015b\u017a\u017c]+\b', text)
        return [word.lower() for word in words]

    def _get_ngrams(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from token list."""
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    def _rouge_n(self, pred_tokens: List[str], ref_tokens: List[str], n: int) -> float:
        """Calculate ROUGE-N score."""
        pred_ngrams = Counter(self._get_ngrams(pred_tokens, n))
        ref_ngrams = Counter(self._get_ngrams(ref_tokens, n))

        if not ref_ngrams:
            return 0.0

        matches = sum(min(pred_ngrams[ngram], ref_ngrams[ngram]) for ngram in pred_ngrams)
        total_ref = sum(ref_ngrams.values())

        return matches / max(total_ref, 1)

    def _rouge_l(self, pred_tokens: List[str], ref_tokens: List[str]) -> float:
        """Calculate ROUGE-L score (Longest Common Subsequence)."""
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])

            return dp[m][n]

        lcs_len = lcs_length(pred_tokens, ref_tokens)

        if len(pred_tokens) == 0 and len(ref_tokens) == 0:
            return 1.0
        elif len(ref_tokens) == 0:
            return 0.0

        recall = lcs_len / len(ref_tokens)
        precision = lcs_len / max(len(pred_tokens), 1)

        if recall + precision == 0:
            return 0.0

        f1 = (2 * recall * precision) / (recall + precision)
        return f1

    def _count_spelling_errors(self, text: str) -> int:
        """Count basic spelling errors in Polish text."""
        errors = 0

        # Check for common Polish spelling mistakes
        common_errors = [
            (r'\bw\s+tym\b', r'w tym'),  # Should be separate
            (r'\bktórz\b', r'który'),   # Incomplete word
            (r'\bze\s+\b', r'że '),     # Wrong form
        ]

        for pattern, correct in common_errors:
            if re.search(pattern, text, re.IGNORECASE):
                errors += 1

        # Check for inconsistent diacritic usage
        words = self._tokenize_polish(text)
        for word in words:
            if len(word) > 3:
                # Check if word has mixed diacritic usage patterns
                has_diacritics = any(char in word for char in 'ąćęłńóśźż')
                base_word = ''.join(POLISH_DIACRITICS_MAP.get(char, char) for char in word)

                # Simple heuristic: if word appears to be Polish but has no diacritics
                # and is longer than 4 characters, it might be an error
                if not has_diacritics and len(word) > 4 and any(char in 'aclnoszzz' for char in word):
                    # This is a very basic check - in practice, you'd use a dictionary
                    pass

        return errors

    def _evaluate_grammar_coherence(self, text: str) -> float:
        """Evaluate basic grammar coherence."""
        sentences = re.split(r'[.!?]+', text)
        coherence_scores = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            words = self._tokenize_polish(sentence)

            if len(words) == 0:
                coherence_scores.append(0.0)
                continue

            # Basic coherence checks
            score = 1.0

            # Check for reasonable sentence length
            if len(words) < 2:
                score -= 0.3
            elif len(words) > 50:
                score -= 0.2

            # Check for presence of verbs (very basic)
            verb_endings = ['ć', 'em', 'esz', 'e', 'emy', 'ecie', 'ą']
            has_verb = any(word.endswith(ending) for word in words for ending in verb_endings)
            if not has_verb and len(words) > 3:
                score -= 0.2

            # Check for balanced particle usage
            particles = sum(1 for word in words if word in self.polish_patterns['polish_particles'])
            if particles > len(words) * 0.3:  # Too many particles
                score -= 0.1

            coherence_scores.append(max(0.0, score))

        return sum(coherence_scores) / max(len(coherence_scores), 1)

    def _evaluate_punctuation(self, text: str) -> float:
        """Evaluate punctuation usage."""
        score = 1.0

        # Check for basic punctuation patterns
        sentences = re.split(r'[.!?]+', text)

        # Should have some sentence-ending punctuation
        if len(sentences) > 1:
            # Good - has sentence breaks
            pass
        elif len(text) > 100:
            # Long text without punctuation
            score -= 0.3

        # Check for proper spacing around punctuation
        spacing_errors = len(re.findall(r'\w[.!?][A-ZĄĆĘŁŃÓŚŹŻ]', text))  # Missing space after punctuation
        spacing_errors += len(re.findall(r'\s+[,.!?]', text))  # Space before punctuation

        if spacing_errors > 0:
            score -= min(0.3, spacing_errors * 0.1)

        # Check for Polish quotation marks
        polish_quotes = len(re.findall(r'„[^"]*"', text))
        other_quotes = len(re.findall(r'"[^"]*"', text))

        if other_quotes > polish_quotes and polish_quotes == 0:
            score -= 0.1  # Minor penalty for not using Polish quotes

        return max(0.0, score)

    def _evaluate_case_agreement(self, text: str) -> float:
        """Evaluate basic case agreement (simplified)."""
        # This is a very simplified check
        # Real case agreement would require full morphological analysis

        score = 1.0
        words = self._tokenize_polish(text.lower())

        # Check for some basic patterns
        for i in range(len(words) - 1):
            current = words[i]
            next_word = words[i + 1]

            # Simple pattern: adjective + noun ending agreement
            if current.endswith('y') and next_word.endswith('a'):
                # Possible disagreement (masculine adj + feminine noun)
                # This is very basic and would need proper linguistic analysis
                pass

        return score

    def _is_answer_pattern(self, answer: str, question_word: str) -> bool:
        """Check if answer addresses the question word."""
        answer_lower = answer.lower()

        answer_patterns = {
            'co': ['to', 'jest', 'oznacza', 'polega'],
            'jak': ['sposób', 'metoda', 'można', 'należy'],
            'gdzie': ['w', 'na', 'miejsce', 'lokalizacja'],
            'kiedy': ['gdy', 'podczas', 'w czasie', 'data'],
            'dlaczego': ['ponieważ', 'dlatego', 'ze względu', 'powód'],
            'kto': ['osoba', 'człowiek', 'autor', 'twórca'],
            'ile': ['liczba', 'ilość', 'około', 'wiele']
        }

        patterns = answer_patterns.get(question_word, [])
        return any(pattern in answer_lower for pattern in patterns)

def calculate_comprehensive_score(
    predictions: List[str],
    references: List[str],
    questions: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation score for Polish language model.

    Args:
        predictions: Generated texts
        references: Reference texts
        questions: Optional questions for Q&A evaluation
        weights: Weights for different metrics

    Returns:
        Comprehensive evaluation scores
    """
    if weights is None:
        weights = {
            'bleu': 0.2,
            'rouge': 0.2,
            'fluency': 0.25,
            'correctness': 0.25,
            'relevance': 0.1
        }

    evaluator = PolishEvaluationMetrics()
    scores = {}

    # BLEU scores
    bleu_scores = evaluator.bleu_score(predictions, references)
    scores.update(bleu_scores)

    # ROUGE scores
    rouge_scores = evaluator.rouge_score(predictions, references)
    scores.update(rouge_scores)

    # Polish fluency
    fluency_scores = evaluator.polish_fluency_score(predictions)
    scores.update({f'fluency_{k}': v for k, v in fluency_scores.items()})

    # Polish correctness
    correctness_scores = evaluator.polish_correctness_score(predictions)
    scores.update({f'correctness_{k}': v for k, v in correctness_scores.items()})

    # Semantic similarity
    semantic_sim = evaluator.semantic_similarity(predictions, references)
    scores['semantic_similarity'] = semantic_sim

    # Response relevance (if questions provided)
    if questions:
        relevance = evaluator.response_relevance(questions, predictions)
        scores['response_relevance'] = relevance

    # Calculate weighted overall score
    overall_components = {
        'bleu_component': bleu_scores.get('bleu_overall', 0.0),
        'rouge_component': (rouge_scores.get('rouge-1', 0.0) + rouge_scores.get('rouge-2', 0.0) + rouge_scores.get('rouge-l', 0.0)) / 3,
        'fluency_component': sum(fluency_scores.values()) / len(fluency_scores),
        'correctness_component': sum(correctness_scores.values()) / len(correctness_scores),
        'relevance_component': scores.get('response_relevance', semantic_sim)
    }

    overall_score = (
        overall_components['bleu_component'] * weights['bleu'] +
        overall_components['rouge_component'] * weights['rouge'] +
        overall_components['fluency_component'] * weights['fluency'] +
        overall_components['correctness_component'] * weights['correctness'] +
        overall_components['relevance_component'] * weights['relevance']
    )

    scores['overall_score'] = overall_score
    scores['component_scores'] = overall_components

    return scores

def evaluate_model_on_polish_tasks(
    model_predictions: Dict[str, List[str]],
    task_references: Dict[str, List[str]],
    task_questions: Optional[Dict[str, List[str]]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate model on multiple Polish language tasks.

    Args:
        model_predictions: Dictionary of task_name -> predictions
        task_references: Dictionary of task_name -> references
        task_questions: Optional dictionary of task_name -> questions

    Returns:
        Evaluation results for each task
    """
    results = {}

    for task_name in model_predictions:
        if task_name not in task_references:
            logger.warning(f"No references found for task: {task_name}")
            continue

        predictions = model_predictions[task_name]
        references = task_references[task_name]
        questions = task_questions.get(task_name) if task_questions else None

        logger.info(f"Evaluating task: {task_name}")

        task_scores = calculate_comprehensive_score(
            predictions=predictions,
            references=references,
            questions=questions
        )

        results[task_name] = task_scores

    # Calculate cross-task averages
    if results:
        cross_task_scores = {}

        # Get all metric names
        all_metrics = set()
        for task_scores in results.values():
            all_metrics.update(task_scores.keys())

        # Calculate averages
        for metric in all_metrics:
            scores = [task_scores.get(metric, 0.0) for task_scores in results.values() if metric in task_scores]
            if scores:
                cross_task_scores[f"avg_{metric}"] = sum(scores) / len(scores)

        results['cross_task_averages'] = cross_task_scores

    return results

def compare_models(
    model_results: Dict[str, Dict[str, float]],
    reference_model: str = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models' evaluation results.

    Args:
        model_results: Dictionary of model_name -> evaluation_results
        reference_model: Model to use as reference for comparison

    Returns:
        Model comparison results
    """
    comparison = {}

    if reference_model and reference_model in model_results:
        reference_scores = model_results[reference_model]

        for model_name, model_scores in model_results.items():
            if model_name == reference_model:
                continue

            comparison[f"{model_name}_vs_{reference_model}"] = {}

            for metric, score in model_scores.items():
                ref_score = reference_scores.get(metric, 0.0)

                if ref_score > 0:
                    improvement = (score - ref_score) / ref_score * 100
                    comparison[f"{model_name}_vs_{reference_model}"][f"{metric}_improvement_%"] = improvement

                comparison[f"{model_name}_vs_{reference_model}"][f"{metric}_diff"] = score - ref_score

    # Overall ranking
    if 'overall_score' in next(iter(model_results.values()), {}):
        ranking = sorted(
            model_results.items(),
            key=lambda x: x[1].get('overall_score', 0.0),
            reverse=True
        )

        comparison['ranking'] = {
            'by_overall_score': [model_name for model_name, _ in ranking],
            'scores': {model_name: scores.get('overall_score', 0.0) for model_name, scores in ranking}
        }

    return comparison