"""
Polish morphological analysis for WronAI.
"""

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ...utils.logging import get_logger

logger = get_logger(__name__)

class PolishMorphologyAnalyzer:
    """
    Analyzer for Polish morphological features.
    """

    def __init__(self):
        # Polish morphological patterns
        self.noun_endings = {
            # Masculine
            'masc': {
                'nom_sg': ['', 'a', 'o'],
                'gen_sg': ['a', 'u', 'y', 'i'],
                'dat_sg': ['owi', 'u', 'y', 'i'],
                'acc_sg': ['a', '', 'o'],
                'ins_sg': ['em', 'ą', 'y', 'i'],
                'loc_sg': ['e', 'u', 'y', 'i'],
                'nom_pl': ['y', 'i', 'e', 'owie'],
                'gen_pl': ['ów', 'y', 'i', ''],
                'dat_pl': ['om', 'ам'],
                'acc_pl': ['ów', 'y', 'i', 'e'],
                'ins_pl': ['ami', 'ами'],
                'loc_pl': ['ach', 'ах']
            },
            # Feminine
            'fem': {
                'nom_sg': ['a', 'i', ''],
                'gen_sg': ['y', 'i', 'y'],
                'dat_sg': ['ie', 'y', 'i'],
                'acc_sg': ['ę', 'y', 'i'],
                'ins_sg': ['ą', 'ą', 'ią'],
                'loc_sg': ['ie', 'y', 'i'],
                'nom_pl': ['y', 'i', 'e'],
                'gen_pl': ['', 'y', 'i'],
                'dat_pl': ['om', 'am'],
                'acc_pl': ['y', 'i', 'e'],
                'ins_pl': ['ami', 'ами'],
                'loc_pl': ['ach', 'ах']
            },
            # Neuter
            'neut': {
                'nom_sg': ['o', 'e', 'ę'],
                'gen_sg': ['a', 'a', 'ęcia'],
                'dat_sg': ['u', 'u', 'ęciu'],
                'acc_sg': ['o', 'e', 'ę'],
                'ins_sg': ['em', 'em', 'ęciem'],
                'loc_sg': ['e', 'e', 'ęciu'],
                'nom_pl': ['a', 'a', 'ęta'],
                'gen_pl': ['', '', 'ąt'],
                'dat_pl': ['om', 'om', 'ętom'],
                'acc_pl': ['a', 'a', 'ęta'],
                'ins_pl': ['ami', 'ami', 'ętami'],
                'loc_pl': ['ach', 'ach', 'ętach']
            }
        }

        self.adjective_endings = {
            'masc': {
                'nom_sg': ['y', 'i'],
                'gen_sg': ['ego', 'iego'],
                'dat_sg': ['emu', 'iemu'],
                'acc_sg': ['ego', 'iego', 'y', 'i'],
                'ins_sg': ['ym', 'im'],
                'loc_sg': ['ym', 'im']
            },
            'fem': {
                'nom_sg': ['a', 'ia'],
                'gen_sg': ['ej', 'iej'],
                'dat_sg': ['ej', 'iej'],
                'acc_sg': ['ą', 'ią'],
                'ins_sg': ['ą', 'ią'],
                'loc_sg': ['ej', 'iej']
            },
            'neut': {
                'nom_sg': ['e', 'ie'],
                'gen_sg': ['ego', 'iego'],
                'dat_sg': ['emu', 'iemu'],
                'acc_sg': ['e', 'ie'],
                'ins_sg': ['ym', 'im'],
                'loc_sg': ['ym', 'im']
            }
        }

        self.verb_endings = {
            'present': {
                '1sg': ['ę', 'em', 'ę', 'am'],
                '2sg': ['esz', 'isz', 'asz'],
                '3sg': ['e', 'i', 'a'],
                '1pl': ['emy', 'imy', 'amy'],
                '2pl': ['ecie', 'icie', 'acie'],
                '3pl': ['ą', 'ą', 'ają']
            },
            'past': {
                'masc_sg': ['łem', 'ł'],
                'fem_sg': ['łam', 'ła'],
                'neut_sg': ['łem', 'ło'],
                'masc_pl': ['liśmy', 'li'],
                'fem_pl': ['łyśmy', 'ły'],
                'mixed_pl': ['liśmy', 'li']
            },
            'infinitive': ['ć', 'c'],
            'imperative': {
                '2sg': ['', 'ij', 'aj'],
                '2pl': ['cie', 'ijcie', 'ajcie']
            }
        }

        # Common prefixes
        self.prefixes = [
            'przy', 'prze', 'po', 'pod', 'nad', 'za', 'na', 'w', 'wy',
            'do', 'od', 'u', 'roz', 'roz', 'bez', 'nie', 'współ', 'przeciw'
        ]

        # Common suffixes for word formation
        self.derivational_suffixes = {
            'noun': ['anie', 'enie', 'ość', 'ość', 'nik', 'arka', 'stwo', 'izm'],
            'adjective': ['ny', 'owy', 'ski', 'icki', 'arski', 'owy', 'alny'],
            'adverb': ['nie', 'sko', 'ąco', 'owo', 'ie']
        }

    def analyze_word(self, word: str) -> Dict[str, any]:
        """
        Analyze morphological features of a Polish word.

        Args:
            word: Polish word to analyze

        Returns:
            Dictionary with morphological analysis
        """
        word = word.lower().strip()

        analysis = {
            'word': word,
            'length': len(word),
            'pos_candidates': [],
            'gender_candidates': [],
            'case_candidates': [],
            'number_candidates': [],
            'person_candidates': [],
            'tense_candidates': [],
            'morphological_complexity': 0,
            'prefix': None,
            'suffix': None,
            'stem': word
        }

        # Detect prefix
        for prefix in sorted(self.prefixes, key=len, reverse=True):
            if word.startswith(prefix) and len(word) > len(prefix) + 2:
                analysis['prefix'] = prefix
                analysis['stem'] = word[len(prefix):]
                analysis['morphological_complexity'] += 1
                break

        stem = analysis['stem']

        # Analyze as noun
        noun_analysis = self._analyze_as_noun(stem)
        if noun_analysis:
            analysis['pos_candidates'].append('noun')
            analysis['gender_candidates'].extend(noun_analysis.get('genders', []))
            analysis['case_candidates'].extend(noun_analysis.get('cases', []))
            analysis['number_candidates'].extend(noun_analysis.get('numbers', []))

        # Analyze as adjective
        adj_analysis = self._analyze_as_adjective(stem)
        if adj_analysis:
            analysis['pos_candidates'].append('adjective')
            analysis['gender_candidates'].extend(adj_analysis.get('genders', []))
            analysis['case_candidates'].extend(adj_analysis.get('cases', []))
            analysis['number_candidates'].extend(adj_analysis.get('numbers', []))

        # Analyze as verb
        verb_analysis = self._analyze_as_verb(stem)
        if verb_analysis:
            analysis['pos_candidates'].append('verb')
            analysis['person_candidates'].extend(verb_analysis.get('persons', []))
            analysis['tense_candidates'].extend(verb_analysis.get('tenses', []))
            analysis['number_candidates'].extend(verb_analysis.get('numbers', []))

        # Detect derivational morphology
        for pos, suffixes in self.derivational_suffixes.items():
            for suffix in suffixes:
                if stem.endswith(suffix):
                    analysis['suffix'] = suffix
                    analysis['morphological_complexity'] += 1
                    if pos not in analysis['pos_candidates']:
                        analysis['pos_candidates'].append(pos)
                    break

        # Calculate morphological complexity
        complexity_factors = [
            len(analysis['pos_candidates']),
            len(analysis['gender_candidates']),
            len(analysis['case_candidates']),
            int(analysis['prefix'] is not None),
            int(analysis['suffix'] is not None)
        ]
        analysis['morphological_complexity'] = sum(complexity_factors)

        # Remove duplicates
        for key in ['pos_candidates', 'gender_candidates', 'case_candidates',
                   'number_candidates', 'person_candidates', 'tense_candidates']:
            analysis[key] = list(set(analysis[key]))

        return analysis

    def _analyze_as_noun(self, word: str) -> Optional[Dict[str, List[str]]]:
        """Analyze word as a potential noun."""
        candidates = {'genders': [], 'cases': [], 'numbers': []}

        for gender, cases in self.noun_endings.items():
            for case_number, endings in cases.items():
                for ending in endings:
                    if word.endswith(ending):
                        candidates['genders'].append(gender)
                        case, number = case_number.split('_')
                        candidates['cases'].append(case)
                        candidates['numbers'].append(number)

        return candidates if any(candidates.values()) else None

    def _analyze_as_adjective(self, word: str) -> Optional[Dict[str, List[str]]]:
        """Analyze word as a potential adjective."""
        candidates = {'genders': [], 'cases': [], 'numbers': ['sg']}  # Adjectives typically singular

        for gender, cases in self.adjective_endings.items():
            for case, endings in cases.items():
                for ending in endings:
                    if word.endswith(ending):
                        candidates['genders'].append(gender)
                        candidates['cases'].append(case)

        return candidates if any(candidates.values()) else None

    def _analyze_as_verb(self, word: str) -> Optional[Dict[str, List[str]]]:
        """Analyze word as a potential verb."""
        candidates = {'persons': [], 'tenses': [], 'numbers': []}

        # Check infinitive
        for ending in self.verb_endings['infinitive']:
            if word.endswith(ending):
                candidates['tenses'].append('infinitive')
                return candidates

        # Check present tense
        for person, endings in self.verb_endings['present'].items():
            for ending in endings:
                if word.endswith(ending):
                    candidates['persons'].append(person)
                    candidates['tenses'].append('present')
                    if person.endswith('sg'):
                        candidates['numbers'].append('sg')
                    elif person.endswith('pl'):
                        candidates['numbers'].append('pl')

        # Check past tense
        for person_gender, endings in self.verb_endings['past'].items():
            for ending in endings:
                if word.endswith(ending):
                    candidates['tenses'].append('past')
                    if 'sg' in person_gender:
                        candidates['numbers'].append('sg')
                    elif 'pl' in person_gender:
                        candidates['numbers'].append('pl')

        # Check imperative
        for person, endings in self.verb_endings['imperative'].items():
            for ending in endings:
                if word.endswith(ending):
                    candidates['persons'].append(person)
                    candidates['tenses'].append('imperative')
                    if person.endswith('sg'):
                        candidates['numbers'].append('sg')
                    elif person.endswith('pl'):
                        candidates['numbers'].append('pl')

        return candidates if any(candidates.values()) else None

    def get_word_forms(self, base_word: str, target_pos: str = 'noun') -> List[str]:
        """
        Generate possible word forms for a given base word.

        Args:
            base_word: Base form of the word
            target_pos: Target part of speech

        Returns:
            List of possible word forms
        """
        base_word = base_word.lower().strip()
        forms = set()

        # Remove common endings to get stem
        stem = self._get_stem(base_word, target_pos)

        if target_pos == 'noun':
            for gender, cases in self.noun_endings.items():
                for case_number, endings in cases.items():
                    for ending in endings:
                        form = stem + ending
                        if form != stem:  # Avoid empty endings creating same word
                            forms.add(form)

        elif target_pos == 'adjective':
            for gender, cases in self.adjective_endings.items():
                for case, endings in cases.items():
                    for ending in endings:
                        form = stem + ending
                        forms.add(form)

        elif target_pos == 'verb':
            # Add infinitive forms
            for ending in self.verb_endings['infinitive']:
                forms.add(stem + ending)

            # Add present tense forms
            for person, endings in self.verb_endings['present'].items():
                for ending in endings:
                    form = stem + ending
                    forms.add(form)

            # Add past tense forms
            for person_gender, endings in self.verb_endings['past'].items():
                for ending in endings:
                    form = stem + ending
                    forms.add(form)

        return sorted(list(forms))

    def _get_stem(self, word: str, pos: str) -> str:
        """Extract stem from word based on POS."""
        # Simple stemming - remove common endings
        if pos == 'noun':
            # Remove common noun endings
            common_endings = ['a', 'o', 'e', 'y', 'i', 'ą', 'ę', 'ów', 'em', 'ie']
            for ending in sorted(common_endings, key=len, reverse=True):
                if word.endswith(ending) and len(word) > len(ending) + 2:
                    return word[:-len(ending)]

        elif pos == 'verb':
            # Remove infinitive endings
            for ending in ['ć', 'c']:
                if word.endswith(ending):
                    return word[:-len(ending)]

        elif pos == 'adjective':
            # Remove common adjective endings
            for ending in ['y', 'i', 'a', 'e', 'ą', 'ę']:
                if word.endswith(ending) and len(word) > len(ending) + 2:
                    return word[:-len(ending)]

        return word

    def analyze_text_morphology(self, text: str) -> Dict[str, any]:
        """
        Analyze morphological features of entire text.

        Args:
            text: Text to analyze

        Returns:
            Morphological analysis of the text
        """
        words = re.findall(r'\b[a-ząćęłńóśźż]+\b', text.lower())

        analysis = {
            'total_words': len(words),
            'unique_words': len(set(words)),
            'morphological_diversity': 0,
            'pos_distribution': defaultdict(int),
            'complexity_distribution': defaultdict(int),
            'average_complexity': 0,
            'most_complex_words': [],
            'prefix_usage': defaultdict(int),
            'suffix_usage': defaultdict(int)
        }

        word_analyses = []
        total_complexity = 0

        for word in words:
            word_analysis = self.analyze_word(word)
            word_analyses.append(word_analysis)

            # Count POS distribution
            for pos in word_analysis['pos_candidates']:
                analysis['pos_distribution'][pos] += 1

            # Count complexity
            complexity = word_analysis['morphological_complexity']
            analysis['complexity_distribution'][complexity] += 1
            total_complexity += complexity

            # Track prefixes and suffixes
            if word_analysis['prefix']:
                analysis['prefix_usage'][word_analysis['prefix']] += 1
            if word_analysis['suffix']:
                analysis['suffix_usage'][word_analysis['suffix']] += 1

        # Calculate averages
        if words:
            analysis['average_complexity'] = total_complexity / len(words)
            analysis['morphological_diversity'] = len(set(words)) / len(words)

        # Find most complex words
        complex_words = sorted(word_analyses, key=lambda x: x['morphological_complexity'], reverse=True)
        analysis['most_complex_words'] = [
            (w['word'], w['morphological_complexity'])
            for w in complex_words[:10]
        ]

        return analysis

    def check_agreement(self, phrase: str) -> Dict[str, any]:
        """
        Check morphological agreement in a phrase.

        Args:
            phrase: Polish phrase to check

        Returns:
            Agreement analysis
        """
        words = re.findall(r'\b[a-ząćęłńóśźż]+\b', phrase.lower())

        if len(words) < 2:
            return {'agreement_score': 1.0, 'issues': []}

        word_analyses = [self.analyze_word(word) for word in words]
        agreement_issues = []
        agreement_score = 1.0

        # Check noun-adjective agreement
        for i in range(len(words) - 1):
            current = word_analyses[i]
            next_word = word_analyses[i + 1]

            # Check if we have potential adjective-noun or noun-adjective pair
            if ('adjective' in current['pos_candidates'] and 'noun' in next_word['pos_candidates']) or \
               ('noun' in current['pos_candidates'] and 'adjective' in next_word['pos_candidates']):

                # Check gender agreement
                current_genders = set(current['gender_candidates'])
                next_genders = set(next_word['gender_candidates'])

                if current_genders and next_genders and not (current_genders & next_genders):
                    agreement_issues.append({
                        'type': 'gender_disagreement',
                        'words': [words[i], words[i + 1]],
                        'expected_genders': list(current_genders),
                        'actual_genders': list(next_genders)
                    })
                    agreement_score -= 0.2

                # Check case agreement
                current_cases = set(current['case_candidates'])
                next_cases = set(next_word['case_candidates'])

                if current_cases and next_cases and not (current_cases & next_cases):
                    agreement_issues.append({
                        'type': 'case_disagreement',
                        'words': [words[i], words[i + 1]],
                        'expected_cases': list(current_cases),
                        'actual_cases': list(next_cases)
                    })
                    agreement_score -= 0.2

        return {
            'agreement_score': max(0.0, agreement_score),
            'issues': agreement_issues,
            'total_pairs_checked': len(words) - 1
        }

    def suggest_corrections(self, word: str, target_features: Dict[str, str]) -> List[str]:
        """
        Suggest morphological corrections for a word.

        Args:
            word: Word to correct
            target_features: Target morphological features

        Returns:
            List of suggested corrections
        """
        analysis = self.analyze_word(word)
        suggestions = []

        target_pos = target_features.get('pos', 'noun')
        target_gender = target_features.get('gender', 'masc')
        target_case = target_features.get('case', 'nom')
        target_number = target_features.get('number', 'sg')

        stem = self._get_stem(word, target_pos)

        if target_pos == 'noun' and target_gender in self.noun_endings:
            case_number = f"{target_case}_{target_number}"
            if case_number in self.noun_endings[target_gender]:
                for ending in self.noun_endings[target_gender][case_number]:
                    suggestion = stem + ending
                    if suggestion != word:
                        suggestions.append(suggestion)

        elif target_pos == 'adjective' and target_gender in self.adjective_endings:
            if target_case in self.adjective_endings[target_gender]:
                for ending in self.adjective_endings[target_gender][target_case]:
                    suggestion = stem + ending
                    if suggestion != word:
                        suggestions.append(suggestion)

        return suggestions[:5]  # Return top 5 suggestions

def analyze_polish_sentence(sentence: str) -> Dict[str, any]:
    """
    Analyze morphological features of a Polish sentence.

    Args:
        sentence: Polish sentence to analyze

    Returns:
        Comprehensive morphological analysis
    """
    analyzer = PolishMorphologyAnalyzer()

    # Basic text analysis
    text_analysis = analyzer.analyze_text_morphology(sentence)

    # Agreement analysis
    agreement_analysis = analyzer.check_agreement(sentence)

    # Word-by-word analysis
    words = re.findall(r'\b[a-ząćęłńóśźż]+\b', sentence.lower())
    word_analyses = [analyzer.analyze_word(word) for word in words]

    return {
        'sentence': sentence,
        'text_analysis': text_analysis,
        'agreement_analysis': agreement_analysis,
        'word_analyses': word_analyses,
        'morphological_summary': {
            'total_words': len(words),
            'complex_words': len([w for w in word_analyses if w['morphological_complexity'] > 2]),
            'average_complexity': text_analysis['average_complexity'],
            'agreement_score': agreement_analysis['agreement_score'],
            'has_agreement_issues': len(agreement_analysis['issues']) > 0
        }
    }

def get_word_forms(word: str, pos: str = 'noun') -> List[str]:
    """
    Get all possible morphological forms of a Polish word.

    Args:
        word: Base word
        pos: Part of speech

    Returns:
        List of word forms
    """
    analyzer = PolishMorphologyAnalyzer()
    return analyzer.get_word_forms(word, pos)

def check_polish_morphology(text: str) -> Dict[str, any]:
    """
    Comprehensive morphological check for Polish text.

    Args:
        text: Polish text to check

    Returns:
        Morphological analysis and quality metrics
    """
    analyzer = PolishMorphologyAnalyzer()

    sentences = re.split(r'[.!?]+', text)
    sentence_analyses = []

    for sentence in sentences:
        if sentence.strip():
            analysis = analyze_polish_sentence(sentence.strip())
            sentence_analyses.append(analysis)

    # Aggregate results
    total_words = sum(s['morphological_summary']['total_words'] for s in sentence_analyses)
    total_complex = sum(s['morphological_summary']['complex_words'] for s in sentence_analyses)

    agreement_scores = [s['morphological_summary']['agreement_score'] for s in sentence_analyses]
    avg_agreement = sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0

    all_issues = []
    for s in sentence_analyses:
        all_issues.extend(s['agreement_analysis']['issues'])

    return {
        'text': text,
        'sentence_count': len(sentence_analyses),
        'total_words': total_words,
        'complex_words': total_complex,
        'complexity_ratio': total_complex / max(total_words, 1),
        'average_agreement_score': avg_agreement,
        'total_agreement_issues': len(all_issues),
        'morphological_quality_score': (avg_agreement + (1 - min(len(all_issues) / max(total_words, 1), 1))) / 2,
        'sentence_analyses': sentence_analyses,
        'issues_by_type': {
            'gender_disagreement': len([i for i in all_issues if i['type'] == 'gender_disagreement']),
            'case_disagreement': len([i for i in all_issues if i['type'] == 'case_disagreement'])
        }
    }

# Additional utility functions for morphological processing

def normalize_polish_morphology(word: str) -> str:
    """
    Normalize Polish word to its base form (lemmatization).

    Args:
        word: Polish word to normalize

    Returns:
        Normalized form
    """
    analyzer = PolishMorphologyAnalyzer()
    analysis = analyzer.analyze_word(word)

    # Simple heuristic lemmatization
    if 'noun' in analysis['pos_candidates']:
        # Try to get nominative singular
        stem = analyzer._get_stem(word, 'noun')
        # Add most common masculine nominative endings
        for ending in ['', 'a', 'o']:
            candidate = stem + ending
            # This is a simplified approach - real lemmatization would need a dictionary
            return candidate

    elif 'verb' in analysis['pos_candidates']:
        # Try to get infinitive
        stem = analyzer._get_stem(word, 'verb')
        return stem + 'ć'

    elif 'adjective' in analysis['pos_candidates']:
        # Try to get masculine nominative singular
        stem = analyzer._get_stem(word, 'adjective')
        return stem + 'y'

    return word

def extract_polish_morphemes(word: str) -> Dict[str, Optional[str]]:
    """
    Extract morphemes from a Polish word.

    Args:
        word: Polish word

    Returns:
        Dictionary with prefix, stem, suffix
    """
    analyzer = PolishMorphologyAnalyzer()
    analysis = analyzer.analyze_word(word)

    return {
        'prefix': analysis.get('prefix'),
        'stem': analysis.get('stem'),
        'suffix': analysis.get('suffix'),
        'full_word': word
    }

def compare_morphological_complexity(text1: str, text2: str) -> Dict[str, any]:
    """
    Compare morphological complexity between two Polish texts.

    Args:
        text1: First text
        text2: Second text

    Returns:
        Comparison results
    """
    analysis1 = check_polish_morphology(text1)
    analysis2 = check_polish_morphology(text2)

    return {
        'text1_complexity': analysis1['complexity_ratio'],
        'text2_complexity': analysis2['complexity_ratio'],
        'complexity_difference': analysis1['complexity_ratio'] - analysis2['complexity_ratio'],
        'text1_agreement': analysis1['average_agreement_score'],
        'text2_agreement': analysis2['average_agreement_score'],
        'agreement_difference': analysis1['average_agreement_score'] - analysis2['average_agreement_score'],
        'more_complex': 'text1' if analysis1['complexity_ratio'] > analysis2['complexity_ratio'] else 'text2',
        'better_agreement': 'text1' if analysis1['average_agreement_score'] > analysis2['average_agreement_score'] else 'text2'
    }