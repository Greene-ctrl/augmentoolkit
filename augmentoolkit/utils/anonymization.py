import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree

# Try to import Presidio
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

_nltk_data_downloaded = False
_analyzer = None
_anonymizer = None

def _ensure_presidio():
    global _analyzer, _anonymizer
    if PRESIDIO_AVAILABLE and _analyzer is None:
        try:
            _analyzer = AnalyzerEngine()
            _anonymizer = AnonymizerEngine()
        except Exception as e:
            print(f"Warning: Could not initialize Presidio: {e}")

def _ensure_nltk_data():
    global _nltk_data_downloaded
    if not _nltk_data_downloaded:
        try:
            nltk.download('punkt_tab', quiet=True)
            nltk.download('averaged_perceptron_tagger_eng', quiet=True)
            nltk.download('maxent_ne_chunker_tab', quiet=True)
            nltk.download('words', quiet=True)
            _nltk_data_downloaded = True
        except Exception as e:
            print(f"Warning: Could not download NLTK data for anonymization: {e}")

def anonymize_text_nltk(text):
    """
    Fallback NLTK-based anonymization.
    """
    _ensure_nltk_data()
    try:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        chunks = ne_chunk(tagged)

        anonymized_tokens = []

        def traverse(t):
            if isinstance(t, Tree):
                label = t.label()
                if label == 'PERSON':
                    anonymized_tokens.append('[PERSON]')
                elif label in ['GPE', 'LOCATION']:
                    anonymized_tokens.append('[CITY]')
                else:
                    for leaf in t:
                        traverse(leaf)
            else:
                anonymized_tokens.append(t[0])

        for chunk in chunks:
            traverse(chunk)

        result = " ".join(anonymized_tokens)
        for char in [".", ",", "!", "?", ":", ";"]:
            result = result.replace(f" {char}", char)
        result = result.replace(" 's", "'s")
        return result
    except Exception as e:
        print(f"Warning: NLTK Anonymization failed: {e}")
        return text

def anonymize_text(text):
    """
    Anonymizes text by replacing names of people and locations with placeholders.
    Uses Microsoft Presidio if available, otherwise falls back to NLTK.
    """
    if not text:
        return text

    if PRESIDIO_AVAILABLE:
        _ensure_presidio()
        if _analyzer and _anonymizer:
            try:
                # Analyze for PERSON and LOCATION (LOCATION often covers cities in Presidio)
                results = _analyzer.analyze(text=text, entities=["PERSON", "LOCATION"], language='en')

                operators = {
                    "PERSON": OperatorConfig("replace", {"new_value": "[PERSON]"}),
                    "LOCATION": OperatorConfig("replace", {"new_value": "[CITY]"}),
                }

                anonymized_result = _anonymizer.anonymize(
                    text=text,
                    analyzer_results=results,
                    operators=operators
                )
                return anonymized_result.text
            except Exception as e:
                print(f"Warning: Presidio anonymization failed, falling back to NLTK: {e}")

    return anonymize_text_nltk(text)
