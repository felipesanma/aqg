from lingua import Language, LanguageDetectorBuilder


LANGUAGES = [Language.ENGLISH, Language.SPANISH]
LANGUAGES_MAPPING = {
    "SPANISH": {"prompt": "Spanish", "language_code": "es"},
    "ENGLISH": {"prompt": "English", "language_code": "en"},
}


class LangDetect:
    def __init__(self) -> None:
        self.languages = LANGUAGES
        self.mapping = LANGUAGES_MAPPING

    def detect_language_from_text(self, text: str):
        detector = LanguageDetectorBuilder.from_languages(*self.languages).build()
        language = detector.detect_language_of(text)
        if not language:
            error_msg = f"language not detected. language: {language}"
            print(error_msg)
            return {"error": error_msg}
        return self.mapping[language.name]
