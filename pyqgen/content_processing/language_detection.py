from lingua import Language, LanguageDetectorBuilder


LANGUAGES = [Language.ENGLISH, Language.SPANISH]
LANGUAGES_MAPPING = {
    "SPANISH": {"prompt": "Spanish", "language_code": "es"},
    "ENGLISH": {"prompt": "English", "language_code": "en"},
}


def detect_language_from_text(text: str, languages=LANGUAGES):
    detector = LanguageDetectorBuilder.from_languages(*languages).build()
    language = detector.detect_language_of(text)
    return LANGUAGES_MAPPING[language.name]


long_text = """ se oye bien como no me permites cambiar el tema de mi tesis renuncio al proyecto de concluirla y defenderla soy consciente de que al renunciar de este modo unilateral perderé la beca que ahora disfruto Pero lo prefiero así con tal de liberarme de la servidumbre intelectual a la que me ha sometido Durante los últimos dos años dos años confirmando artículos que he escrito yo solita dos años impartiendo más docencia de la que me toca sobre asignaturas que ignoro por completo y a alumnos que parecen retrasado
"""
short_text = ""

language = detect_language_from_text(short_text)

print(language)
