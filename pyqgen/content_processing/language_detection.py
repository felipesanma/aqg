from lingua import Language, LanguageDetectorBuilder

languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH]

detector = LanguageDetectorBuilder.from_languages(*languages).build()

long_text = """ se oye bien como no me permites cambiar el tema de mi tesis renuncio al proyecto de concluirla y defenderla soy consciente de que al renunciar de este modo unilateral perderé la beca que ahora disfruto Pero lo prefiero así con tal de liberarme de la servidumbre intelectual a la que me ha sometido Durante los últimos dos años dos años confirmando artículos que he escrito yo solita dos años impartiendo más docencia de la que me toca sobre asignaturas que ignoro por completo y a alumnos que parecen retrasado
"""
short_text = "Hola, qué tal? Me gustaría invitarte un café"
confidence_values = detector.compute_language_confidence_values(short_text)

print(detector.detect_language_of(short_text))
print(confidence_values[0].language)
