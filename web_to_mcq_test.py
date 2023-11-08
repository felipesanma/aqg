from pyqgen import PyQGen


urls = [
    "https://website.understandingdata.com/",
    "https://sempioneer.com/",
    "https://blog.chasquillaengineer.com/p/las-preguntas-esenciales-para-disenar",
    "https://understandingdata.com/posts/how-to-extract-the-text-from-multiple-webpages-in-python/",
]

pyqgen = PyQGen()
for url in urls:
    text = pyqgen.web_to_text.extract_text_from_single_web_page(url=url)
    print(text)
