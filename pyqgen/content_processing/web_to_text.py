import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import requests
from requests.models import MissingSchema
import trafilatura
import json


class Web2text:
    def __init__(self) -> None:
        self.tag_black_list = [
            "[document]",
            "noscript",
            "header",
            "html",
            "meta",
            "head",
            "input",
            "script",
            "style",
        ]

    def clean_text(self, *, s: str) -> str:
        clean = re.compile("<.*?>")
        s = re.sub(clean, "", s)
        s = s.replace("\r", " ")
        s = re.sub(r"\.+", ".", s)
        # s = s.replace("\n", " ").replace("\r", " ")
        # s = s.replace(":selected:", "").replace(":unselected:", "")
        # s = s.replace('\"', '')
        # s = s.replace(".", "")
        return s

    def beautifulsoup_extract_text_fallback(self, *, response_content):
        """
        This is a fallback function, so that we can always return a value for text content.
        Even for when both Trafilatura and BeautifulSoup are unable to extract the text from a
        single URL.
        """

        # Create the beautifulsoup object:
        soup = BeautifulSoup(response_content, "html.parser")

        # Finding the text:
        text = soup.find_all(text=True)

        # Remove unwanted tag elements:
        cleaned_text = ""
        # Then we will loop over every item in the extract text and make sure that the beautifulsoup4 tag
        # is NOT in the blacklist
        for item in text:
            if item.parent.name not in self.tag_black_list:
                cleaned_text += "{} ".format(item)

        # Remove any tab separation and strip the text:
        cleaned_text = cleaned_text.replace("\t", "")
        return {"text": cleaned_text.strip()}

    def extract_text_from_single_web_page(self, *, url: str):
        downloaded_url = trafilatura.fetch_url(url)
        try:
            a = trafilatura.extract(
                downloaded_url,
                output_format="json",
                with_metadata=True,
                include_comments=False,
                date_extraction_params={
                    "extensive_search": True,
                    "original_date": True,
                },
            )
        except AttributeError:
            a = trafilatura.extract(
                downloaded_url,
                output_format="json",
                with_metadata=True,
                date_extraction_params={
                    "extensive_search": True,
                    "original_date": True,
                },
            )
        if a:
            json_output = json.loads(a)
            output = {
                key: json_output[key]
                for key in ["title", "source", "source-hostname", "text"]
            }
            output["text"] = self.clean_text(s=output["text"])
            output["scraper"] = "trafilatura"
            return output
        else:
            try:
                resp = requests.get(url)
                # We will only extract the text from successful requests:
                if resp.status_code == 200:
                    output = self.beautifulsoup_extract_text_fallback(
                        response_content=resp.content
                    )
                    output["scraper"] = "bs4"
                    output["text"] = self.clean_text(s=output["text"])
                    return output
                else:
                    # This line will handle for any failures in both the Trafilature and BeautifulSoup4 functions:
                    return np.nan
            # Handling for any URLs that don't have the correct protocol
            except MissingSchema:
                return np.nan
