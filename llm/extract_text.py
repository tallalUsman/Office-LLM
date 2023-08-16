from typing import Generator, Tuple, Iterator
from pathlib import Path
import requests
import io
import logging
from bs4 import BeautifulSoup

#from llm import config
import config


logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

def extract(urls: list,
            extraction_path: Path) -> None:
    """Extract text from a url for each episode
    Args:
        urls (list): url's of the page holding the episodes.
    """
    conversation = []
    # Send a GET request to the server and store the response
    for url in urls:
        response = requests.get(url)

        # Print the HTML content of the page
        html_doc = response.text

        # Create a Beautiful Soup object
        soup = BeautifulSoup(html_doc, 'html.parser')

        # Find all <a> tags
        for a_tag in soup.find_all('a'):
            # Extract the href attribute
            url = a_tag.get('href')
            if url and (url.startswith('./viewtopic.php?t=') or url.startswith('./viewtopic.php?p=')) and '#' not in url:
                new_url = url.replace('./viewtopic.', 'https://transcripts.foreverdreaming.org/viewtopic.')
                print(new_url)
                response = requests.get(new_url)
                # Print the HTML content of the page
                html_doc = response.text
                # Parse the HTML
                soup = BeautifulSoup(html_doc, 'html.parser')

                # Find all 'strong' tags with class 'text-strong'
                elements = soup.find_all('strong', class_='text-strong')

                for element in elements:
                    # Extract the speaker's name
                    speaker = element.text

                    # Extract the text after the 'strong' tag until the next 'br' tag
                    text = element.next_sibling

                    conversation.append(f'{speaker}{text}')
            
    conversation = '\n'.join(conversation)
    to_text_file(conversation, path=extraction_path)

def to_text_file(dialogue, path: Path) -> None:
    LOGGER.info(f'Start writing to {path}')
    # We append text to the existing file with "a" mode (append)
    with open(path, 'a') as f:
        f.write(dialogue)

    LOGGER.info(f'Finished writing to {path}')


if __name__ == "__main__":
    extract(urls=config.urls, 
            extraction_path=config.extraction_path)