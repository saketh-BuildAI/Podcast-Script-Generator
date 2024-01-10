import requests
import openai
from bs4 import BeautifulSoup
from fastapi import FastAPI
from transformers import pipeline
import re
import nltk

nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import heapq
from string import punctuation

from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
# from gensim.summarization.summarizer import summarize
templates = Jinja2Templates(directory="templates")

app = FastAPI()
links_list = []
headings = []
api_key = "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"
openai.api_key = api_key
model = "gpt-3.5-turbo-instruct"
max_tokens = 500

app.mount("/static", StaticFiles(directory="static"), name="static")

def get_headings_and_links(url: str):
    # url = "https://indianexpress.com/latest-news/"
    response = requests.get(url)
    html_content = response.content

    # Step 3: Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_content, 'lxml')

    # Step 4: Extract the first 5 div tags with class='articles'
    articles_divs = soup.find_all('div', class_='articles')[:5]

    # List to store the links

    # Process and print the content of the selected div tags, nested divs, anchor links, and <p> tag text
    for idx, article_div in enumerate(articles_divs, start=1):
        # print(f"Article {idx}:")

        # Find the nested div with class='img-context' within each 'articles' div
        img_context_div = article_div.find('div', class_='img-context')

        if img_context_div:
            # print("Image Context:")

            # Find the div with class='title' within 'img-context'
            title_div = img_context_div.find('div', class_='title')

            if title_div:
                # Find all anchor tags within the 'title' div
                anchor_tags = title_div.find_all('a')

                # Extract and print the links
                for anchor_tag in anchor_tags:
                    link = anchor_tag.get('href')
                    links_list.append(link)
                    # print(f"Link: {link}")
                    heading_text = anchor_tag.get_text()
                    headings.append(heading_text)

                # Find all <p> tags within the 'img-context' div
                p_tags = img_context_div.find_all('p')

                # Extract and print the text inside each <p> tag
                for p_tag in p_tags:
                    p_text = p_tag.get_text()
                    # print(f"<p> tag text: {p_text}")

                date_div = img_context_div.find('div', class_='date')

                if date_div:
                    date_content = date_div.get_text()
                    # print(f"Date content: {date_content}")

            else:
                print("No 'title' div found within 'img-context'.")

        else:
            print("No 'img-context' div found.")

        # print("\n---\n")
    return headings, links_list


# Print the list of links
# print("List of Links:")
# print(links_list)
# i = 1
# j = 0
# for link in links_list:
#     file_path = f"news-article {i}.txt"
#     with open(file_path, 'w', encoding='utf-8') as file:
#         file.write(f"TITLE:\n{headings[j]}\n")
#         link_response = requests.get(link)
#         link_html_content = link_response.content
#
#         # Parse the HTML content of the link using BeautifulSoup
#         link_soup = BeautifulSoup(link_html_content, 'lxml')
#
#         # Find the div with class='heading-part' within the link's content
#         heading_part_div = link_soup.find('div', class_='heading-part')
#
#         if heading_part_div:
#             # Find the <h2> tag with class='synopsis' within 'heading-part'
#             synopsis_h2 = heading_part_div.find('h2', class_='synopsis')
#
#             if synopsis_h2:
#                 synopsis_content = synopsis_h2.get_text()
#                 file.write(f"SYNOPSIS CONTENT:\n{synopsis_content}\n")
#
#             else:
#                 file.write("No 'synopsis' <h2> tag found within 'heading-part'.\n")
#
#         else:
#             file.write("No 'heading-part' div found within the link's content.\n")
#
#         # Find the div with class='story_details' within the link's content
#         story_details_div = link_soup.find('div', class_='story_details')
#         file.write("CONTENT: \n")
#         if story_details_div:
#             # Find all <p> tags within the 'story_details' div
#             story_details_p_tags = story_details_div.find_all('p')
#
#             # Extract and print the text inside each <p> tag
#             for p_tag in story_details_p_tags:
#                 p_text = p_tag.get_text()
#                 file.write(f"{p_text}\n")
#
#
#         else:
#             file.write("No 'story_details' div found within the link's content.\n")
#
#         # Find the div with class='ev-meter-content ie-premium-content-block' within the link's content
#         ev_meter_content_div = link_soup.find('div', class_='ev-meter-content ie-premium-content-block')
#
#         if ev_meter_content_div:
#             # Find all <p> tags within the 'ev-meter-content' div
#             ev_meter_content_p_tags = ev_meter_content_div.find_all('p')
#
#             # Extract and print the text inside each <p> tag
#             for p_tag in ev_meter_content_p_tags:
#                 p_text = p_tag.get_text()
#                 file.write(f"{p_text}\n")
#
#         else:
#             file.write("No 'ev-meter-content' div found within the link's content.\n")
#
#     i += 1
#     j += 1
#     break


def get_file_content(url: str, index: int):
    titles, urls = get_headings_and_links(url)
    file_path = f"news-article {index}.txt"
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f"TITLE:\n{titles[index]}\n")
        link_response = requests.get(urls[index])
        link_html_content = link_response.content

        # Parse the HTML content of the link using BeautifulSoup
        link_soup = BeautifulSoup(link_html_content, 'lxml')

        # Find the div with class='heading-part' within the link's content
        heading_part_div = link_soup.find('div', class_='heading-part')

        if heading_part_div:
            # Find the <h2> tag with class='synopsis' within 'heading-part'
            synopsis_h2 = heading_part_div.find('h2', class_='synopsis')

            if synopsis_h2:
                synopsis_content = synopsis_h2.get_text()
                file.write(f"SYNOPSIS CONTENT:\n{synopsis_content}\n")

            else:
                file.write("No 'synopsis' <h2> tag found within 'heading-part'.\n")

        else:
            file.write("No 'heading-part' div found within the link's content.\n")

        # Find the div with class='story_details' within the link's content
        story_details_div = link_soup.find('div', class_='story_details')
        file.write("CONTENT: \n")
        if story_details_div:
            # Find all <p> tags within the 'story_details' div
            story_details_p_tags = story_details_div.find_all('p')

            # Extract and print the text inside each <p> tag
            for p_tag in story_details_p_tags:
                p_text = p_tag.get_text()
                file.write(f"{p_text}\n")


        else:
            file.write("No 'story_details' div found within the link's content.\n")

        # Find the div with class='ev-meter-content ie-premium-content-block' within the link's content
        ev_meter_content_div = link_soup.find('div', class_='ev-meter-content ie-premium-content-block')

        if ev_meter_content_div:
            # Find all <p> tags within the 'ev-meter-content' div
            ev_meter_content_p_tags = ev_meter_content_div.find_all('p')

            # Extract and print the text inside each <p> tag
            for p_tag in ev_meter_content_p_tags:
                p_text = p_tag.get_text()
                file.write(f"{p_text}\n")

        else:
            file.write("No 'ev-meter-content' div found within the link's content.\n")

    return file_path


# @app.get("/get_headings_and_links/")
# def get_headings_and_links_endpoint(url: str):
#     return get_headings_and_links(url)
#
#
#
# def remove_control_characters(input_str):
#     return ''.join(char for char in input_str if ord(char) > 31 or ord(char) == 9)
#

def remove_control_characters(content):
    # Define a translation table to remove specific control characters
    control_characters = bytes([0x98, 0x99, 0x80, 0x93])
    translation_table = dict.fromkeys(control_characters, None)

    # Use translate to remove control characters
    cleaned_content = content.translate(translation_table)
    return cleaned_content


#
# def summarize_text(text, max_length=150):
#     summarizer = pipeline("summarization")
#     summarized_text = summarizer(text, max_length=max_length, min_length=50, return_text=True)
#     return summarized_text


# def summarize_text_gensim(text, ratio=0.2):
#     summarized_text = summarize(text, ratio=ratio)
#     return summarized_text

def summarize(text):
    sentence_list = nltk.sent_tokenize(text)

    stopwords = nltk.corpus.stopwords.words('english')
    word_frequencies = {}

    for word in nltk.word_tokenize(text):
        if word not in stopwords and word not in punctuation:
            if word not in word_frequencies.keys():
                word_frequencies[word] = 1
            else:
                word_frequencies[word] += 1
    maximum_frequncy = max(word_frequencies.values())

    for word in word_frequencies.keys():
        word_frequencies[word] = (word_frequencies[word] / maximum_frequncy)

    sentence_scores = {}

    for sent in sentence_list:
        for word in nltk.word_tokenize(sent.lower()):
            if word in word_frequencies.keys():
                if len(sent.split(' ')) < 30:
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

    summary_sentences = heapq.nlargest(7, sentence_scores, key=sentence_scores.get)
    summary = ' '.join(summary_sentences)

    return summary


# Function to generate podcast script using OpenAI GPT-3.5
# def generate_podcast_script(prompt, max_token=500):
#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=prompt,
#         max_tokens=max_token,
#         temperature=0.7
#     )
#     podcast_script = response['choices'][0]['text']
#     return podcast_script


@app.get("/get_file_content/{index}")
def read(index: int, url: str):
    file_name = get_file_content(url, index)
    with open(file_name, "r", encoding="ISO-8859-1") as file:
        file_content = file.read().replace('\n', ' ')
        file_content = file_content.replace('â', ' ')
        cleaned_content = remove_control_characters(file_content)
    return {"file_content": cleaned_content}


@app.get("/get_podcast/{index}")
def read(index: int):
    file_name = f"news-article {index}.txt"
    with open(file_name, "r", encoding="ISO-8859-1") as file:
        file_content = file.read().replace('\n', ' ')
        file_content = file_content.replace('â', ' ')
        cleaned_content = remove_control_characters(file_content)

        max_context_length = 4097
        if len(cleaned_content.split()) > max_context_length:
            # If it exceeds, summarize the text
            summarized_text = summarize(cleaned_content)

            # Use the summarized text as the prompt for podcast script generation

        else:
            summarized_text = cleaned_content

        prompt = f"generate a best podcast script for the following content  {summarized_text}"

        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens
        )

        generated_content = response['choices'][0]['text']
        generated_content = generated_content.replace('\n', ' ')
        generated_content = generated_content.replace('â', ' ')
        generated_content = remove_control_characters(generated_content)

    return generated_content


# @app.get("/")
# def read_root():
#     return {"Hello": "World"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def fun(index: int):
    file_name = f"news-article {index}.txt"
    with open(file_name, "r", encoding="ISO-8859-1") as file:
        file_content = file.read().replace('\n', ' ')
        file_content = file_content.replace('â', ' ')
        cleaned_content = remove_control_characters(file_content)

        max_context_length = 4097
        if len(cleaned_content.split()) > max_context_length:
            # If it exceeds, summarize the text
            summarized_text = summarize(cleaned_content)

            # Use the summarized text as the prompt for podcast script generation

        else:
            summarized_text = cleaned_content

        prompt = f"generate a best podcast script for the following content  {summarized_text}"

        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=max_tokens
        )

        generated_content = response['choices'][0]['text']
        generated_content = generated_content.replace('\n', ' ')
        generated_content = generated_content.replace('â', ' ')
        generated_content = remove_control_characters(generated_content)

    return generated_content


# @app.post("/get_podcastscript", response_class=HTMLResponse)
# async def get_podcastscript(request: Request, url: str = Form(...), index: int = Form(...)):
#     # Implement your logic here to fetch podcast script based on URL and index
#     # For demonstration purposes, we'll just echo the input
#     print(index)
#     # result = f"URL: {url}, Index: {index}"
#     result = fun(index)
#
#     return templates.TemplateResponse("result.html", {"request": request, "result": result})


@app.post("/process_form", response_class=HTMLResponse)
async def process_form(request: Request, url: str = Form(...), index: int = Form(...), action: str = Form(...)):
    if action == "get_content":
        file_name = get_file_content(url, index)
        with open(file_name, "r", encoding="ISO-8859-1") as file:
            file_content = file.read().replace('\n', ' ')
            file_content = file_content.replace('â', ' ')
            cleaned_content = remove_control_characters(file_content)
        result = cleaned_content
        # result = f"Podcast Script - URL: {url}, Index: {index}"
        return templates.TemplateResponse("filecontent.html", {"request": request, "result": result})
    elif action == "get_podcastscript":
        result = fun(index)
        return templates.TemplateResponse("result.html", {"request": request, "result": result})




