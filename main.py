import requests
import openai
from bs4 import BeautifulSoup
from fastapi import FastAPI
from transformers import pipeline
import re
import nltk
import pandas as pd

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
api_key = ""

openai.api_key = api_key
model = "gpt-3.5-turbo-instruct"
max_tokens = 500

app.mount("/static", StaticFiles(directory="static"), name="static")


def get_headings_and_links(url: str, count: int):
    response = requests.get(url)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'lxml')
    articles_divs = soup.find_all('div', class_='articles')[:count]
    for idx, article_div in enumerate(articles_divs, start=1):
        # print(f"Article {idx}:")


        img_context_div = article_div.find('div', class_='img-context')

        if img_context_div:



            title_div = img_context_div.find('div', class_='title')

            if title_div:

                anchor_tags = title_div.find_all('a')


                for anchor_tag in anchor_tags:
                    link = anchor_tag.get('href')
                    links_list.append(link)

                    heading_text = anchor_tag.get_text()
                    headings.append(heading_text)


                p_tags = img_context_div.find_all('p')


                for p_tag in p_tags:
                    p_text = p_tag.get_text()


                date_div = img_context_div.find('div', class_='date')

                if date_div:
                    date_content = date_div.get_text()


            else:
                print("No 'title' div found within 'img-context'.")

        else:
            print("No 'img-context' div found.")


    return headings, links_list



def get_file_content(url: str, index: int, count: int):
    titles, urls = get_headings_and_links(url, count)
    file_path = f"news-article {index}.txt"
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(f"TITLE:\n{titles[index]}\n")
        link_response = requests.get(urls[index])
        link_html_content = link_response.content


        link_soup = BeautifulSoup(link_html_content, 'lxml')


        heading_part_div = link_soup.find('div', class_='heading-part')

        if heading_part_div:

            synopsis_h2 = heading_part_div.find('h2', class_='synopsis')

            if synopsis_h2:
                synopsis_content = synopsis_h2.get_text()
                file.write(f"SYNOPSIS CONTENT:\n{synopsis_content}\n")

            else:
                file.write("No 'synopsis' <h2> tag found within 'heading-part'.\n")

        else:
            file.write("No 'heading-part' div found within the link's content.\n")


        story_details_div = link_soup.find('div', class_='story_details')
        file.write("CONTENT: \n")
        if story_details_div:

            story_details_p_tags = story_details_div.find_all('p')


            for p_tag in story_details_p_tags:
                p_text = p_tag.get_text()
                file.write(f"{p_text}\n")


        else:
            file.write("No 'story_details' div found within the link's content.\n")


        ev_meter_content_div = link_soup.find('div', class_='ev-meter-content ie-premium-content-block')

        if ev_meter_content_div:

            ev_meter_content_p_tags = ev_meter_content_div.find_all('p')


            for p_tag in ev_meter_content_p_tags:
                p_text = p_tag.get_text()
                file.write(f"{p_text}\n")

        else:
            file.write("No 'ev-meter-content' div found within the link's content.\n")

    return file_path




def remove_control_characters(content):
    control_characters = bytes([0x98, 0x99, 0x80, 0x93])
    translation_table = dict.fromkeys(control_characters, None)
    cleaned_content = content.translate(translation_table)
    return cleaned_content




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


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



@app.post("/process_form", response_class=HTMLResponse)
async def process_form(request: Request, url: str = Form(...), index: int = Form(...), count: int = Form(...),
                       action: str = Form(...),
                       ):
    if action == "get_content":
        file_name = get_file_content(url, index-1, count)
        with open(file_name, "r", encoding="ISO-8859-1") as file:
            file_content = file.read().replace('\n', ' ')
            file_content = file_content.replace('â', ' ')
            cleaned_content = remove_control_characters(file_content)
        result = cleaned_content
        # result = f"Podcast Script - URL: {url}, Index: {index}"
        return templates.TemplateResponse("filecontent.html", {"request": request, "result": result})
    elif action == "get_podcastscript":
        result = fun(index-1)
        return templates.TemplateResponse("result.html", {"request": request, "result": result})

    elif action == "generate_excel":
        titles, urls = get_headings_and_links(url, count)
        data_list = []
        for i in range(count):
            file_name = get_file_content(url, i, count)
            with open(file_name, "r", encoding="ISO-8859-1") as file:
                file_content = file.read().replace('\n', ' ')
                file_content = file_content.replace('â', ' ')
                cleaned_content = remove_control_characters(file_content)

                max_context_length = 4097
                if len(cleaned_content.split()) > max_context_length:

                    summarized_text = summarize(cleaned_content)



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

                data_dict = {"Heading": titles[i], "URL": urls[i], "Content": cleaned_content,
                             "Script": generated_content}
                data_list.append(data_dict)

        df = pd.DataFrame(data_list)
        excel_file_path = 'excel-data.xlsx'
        df.to_excel(excel_file_path, index=False)

        return templates.TemplateResponse("excel_gen.html",
                                          {"request": request, "result": "File Downloaded Successfully"})

