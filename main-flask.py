import os
import re
import json
import asyncio
import aiohttp
import time
from flask import Flask, request, jsonify
from openai import OpenAI
from fuzzywuzzy import fuzz
import time

# Initialize Flask app
app = Flask(__name__)

# Set up your OpenAI API key
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_topic_from_query(query):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""
             Your tasks are the following:
             0. Read the user's query: {query} 
             1. Then you will identify and extract the principal main topic of the user's query (NOTE: THERE CAN ONLY BE ONE PRINCIPAL/MAIN )
             2. Check if the user's query is inappropriate or nonsense input.
             3. You must return the main topic only without any other text, just the sole main topic, as standalone text.
             4. DO NOT include anything else other than the main topic, NO SYMBOLS, NO DECORATORS, NO CURLEY BRACKETS, JUST THE MAIN TOPIC!
             """}
        ],
        temperature=0.5,
        max_tokens=50
    )
    topic = response.choices[0].message.content.strip()
    print(topic)
    return topic

async def search_wikipedia(topic, session):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "format": "json"
    }
    async with session.get(url, params=params) as response:
        data = await response.json()
        if 'query' in data and 'search' in data['query']:
            search_results = data['query']['search']
            if search_results:
                return search_results[0]['title']
        return None

async def fetch_text_and_images(title, session):
    text_task = fetch_text(session, title)
    image_titles_urls_task = fetch_image_titles_urls(session, title)
    
    text, image_titles_urls = await asyncio.gather(text_task, image_titles_urls_task)
    images = await fetch_images(session, image_titles_urls)
    return text, images, image_titles_urls

async def fetch_text(session, title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",
    }
    async with session.get(url, params=params) as response:
        data = await response.json()
        page = next(iter(data["query"]["pages"].values()), None)
        if page and 'extract' in page:
            return page['extract']
        return None

async def fetch_image_titles_urls(session, title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "images",
        "imlimit": "max"
    }
    async with session.get(url, params=params) as response:
        data = await response.json()
        page = next(iter(data["query"]["pages"].values()), None)
        if page and 'images' in page:
            image_titles = [img['title'] for img in page['images']]
            image_titles = [img for img in image_titles if img.lower().endswith(('.png', '.jpg', '.jpeg', '.svg'))]
            image_info_tasks = [fetch_image_info(session, img_title) for img_title in image_titles]
            image_info_list = await asyncio.gather(*image_info_tasks)
            image_titles_urls = [info for info in image_info_list if info]
            return image_titles_urls
        return []

async def fetch_image_info(session, title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
    }
    async with session.get(url, params=params) as response:
        data = await response.json()
        page = next(iter(data["query"]["pages"].values()), None)
        if page and 'imageinfo' in page:
            image_info = page['imageinfo'][0]
            if 'url' in image_info and image_info['url'].startswith('http'):
                caption = image_info['extmetadata'].get('ImageDescription', {}).get('value', 'No caption available')
                description = image_info['extmetadata'].get('ObjectName', {}).get('value', 'No description available')
                if is_plain_text(caption):
                    return {'title': title, 'url': image_info['url'], 'description': description, 'caption': caption}
                else:
                    return {'title': title, 'url': image_info['url'], 'description': description, 'caption': "No suitable image description returned."}
        return None

async def fetch_images(session, image_titles_urls):
    images = []
    for image_info in image_titles_urls:
        url = image_info['url']
        async with session.get(url) as response:
            if response.status == 200:
                images.append({'title': image_info['title'], 'url': url, 'data': await response.read(), 'description': image_info['description'], 'caption': image_info['caption']})
            else:
                print(f"Failed to fetch image: {image_info['title']} with status: {response.status}")
    return images

def rank_images_by_relevance(query, image_titles_urls):
    cleaned_query = query.lower()
    ranked_images = []

    for image_info in image_titles_urls:
        title = re.sub(r'^File:|\.jpg$|\.png$|\.gif$|\.svg$|\.ogv$|\.ogg$', '', image_info['title']).replace('_', ' ').strip()
        url = image_info['url']
        cleaned_title = title.lower()
        score = fuzz.partial_ratio(cleaned_query, cleaned_title)
        ranked_images.append((title, url, score, image_info['description'], image_info['caption']))

    ranked_images.sort(key=lambda x: x[2], reverse=True)
    
    top_images = [(title, url, description, caption) for title, url, score, description, caption in ranked_images if score > 0]
    
    print("Top ranked image URLs:")
    for title, url, description, caption in top_images[:3]:
        print(f"Image Title: {title}, Image URL: {url}, Description: {description}, Caption: {caption}")
    return top_images

def generate_summary(query, text, max_tokens=300, temperature=0.5):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": """
                You are a skilled summarizer specializing in Wikipedia content. Your goal is to craft concise, informative summaries that directly answer the user's query. 
                Focus on the most relevant information and present it in a way that enhances the user's understanding of the topic.
            """},
            {"role": "user", "content": f"""
                Question: {query}

                Wikipedia Text:

                {text}

                Summary:
            """}
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )
    summary = response.choices[0].message.content.strip()
    return summary

async def process_query(query):
    async with aiohttp.ClientSession() as session:
        main_topic = get_topic_from_query(query)
        if main_topic:
            article_title = await search_wikipedia(main_topic, session)
            if article_title:
                text, images, image_titles_urls = await fetch_text_and_images(article_title, session)
                if text:
                    summary = await asyncio.to_thread(generate_summary, query, text)
                    top_images = rank_images_by_relevance(main_topic, image_titles_urls)
                    
                    top_three_images = {}
                    for i, (title, url, description, caption) in enumerate(top_images[:3]):
                        top_three_images[f"Image {i+1}"] = {"Title": title, "URL": url, "Description": description, "Caption": caption}
                    
                    json_response = {
                        "Summary": summary,
                        "Top Three Images": top_three_images
                    }
                    
                    return json_response
                else:
                    return {"error": "Failed to fetch Wikipedia text."}
            else:
                return {"error": "No relevant Wikipedia article found."}
        else:
            return {"error": "Failed to process the query."}

def is_plain_text(text):
    return not bool(re.search(r'<[^>]+>', text))

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.json
    query = data.get('query', '')
    print("Received query:", query)  # Debugging print statement

    start_time = time.time()  # Record the start time
    print("Start time:", start_time)  # Debugging print statement

    if not query:
        return jsonify({"error": "No query provided."}), 400
    
    try:
        result = asyncio.run(process_query(query))
        end_time = time.time()  # Record the end time
        print("End time:", end_time)  # Debugging print statement
        print(f"Time taken: {end_time - start_time} seconds")  # Print the duration    
        return jsonify(result)
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": "An error occurred while processing the query."}), 500


if __name__ == "__main__":
    app.run(debug=True)
