import asyncio
import aiohttp
from pathlib import Path
import wikipedia
import os
from openai import OpenAI
import re
from google.cloud import storage
from urllib.parse import quote
import time

# Set up your OpenAI API key
client = OpenAI(api_key='###')
storage_client = storage.Client.from_service_account_json('C:/Users/jram1/CapstoneML_Test/eng4k-capstone-server-978cc76d266b.json')
bucket_name = 'wikipedia-images'

def get_topic_from_query(query):
    response = client.chat.completions.create(
        model="gpt-4o",
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
        max_tokens=150
    )
    topic = response.choices[0].message.content
    print(topic)
    return topic

# async def fetch_image_descriptions(session, image_path):
    # for image_file in image_path.glob('*.*'):  # Adjust glob pattern for image types you are handling
    #     file_title = f"File:{image_file.name}"
    #     url = "https://commons.wikimedia.org/w/api.php"
    #     params = {
    #         "action": "query",
    #         "format": "json",
    #         "titles": file_title,
    #         "prop": "imageinfo",
    #         "iiprop": "extmetadata",
    #         "iiextmetadatalanguage": "en"
    #     }
    #     async with session.get(url, params=params) as response:
    #         data = await response.json()
    #         page = next(iter(data['query']['pages'].values()), None)
    #         if page and 'imageinfo' in page:
    #             try:
    #                 extmetadata = page['imageinfo'][0].get('extmetadata', {})
    #                 description = extmetadata.get('ImageDescription', {}).get('value', 'No description available')
    #                 print(f"Description for {image_file.name}: {description}")
    #             except KeyError:
    #                 print(f"No description found for {image_file.name}")
    #         else:
    #             print(f"Failed to fetch description for {image_file.name}")

async def fetch_text_and_images(title, session):
    data_path = Path("data_wiki_text")
    image_path = Path("data_wiki_images")

    text, page = await fetch_text(session, title, data_path)
    # if text:
    #     # print(f"Text for {title} fetched and saved.")
    # else:
    #     # print(f"No text found for {title}")

    await fetch_and_save_images(title, image_path, session)
    # await fetch_image_descriptions(session, image_path)
    return text, image_path

async def fetch_text(session, title, data_path):
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
            text_path = data_path/ f"{title}.txt"
            data_path.mkdir(parents=True, exist_ok=True)
            with open(text_path, "w", encoding='utf-8') as file:
                file.write(page['extract'])
            return page['extract'], page
        return None, None

async def fetch_and_save_images(title, image_path, session):
    page_py = wikipedia.page(title)
    all_image_urls = page_py.images
    # print("\nAll URLs:", all_image_urls)

    filtered_urls = [url for url in all_image_urls if url.endswith((".jpg", ".png", ".svg"))]
    # print("\nFiltered image URLs:", filtered_urls)

    tasks = [download_and_upload_image(session, url, image_path / f"{title}_{url.split('/')[-1]}") for url in filtered_urls]
    await asyncio.gather(*tasks)

async def download_and_upload_image(session, url, image_path):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                image_path.parent.mkdir(parents=True, exist_ok=True)
                with open(image_path, "wb") as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)
                # print(f"\nSaved image to {image_path}")

                # Upload the image to Google Cloud Storage
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(str(image_path))
                blob.upload_from_filename(str(image_path))
                # print(f"\nUploaded image to {bucket_name}/{image_path}")

            else:
                print(f"Failed to download image from {url}")
    except Exception as e:
        print(f"Error downloading image: {e}")
    
# async def download_image(session, url, image_path):
#     try:
#         async with session.get(url) as response:
#             if response.status == 200:
#                 image_path.parent.mkdir(parents=True, exist_ok=True)
#                 with open(image_path, "wb") as f:
#                     while True:
#                         chunk = await response.content.read(1024)
#                         if not chunk:
#                             break
#                         f.write(chunk)
#                 print(f"Saved image to {image_path}")
#             else:
#                 print(f"Failed to download image from {url}")
#     except Exception as e:
#         print(f"Error downloading image: {e}")

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


def rank_images_by_title(query, image_titles, image_urls):
    # print("\nImage titles", image_titles)
    # print("\nImage URLs", image_urls)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful assistant designed to output in JSON."},
            {"role": "user", "content": f"""
             Read the image titles: {image_urls}.
             Rank the titles within the image urls based on their relevance to the user's query: {query}.
             Return the top 3 imageURLs using the following structure
                        
                    (
            "Top URLs": 
                "ImageURL1": "Actual Image URL",
                "ImageURL2": "Actual Image URL",
                "ImageURL3": "Actual Image URL"
            )
         
             """}
        ],
        temperature=0.5,
        max_tokens=200
    )
    top_images = response.choices[0].message.content    
    print("\n", top_images)

async def delete_gcs_directory(bucket_name, prefix):
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    for blob in blobs:
        # print(f"Deleting blob: {blob.name}")
        blob.delete()


async def main():
    async with aiohttp.ClientSession() as session:
        query = input("Enter the query you want to ask: ")
        main_topic = get_topic_from_query(query)
        if main_topic:
            text, image_path = await fetch_text_and_images(main_topic, session)
            # print(f"Information was fetched from the Wikipedia article titled: {main_topic}")
            if text:
                summary = generate_summary(query, text)
                print(f"\nSummary: {summary}")
                image_titles = [img.stem for img in image_path.glob('*.*')]
                image_urls = ["https://storage.googleapis.com/{}/{}".format(bucket_name, quote(str(img))) for img in image_path.glob('*.*')]              
                top_images_with_urls = rank_images_by_title(query, image_titles, image_urls)
            
            else:
                print("Failed to fetch Wikipedia text.")
        else:
            print("Failed to process the query.")
        time.sleep(30)
        await delete_gcs_directory(bucket_name, "data_wiki_images\\")

if __name__ == "__main__":
    asyncio.run(main())