import requests

api_key = "YOUR API KEY"

def bt(example, language):
    url = f"https://translation.googleapis.com/language/translate/v2?key={api_key}&q={example}&target={language}&source=en"
    response = requests.get(url)

    translated_text = response.json()['data']['translations'][0]['translatedText']

    url = f"https://translation.googleapis.com/language/translate/v2?key={api_key}&q={translated_text}&target=en&source={language}"
    response = requests.get(url)
    back_translated_text = response.json()['data']['translations'][0]['translatedText']

    return back_translated_text