def clean_text(text: str):

    if not text:
        return ""

    text = text.replace("\n", " ")

    text = " ".join(text.split())

    return text.strip()