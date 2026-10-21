


import moviepy
import whisper
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#from openai import APIError, RateLimitError
#from openai import OpenAI
#ssl._create_default_https_context = ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

#secret key = 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
#client = OpenAI(api_key="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")

from transformers import pipeline

semantic_analyzer = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    top_k=None
)
print(semantic_analyzer("I will kill you"))



#Extract audio from a video
def extract_audio(video_path, audio_path):
    video = moviepy.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path, codec='mp3')
    print("✅ Audio extracted to:", audio_path)  
    video.close()

#Convert audio → text
def transcribe_audio(audio_path):
    model = whisper.load_model("small")
    result = model.transcribe(audio_path)
    text = result["text"]
    print("📝 Transcription:\n", text)
   
    phrases = []
    for seg in result["segments"]:
        phrases.append({
            "text": seg["text"].strip(),
            "start": seg["start"],
            "end": seg["end"]
        })

    return phrases





def analyse_semantique(text):
    resultats = semantic_analyzer(text)

    scores = {
        "toxicity": 0.0,
        "hate": 0.0,
        "violence": 0.0
    }

    for r in resultats[0]:
        label = r["label"].lower()
        score = r["score"]

        if "hate" in label:
            scores["hate"] = max(scores["hate"], score)
        if "toxic" in label:
            scores["toxicity"] = max(scores["toxicity"], score)
        if "violence" in label:
            scores["violence"] = max(scores["violence"], score)

    return scores





def heuristiques(phrase, historique):
    score = 0

    # répétition
    repetitions = sum(1 for p in historique if p["text"] == phrase["text"])
    if repetitions >= 2:
        score += 1

    # intensité (ponctuation, cris)
    if "!" in phrase["text"]:
        score += 1
    if phrase["text"].isupper():
        score += 2

    return score






def contexte(phrases, index, window=2):
    start = max(0, index - window)
    end = min(len(phrases), index + window + 1)
    return " ".join(p["text"] for p in phrases[start:end])





def score_global(semantic, heuristique):
    score = 0

    if semantic["toxicity"] > 0.7:
        score += 3
    if semantic["hate"] > 0.6:
        score += 3
    if semantic["violence"] > 0.6:
        score += 3

    score += heuristique
    return score








def analyser_contenu(audio_path):
    phrases = transcribe_audio(audio_path) 
    alertes = []

    for i, phrase in enumerate(phrases):
        semantic = analyse_semantique(phrase["text"])
        heur = heuristiques(phrase, phrases[:i])
        ctx = contexte(phrases, i)
        score = score_global(semantic, heur)

        if score >= 5:
            alertes.append({
                "text": phrase["text"],
                "start": phrase["start"],
                "end": phrase["end"],
                "score": score,
                "semantic": semantic,
                "context": ctx
            })

    return alertes


















# ---------- RUN EVERYTHING ----------
video_path = "journalTélévisé.mp4"
audio_path = "audio.mp3"

extract_audio(video_path, audio_path)
analyser_contenu(audio_path)

#You are getting this error only because the moderation endpoint is blocking you :
#openai.RateLimitError: Too Many Requests
#You can't avoid it with your current plan : the moderation API is extremely limited for free or no-credit accounts.