"""Sarvam voice model catalog: available STT/TTS models, speakers, and languages.

Single source of truth for what the voice APIs can run. Kept here so the
speech service, the orchestrator bridge, and the chat UI all agree on the
selectable options without hardcoding them in multiple places.
"""

# ── Speech-to-Text (STT) ──
# See https://docs.sarvam.ai/api-reference/speech-to-text/transcribe
STT_MODELS: list[str] = [
    "saaras:v3",
    "saaras:v4",
    "sarvam-1",
    "sarvam-1-20x-hi-en-2025-03-04",
]
STT_DEFAULT_MODEL: str = "saaras:v3"

STT_MODES: list[str] = [
    "transcribe",
    "translate",
    "verbatim",
    "translit",
    "codemix",
]

STT_LANGUAGES: list[str] = [
    "unknown",
    "hi-IN",
    "bn-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "od-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
    "en-IN",
    "gu-IN",
    "as-IN",
    "ur-IN",
    "ne-IN",
    "kok-IN",
    "ks-IN",
    "sd-IN",
    "sa-IN",
    "sat-IN",
    "mni-IN",
    "brx-IN",
    "mai-IN",
    "doi-IN",
]

# ── Text-to-Speech (TTS) ──
# See https://docs.sarvam.ai/api-reference/text-to-speech/convert
TTS_MODELS: list[str] = [
    "bulbul:v2",
    "bulbul:v3",
]
TTS_DEFAULT_MODEL: str = "bulbul:v2"

# Speakers per model (the API rejects speakers that don't match the model).
TTS_SPEAKERS: dict[str, list[str]] = {
    "bulbul:v2": [
        "anushka",
        "manisha",
        "vidya",
        "arya",
        "abhilash",
        "karun",
        "hitesh",
    ],
    "bulbul:v3": [
        "shubh",
        "aditya",
        "ritu",
        "priya",
        "neha",
        "rahul",
        "pooja",
        "rohan",
        "simran",
        "kavya",
        "amit",
        "dev",
        "ishita",
        "shreya",
        "ratan",
        "varun",
        "manan",
        "sumit",
        "roopa",
        "kabir",
        "aayan",
        "ashutosh",
        "advait",
        "anand",
        "tanya",
        "tarun",
        "sunny",
        "mani",
        "gokul",
        "vijay",
        "shruti",
        "suhani",
        "mohit",
        "kavitha",
        "rehan",
        "soham",
        "rupali",
    ],
}

# Default speaker per TTS model (matches the API defaults).
TTS_DEFAULT_SPEAKER: dict[str, str] = {
    "bulbul:v2": "anushka",
    "bulbul:v3": "shubh",
}

TTS_LANGUAGES: list[str] = [
    "bn-IN",
    "en-IN",
    "gu-IN",
    "hi-IN",
    "kn-IN",
    "ml-IN",
    "mr-IN",
    "od-IN",
    "pa-IN",
    "ta-IN",
    "te-IN",
]
TTS_DEFAULT_LANGUAGE: str = "en-IN"
