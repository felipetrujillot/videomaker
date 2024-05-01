import requests
import base64
from pydub import AudioSegment
import os
from tempfile import mktemp
import torch
from transformers import pipeline
from ffutils import ffprog
import shutil
from pathlib import Path
from mutagen.mp3 import MP3
from images import generateImage

"""
Uploads a text to API and return a mp3 file
"""
def submitTts(text, outDir):
    ENDPOINT = "https://tiktok-tts.weilnet.workers.dev"
    try:
        response = requests.post(f"{ENDPOINT}/api/generation", json={
            "text": text,
            "voice": "en_male_jomboy"
        })

        data = response.json()
        audio_base64 = data["data"]
        audio_bytes = base64.b64decode(audio_base64)

        with open(outDir, "wb") as audio_file:
            audio_file.write(audio_bytes)

        return outDir

    except Exception as e:
        raise e

"""
Receives a full text and it returns it on groups of
300 characters
"""
def string_parser (text):

    splittedText = text.split(' ')
    totalCharacters = 0
    newArr = []
    newDict = []

    for value in range(len(splittedText)):

        i = splittedText[value]
        
        if(totalCharacters + len(i) < 300 ):
            newArr.append(i)
            totalCharacters += len(i) + 1

            if(totalCharacters > 280):
                newString = ''

                for a in newArr:
                    newString += a + ' '

                #print(newString)
                #print(len(newString))
                newDict.append(newString)

                newArr = []
                totalCharacters = 0 


        if value + 1 == len(splittedText):

            newString = ''

            for a in newArr:
                newString += a + ' '

            newDict.append(newString)
            #print(newString)
            #print(len(newString))
            newArr = []
            totalCharacters = 0


    return newDict
    
"""
"""
def get_mp3_length(file_path):
    audio = MP3(file_path)
    length_in_seconds = audio.info.length
    return int(length_in_seconds) + 1
                    
"""
"""
def mergeVideoSrt(videoName, audioName,srtName, outDir):

    audioLength = get_mp3_length(outDir + "/" + audioName)
    videoName = Path(videoName).absolute()
    srtName = Path(outDir+ "/" + srtName).absolute()
    audioName = Path(outDir+ "/" + audioName).absolute()

    if outDir:
        outDir = Path(outDir).absolute()
        outDir.mkdir(parents=True, exist_ok=True)
    else:
        outDir = Path(os.getcwd())

    videoOut = outDir / f"{videoName.stem}_out.mp4"
   
    ffprog(
        ["ffmpeg", "-y", "-i", str(videoName), "-i", str(audioName), "-vf",
        f"subtitles={str(srtName.name)}:force_style='Fontname=Arial,Fontsize=30,OutlineColour=&H80000000,PrimaryColour=&H03fcff,BorderStyle=4,"
        "BackColour=&H80000000,Outline=1,Shadow=0,MarginV=10,Alignment=2,Bold=-1'", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0", "-ss" , "00:00:00", "-t", str(audioLength),
        str(videoOut)],
        cwd=str(srtName.parent),
        desc=f"Burning subtitles and audio into video",
    )
    return videoOut

"""
Receives an array of files and return an array of all of them combined
"""
def mergeMp3(fileArray,fileDir):
    combined_audio = AudioSegment.empty()

    for i in fileArray:
        audio_segment = AudioSegment.from_file(os.path.join(i))
        combined_audio += audio_segment

    fileName = fileDir + "/combined.mp3"
    combined_audio.export(fileName, format='mp3')
    return fileName

"""
"""
def seconds_to_srt_time_format(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

"""
Receives an mp3 and create an srt
"""
def mp3ToSrt(fileName, outDir):

    if outDir:
        outDir = Path(outDir).absolute()
        outDir.mkdir(parents=True, exist_ok=True)
    else:
        outDir = Path(os.getcwd())

    fileName = Path(fileName).absolute()

    audio_file = mktemp(suffix=".aac", dir=outDir)

    ffprog(
        ["ffmpeg", "-y", "-i", str(fileName), "-vn", "-c:a", "aac", audio_file],
        desc=f"Extracting audio from video",
    )

    temp_srt = mktemp(suffix=".srt", dir=outDir)

    model = "medium"
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32

    print(f"{device.upper()} is being used for this transcription, this process may take a while.")

    pipe = pipeline(
        "automatic-speech-recognition",
        model=f"openai/whisper-{model}",
        torch_dtype=dtype,
        device=device,
        model_kwargs={"attn_implementation": "sdpa"},
    )

    if device == "mps":
        torch.mps.empty_cache()

    outputs = pipe(
        audio_file,
        chunk_length_s=30,
        batch_size=24,
        generate_kwargs={"task": "transcribe", "language": "english"},
        return_timestamps=True,
    )

    with open(temp_srt, "w", encoding="utf-8") as f:
        for index, chunk in enumerate(outputs['chunks']):
            start_time = seconds_to_srt_time_format(chunk['timestamp'][0])
            end_time = seconds_to_srt_time_format(chunk['timestamp'][1])
            f.write(f"{index + 1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{chunk['text'].strip()}\n\n")

    os.remove(audio_file)
    srt_filename = outDir / f"{fileName.stem}.srt"
    shutil.move(temp_srt, srt_filename)

    return srt_filename

"""
"""
def start(reddit, title, story, outDir):

    newOut = Path(outDir).absolute()
    newOut.mkdir(parents=True, exist_ok=True)

    groupedArray = string_parser(title + ". " + story)
    fileArray = []

    for i in range(len(groupedArray)):
        fileDir = outDir +  "/" + str(i) + ".mp3"
        inputText = groupedArray[i]
        submitTts(inputText, fileDir)
        fileArray.append(fileDir)

    combinedMp3 = mergeMp3(fileArray, outDir)

    mp3ToSrt(combinedMp3, outDir)

    videoName = mergeVideoSrt("minecraft.mp4", "combined.mp3", "combined.srt", outDir)

    generateImage(title,reddit,outDir)
    
    return videoName

start(  
    "relationship_advice",
    "My wife (32 Female) just walked out on me (36 Male) with zero explanation and I'm lost",
      ""
    """
We have "talked" a couple times now. Each time I'm trying to give her time to speak to me but it still doesn't make any sense. We cry, she says she still cares but can't be with me, I fall eternally deeper in despair.

She said even before the wedding she felt like things were off and instead of talking to me, she just put it aside and figured things would get better on there own. I'm still asking what did I do and get the "you were nothing but amazing" and it wasn't my fault.

Then she hit me yesterday with the "when are we selling the house" talk. She says she cannot move back in (I offered to just sleep in the basement) and needs to find a place asap. Am I insane to think this is going way too fast? It's barely been over a week and I've had no time to grieve, to heal, to learn how to do this on my own again. I've been even worse since she dropped that news.

She also offered to cover my half of the mortgage because I've not been to work since she left me, then today she hits me with the "actually...." she has missed no work. Fuck I don't even think she's missing sleep.

I was really hoping for something, anything to give me hope for the future but like all I see is a void these days. I always made it OUR future but without her I don't know what to do.

Tomorrow I at least have my first therapist appointment. I hope it helps. The worst is what's the lesson from all this? Don't rely on or love anyone ever again?

Please go home tonight, tell your spouse you love them, and give them a hug like you never wanna let go.
""", "final")
