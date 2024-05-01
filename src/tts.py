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
            "voice": "en_us_010"
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

    groupedArray = string_parser(story)
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

""""""
start(  
    reddit="relationship_advice",
    title="Husband (29m) has secret trip planned with AP. When/How should I (25F) ruin it?",
    story="""
Husband (29 male) has secret trip planned with AP. When/How should I (25 Female) ruin it?.

Husband told me he had a “guys” trip to another state. He has a past of infidelity with one certain girl. I forgave him, stupidly. Something was telling me things weren’t adding up when he informed me of this trip. He was still signed into my email. I found the tickets. It’s to a completely different INTERNATIONAL location. I know I’m leaving but any suggestions on the best way to do it? I feel spiteful and want to send “enjoy your trip in….” when they land, just to fuck it up from the start and have all my stuff moved out by the time he gets back. She knows he is married as well. Both are terrible people but he’s the one I want to hurt. I pay for all the bills as I am the main breadwinner. He spent 40 bucks on a birthday breakfast for me and is flying her out for hers. It’s a spit in the face.

Please don’t tell me to just move out. This has been years of pain. I want to do something spiteful I’m sorry

""", outDir="final")
