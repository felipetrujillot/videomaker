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
def submitTts(text, outDir, female):
    ENDPOINT = "https://tiktok-tts.weilnet.workers.dev"

    voice = "en_us_010"

    if female == True:
        voice = "en_female_makeup"
    try:
        response = requests.post(f"{ENDPOINT}/api/generation", json={
            "text": text,
            "voice": voice
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

    for i in fileArray:
        os.remove(i)

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
        submitTts(inputText, fileDir, female=True)
        fileArray.append(fileDir)

    combinedMp3 = mergeMp3(fileArray, outDir)

    mp3ToSrt(combinedMp3, outDir)

    videoName = mergeVideoSrt("minecraft.mp4", "combined.mp3", "combined.srt", outDir)

    generateImage(title,reddit,outDir)
    
    return videoName



start(
    reddit="relationship_advice",
    title="How do I (29 F) recommunicate to my fiance (39M) why an Infidelity Clause in our Prenup important to me but also not personal to him?",
    story="""
    How do I (29 Female) recommunicate to my fiance (39 Male) why an Infidelity Clause in our Prenup important to me but also not personal to him?.

    My Fiance told me he would ask for a pre nup a year before we got engaged. 
    If I were in his financial situation I would ask for one as well. I am 29 Female and have 500,000 in my own assets/net worth,
    My Dad had multiple affairs throughout his marriage to my mom. 
    Her screams and sobs when she found out are burned into my memory.
    I had never really thought much about prenups until my fiance brought it up again after we got engaged. 
    I did some research and decided that if I have to pay a lawyer to review whatever his lawyer drafts to protect his assets, 
    I want to add an infidelity clause too. I think of it a cathartic nod to the 18 year old version of myself who saw my mom be cheated on as well 
    as the aftermath from all the affairs. I do not think my fiance would cheat. The best I can describe my feelings towards the infidelity clause is like this, 
    It's like if your house catches on fire and burns down, once you get a new one you will always pay extra to have the fire insurance. It will 99.9999% never burn down again, 
    but you still sleep more soundly and you were getting insurance anyway.

    When I brought this up a 3 months ago it was a long conversation but he eventually agreed he was at peace with me having my lawyer add it.
    His lawyer was still drafting the prenup at this time.
    Tonight at dinner he brought up that we need to review the prenup. I said "sure" and that "at a glance it looked fine to me." 
    I then told him that I was still going to have my lawyer add the clause. When I brought this up, 
    he got really upset and started to say things like "this clause is just going to cause more back and forth with the lawyers. 
    I don't want to do that" To which I responded, "well that's why we would come up with the terms together, 
    like how we went over the prenup terms before your lawyer drafted it." 
    Then he said "well we would have to define what counts as cheating, what if you decide to divorce me one day so you just say I cheated." 
    "What if I am alone with someone and you say its cheating?"

    When he said this I was caught off guard. It felt really irrational because in divorce courts you need hard proof of cheating, 
    such as photos, documentation or an omission from the spouse. 
    It's not something you can just say based on feeling and a judge will agree with you. 
    Falsely accusing someone of cheating is also something I would never do. 
    It felt like a bunch of excuses and I was confused on why he was now not okay with it after we had a long discussion a few months ago where he agreed to it. 
    He said "he would never cheat and doesn't even want to think about cheating and adding the clause would make him think about it."
    I understand that your partner asking for an infidelity clause could feel shitty, emotional or like your partner doesn't trust you. 
    I think only people who have been in my shoes could be asked to add it to a prenup and shrug it off without reading too much into it, emotionally.
    It's important to me to add it, 
    I have to pay for a lawyer regardless and I would like my request to be understood and respected in the same way I respected his request for a prenup that protects his assets.
    Thanks for reading this (if anyone does). Looking forward to hearing your thoughts""",
    outDir="nuevo"
)

quit()
""""""
start(  
    reddit="relationship_advice",
    title="Husband (29m) has secret trip planned with AP. When/How should I (25F) ruin it?",
    story="""
Husband (29 male) has secret trip planned with AP. How should I (25 Female) ruin it?.

Husband told me he had a “guys” trip to another state. 
He has a past of infidelity with one certain girl. 
I forgave him, stupidly. 
Something was telling me things weren’t adding up when he informed me of this trip. 
He was still signed into my email. 
I found the tickets. 
It’s to a completely different INTERNATIONAL location. 
I know I’m leaving but any suggestions on the best way to do it? 
I feel spiteful and want to send “enjoy your trip in….” when they land,
just to fuck it up from the start and have all my stuff moved out by the time he gets back. 
She knows he is married as well. 
Both are terrible people but he’s the one I want to hurt. 
I pay for all the bills as I am the main breadwinner. 
He spent 40 bucks on a birthday breakfast for me and is flying her out for hers. 
It’s a spit in the face.

Please don’t tell me to just move out. This has been years of pain. 
I want to do something spiteful I’m sorry

""", 
    outDir="final")
