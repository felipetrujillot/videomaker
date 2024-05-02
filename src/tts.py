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

            if(totalCharacters > 260):
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

    model = "large"
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
def start(reddit, title, story, outDir, female):

    newOut = Path(outDir).absolute()
    newOut.mkdir(parents=True, exist_ok=True)

    groupedArray = string_parser(story)
    fileArray = []

    for i in range(len(groupedArray)):
        fileDir = outDir +  "/" + str(i) + ".mp3"
        inputText = groupedArray[i]
        submitTts(inputText, fileDir, female)
        fileArray.append(fileDir)

    combinedMp3 = mergeMp3(fileArray, outDir)

    mp3ToSrt(combinedMp3, outDir)

    videoName = mergeVideoSrt("minecraft.mp4", "combined.mp3", "combined.srt", outDir)

    generateImage(title,reddit,outDir)
    
    return videoName


start(
    reddit="relationship_advice",
    title="FINAL UPDATE: She stole again. I (26M) threw my pregnant girlfriend (22F) out because she refuses to pay rent or her share of the bills?",
    story="""

        FINAL UPDATE: She stole again. I (26 Male) threw my pregnant girlfriend (22 Female) out because she refuses to pay rent or her share of the bills?

        Original story:

        Long story short, my girlfriend and I have been living together for around 10 months.
        When she first moved in she insisted on paying rent and I was reluctant to charge her if it didn’t work out but she forced it and paid a month.
        Then I found out she’s struggling for money, unable to pay for things,
        is in a lot of debt and lives month to month. She agreed with me that she’d start when she clears the debt.
        Fast forward to Christmas I find out she’s been stealing my clothes to give to her family as gifts (another post on here). 
        She lied for 2 weeks blaming me until I showed her footage of her taking the things from the camera in the living room (to watch the dog when I’m out).

        I later then discover through letters and texts I’ve seen appear on her phone she’s been doing nothing to pay any of it off,
        so I confront her. She tells me and shows me messages that her mother and sisters constantly guilt trip her into giving them money and have for years.

        They’ll message her on pay day asking for it and she feels bad saying no, 
        despite non of them ever paying it back. Her mum alone owes her over £6000. 
        She has taken a ton of loans out for her family and they leave her with the debt and don’t pay it back. 
        Luckily her credit is now at the point where nobody will loan to her but she still tries and does it for them. 
        I also find out (I went through her finances, 
        yes I shouldn’t have but something wasn’t adding up and I was being lied to) that in the space of 20 minutes she spent £300 on gambling sites. 
        All during this time she isn’t paying a penny towards rent, bills anything. She’ll occasionally buy food shopping or trips out to Starbucks. 
        I tell her enough is enough and she needs to start paying her way. 
        If she can give handouts to her family and gamble she can pay for where she lives and she’s taken me for a ride when she should’ve been saving and clearing debts.

        I make roughly 5x what she does but I’ve been fair in that the bills are split proportionally to income. 
        She’ll earn £1400 per month and pays £600 which includes rent and her share of the bills. 
        I take on the rest which is substantially more but I believe it’s not fair to take more.

        On the 1st of this month she tells me she can’t pay rent. She says she’s paid out too much on our trips to Starbucks, 
        food shopping and I’ll get it when I get it but she doesn’t understand why I need it this month when she’s lived for free the past 9 months anyway. 
        I’ve asked her to explain where her money has exactly gone but she tells me I’m controlling and it’s non of my business. 
        In fairness she will pay when we go food shopping but rarely in comparison to me. 
        I’ve kicked her out as of yesterday and told her she needs to find somewhere to live. 
        She is however pregnant and she’s using that card as a way to guilt trip me and make out I’ve thrown out her and my child onto the streets.

        In my opinion she is taking me for a ride and prioritising her family that is using her over her own family she’s started? What’s the solution here to getting her to see she’s not treating me fairly?

        TLDR: Girlfriend hasn’t paid rent for 9 months whilst she was supposed to be clearing debts. 
        Instead she was giving money to her family, gambling and I’ve thrown her out because she’s refusing to pay again. She is pregnant.

        Update:

        I sat her down and gave her an ultimatum early last week. 
        I explained to her that we are a family, and became a family when she decided to have a baby with me. 
        I told her if we’re going to stay together she’s going to have to be a lot more open, 
        contribute and no more taking on debt she can’t afford which brings it to my door when she can’t pay. 
        I also told her I want to see her bank statements because I suspect she has a gambling problem and is in some serious debt. 
        She agreed to all of this and committed to showing me the bank statements when I ask and says going forward she’ll pay towards bills. 
        I believe she’s turned a corner and start getting along with her better and she moves back in.

        As I was sat next to her phone last night when she went to grab a drink her phone lit up with a text message. 
        It read “loan accepted by X lender, click here to accept.” 
        I immediately called her out and she starts crying telling me she has no money left again for the month and she’s had to resort to payday loans for some money. 
        I tell her she should’ve have come to me and tell her I explicitly said no more loans.

        She also tells me she won’t be able to afford to pay towards bills again. 
        She works full time and brings home around £1400-£1200 a month dependant on hours but a lot of the time she phones sick so gets sick pay which is a lot less. 
        I ask to see her bank statements and she refuses telling me I’m being controlling by asking when she’s told me and I don’t need to see them. 
        That’s the last straw for me. I’m almost certain she’s been giving it away at this point again or gambling. 
        I give her a scenario: “Your baby is starving and needs food and there’s non in the house, what are you going to do.”

        She replies “you’ll have to pay.” That’s fine I’ll happily support my son I tell her because the mother is clearly a deadbeat. 
        So I ask to see her Facebook Messenger to see if her family have been hitting her up for free money again and conveniently all of the family members that borrow from her have the chats cleared (she says she deletes them to be tidy, yet mines still there).

        I told her this isn’t going to work and she tells me I’m a controlling freak basically and she agrees and I’ve not heard from her since. Moral of the story is she’s too damaged from her upbringing I’m guessing and some people you just can’t change. She still messages me asking how I am but I’m sjust ignoring her except from anything baby related. I need to move on.

        I know a lot of people questioned whether she’s pregnant, how stupid I was to get her pregnant (I agree) and if it’s mine. I’ve been to every scan so I know she’s pregnant, as for if it’s mine I’ve never suspected cheating but she’s a serial liar so I will be forcing a DNA test through the courts. I posted on a couple of different subs to make sure I wasn’t getting biased opinions. The above story is 100% true (I wish it wasn’t believe me) but my focus is now getting as far away as possible from her for my own sake.

        
        Final update:

        After a couple of weeks or learning she was sleeping around on family members or friends sofas I allowed her back into the house given that she is pregnant. Around a month ago. Out of concern for the baby really given she’s now 8 months pregnant. Stupid on my part and I’m now going to explain why I regret it.

        I’ve recently moved house (a couple of months ago) and she was involved in the packaging and unpacking whilst I was out. Mainly unpacking. I had a pretty large stack of cash in the drawer of a cabinet in living room. Around £400-500. This was a Christmas gift from my parents. During this time I also sold a lot of old furniture including a sofa which she begged and begged for me to sell it to her mother. I begrudgingly accepted this. She told me her mother had asked to borrow the money from her repeatedly to buy it from me and asked if she could pay a couple of weeks after she took it. No biggie, that’s fine I tell her.

        Her mother collects the sofa, giving me £100 cash initially and tells me the rest will be with me in a week. A week comes round and she tells me it’ll be next month but she’s not happy as it’s collapsed and I need to come take a look. I tell her it was fine was she collected it and I’m not taking a look. Basically if you don’t want it I’ll collect it and sell it to someone who wants to pay. She tells me I’m not welcome in their house. My girlfriend (ex) told her there was nothing wrong with the sofa at all when it was collected and her mother tells her she’s also not welcome. My ex then flips it onto me telling me I’m controlling and she didn’t need to get involved to fall out with her family. I didn’t make her but I told her it showed where her priorities lie when she’s defending them and not wanting to get involved over them screwing me over. It was left at that.

        Back to the money, I went to see where the money went and searched the entire house. It’s not there but everything that was unpacked was there, even pointless shit like a blown light bulb was packed and unpacked. I ask her where the money is and she immediately gets defensive. Tells me “it’s somewhere” and immediately I think “this is all the same answers as last time.”

        It then dawned on me that the money I was gifted, was in £10 notes and the money I was part paid for my sofa was also in £10 notes so my suspicion is she’s stole my money for her to hand to her mother to pay me. I’ve basically paid myself minus what been taken. I confronted her and she replied “even if I did admit it to try and sort things I don’t care about you anymore anyway so I don’t need to.” Probably makes sense why she was trying to take a loan out roughly the same time she would’ve taken the money. So there we have it, I let her stop for a while and this is where it’s landed me. Her stealing again. Whilst I have no solid proof whatsoever it could only be her that took it and if everything else got unpacked then she’s certainly took it. Shes now threatening to out me to people for who I really am (a victim of theft I guess?) and she’ll tell everyone how awful I am and not to bother contacting her. I’ve thrown her back out again for the very last time and I’m just relieved. Not sad at all. Whilst I have no proof her reaction is all the proof I need. Now I’m forcing a DNA test at birth and will fight to make sure no child of mine is brought up in a family like hers. She is poison. People like her don’t change. They just take more.
    """,
    outDir="nuevo",
    female=False,
)

quit()
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
