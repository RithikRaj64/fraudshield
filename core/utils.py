import openai
from pytube import YouTube
import random
import requests
from PIL import Image
from io import BytesIO
import easyocr
import os

from langchain_community.llms import OpenAI
from langchain.agents import AgentExecutor, AgentType, initialize_agent, load_tools  # type: ignore
from langchain_community.tools import BaseTool
from typing import List

messages = [
    {"role": "system", "content": "Investor seeking guidance. Provide a credibility score (1-100) for the given content, considering source reliability (40%), market conditions (30%), and risk factors (30%). Your response format: credibility score: (your answer) in one line, followed by reason:  in a concise paragraph (max 150 words). Emphasize due diligence importance, exercise caution, and maintain a highly critical approach. Address fraudulent activities and refrain from accepting information without proper evidence. The user relies on your assessment for investment decisions, so precision is crucial. The content is as follows:{topic} "},
]

openKey = os.getenv("OPENAI_API_KEY")
openai.api_key = "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"

def processText(query : str):
    # Append user's message to messages
    messages.append({"role": "user", "content": query})

    # Call OpenAI's Chat API
    chat = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k-0613",
        messages=messages,
        api_key = openKey
    )

    # Get the assistant's reply
    res = chat.choices[0].message["content"]
    
    print(res)

    # Append assistant's reply to messages
    messages.append({"role": "assistant", "content": res})
    # res1=res[:22]
    # res2=res[22:]

    return res


def processVideo(yt_link : str):
    video_caller = YouTube(yt_link)

    a = random.randint(1, 10000)
    a = str(a)

    titlename=video_caller.title
    video_caller.title = a
    
    video_caller.streams.filter(only_audio=True).first().download()
    b = a + ".mp4"

    with open(b, "rb") as audio_file:
        transcript2 = openai.Audio.transcribe(
            file=audio_file, model="whisper-1", response_format="srt", language="en"
        )
    # st.write(transcript2)
    if transcript2:
        # Append user's message to messages
        query = "Video title name : " + titlename + "\n" + "transcription: " + transcript2
        processText(query)


def processImage(img_link : str):
    res: requests.Response = requests.get(img_link, allow_redirects=True)
    Image.open(BytesIO(res.content)).save("curr.jpg")
    path = "./curr.jpg"

    Reader = easyocr.Reader(["en"])
    text = " ".join(Reader.readtext(path, detail=0))
    res=fraud(text).strip().capitalize()

    return res

def fraud(search: str):
    # Create a new instance of the OpenAI class
    llm = OpenAI(
        openai_api_key="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",
        max_tokens=200,
        temperature=0,
        client=None,
        model="text-davinci-003",
        frequency_penalty=1,
        presence_penalty=0,
        top_p=1,
    )

    # Load the tools
    tools: List[BaseTool] = load_tools(["google-serper"], llm=llm)

    # Create a new instance of the AgentExecutor class
    agent: AgentExecutor = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )

    template = """Investor seeking guidance. Provide a credibility score (1-100) for the given content, considering source reliability (40%), market conditions (30%), and risk factors (30%). Your response format: 'credibility score: (your answer)' in one line, followed by 'reason: ' in a concise paragraph (max 150 words). Emphasize due diligence importance, exercise caution, and maintain a highly critical approach. Address fraudulent activities and refrain from accepting information without proper evidence. The user relies on your assessment for investment decisions, so precision is crucial.The content is as follows: {topic}"""

    # Generate the response
    response: str = agent.run(template.format(topic=search))

    # Print the response
    print(response)

    return response

    # # Convert the response to a dictionary
    # result = json.loads(response)