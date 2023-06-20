import gradio as gr
from main import index, run
from gtts import gTTS
import os, time

from transformers import pipeline

p = pipeline("automatic-speech-recognition")

"""Use text to call chat method from main.py"""

def add_text(history, text):
    print("Question asked: " + text)
    response = run_model(text)
    history = history + [(text, response)]
    print(history)
    return history, ""


def run_model(text):
    start_time = time.time()
    print("start time:" + str(start_time))
    response = run(text)
    end_time = time.time()
    # If response contains string `SOURCES:`, then add a \n before `SOURCES`
    if "SOURCES:" in response:
        response = response.replace("SOURCES:", "\nSOURCES:")
        # response = response + "\n\n" + "Time taken: " + str(end_time - start_time)
    print(response)
    print("Time taken: " + str(end_time - start_time))
    return response



def get_output(history, audio):

    txt = p(audio)["text"]
    # history.append(( (audio, ) , txt))
    audio_path = 'response.wav'
    response = run_model(txt)
    # Remove all text from SOURCES: to the end of the string
    trimmed_response = response.split("SOURCES:")[0]
    myobj = gTTS(text=trimmed_response, lang='en', slow=False)
    myobj.save(audio_path)
    # split audio by / and keep the last element
    # audio = audio.split("/")[-1]
    # audio = audio + ".wav"
    history.append(( (audio, ) , (audio_path, )))
    print(history)
    return history

def set_model(history):
    history = get_first_message(history)
    index()
    return history


def get_first_message(history):
    history = [(None,
                'Get your canvas disucssion graded. Add your discussion url and get your discussions graded in instantly.')]
    return history


def bot(history):
    return history

with gr.Blocks() as demo:

    chatbot = gr.Chatbot(get_first_message([]), elem_id="chatbot").style(height=600)

    with gr.Row():
        with gr.Column(scale=0.75):
            txt = gr.Textbox(
                label="8 Nous Grading Bot",
                placeholder="Enter text and press enter, or upload an image", lines=1
            ).style(container=False)

        with gr.Column(scale=0.25):
            audio = gr.Audio(source="microphone", type="filepath").style(container=False)

    txt.submit(add_text, [chatbot, txt], [chatbot, txt], postprocess=False).then(
        bot, chatbot, chatbot
    )

    audio.change(fn=get_output, inputs=[chatbot, audio], outputs=[chatbot]).then(
        bot, chatbot, chatbot
    )

    audio.change(lambda:None, None, audio)

    set_model(chatbot)

if __name__ == "__main__":
    demo.queue()
    demo.queue(concurrency_count=5)
    demo.launch(debug=True)
