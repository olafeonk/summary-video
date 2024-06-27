# Use a pipeline as a high-level helper
from transformers import pipeline
from datasets import Dataset
from moviepy.editor import VideoFileClip


def convert_to_mp3(input_file, output_file):
    video = VideoFileClip(input_file)
    audio = video.audio
    audio.write_audiofile(output_file)

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def main():
    input_file = 'video-0.m4v'
    output_file = './пук среньк/output_new.mp3'

    convert_to_mp3(input_file, output_file)

    audio_dataset = Dataset.from_dict({"audio": [output_file]})
    pipe = pipeline("automatic-speech-recognition", model="aangry-mouse/whisper-base-ml-ru")

    subtitle = pipe(audio_dataset[0]["audio"])['text']
    print(subtitle)

    pipe_sum = pipeline("summarization", model="IlyaGusev/mbart_ru_sum_gazeta")

    for i in chunks(subtitle, 2048):
        print(i)
        summary = pipe_sum(f"summary to ru: {i}")

        print(summary)


if __name__ == "__main__":
    main()
