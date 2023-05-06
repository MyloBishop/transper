# transper
A simple implementation of real-time output device audio transcription and translation using "faster_whisper" and "pyaudiowpatch".

This is a Python script that uses the Whisper model and PyAudio library to perform real-time audio transcription.  
The script records audio from the default output device (e.g., speakers) using PyAudio and saves it to a temporary WAV file.  
Then, it transcribes the audio using the Whisper model.  
The transcription is displayed in real-time, with each segment of audio and its corresponding text displayed as soon as it is transcribed.

## Installation
To run the script, you will need to install the following packages:  
```
torch
pyaudiowpatch
faster-whisper
```
You can install them using pip:  
```
pip install -r requirements.txt
```  
Note: It is highly reccomended to use GPU for whisper transcription. If you have a CUDA enabled GPU, please download the correct version from [Here](https://pytorch.org/get-started/previous-versions/)  

## Usage
To run the script, simply execute the main.py file:  
```
python main.py
``` 
The script will start recording audio from the default output device and display the transcriptions in real-time. To stop the script, press Ctrl-C.  

## Contributing
Contributions are welcome! This is a admittedly naive implementation made to fit my usecase.  
If you find any issues with the script or want to add new features, feel free to submit a pull request.  

## License
This project is licensed under the MIT License - see the LICENSE file for details.
