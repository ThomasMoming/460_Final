# 460_Final
 
# Echo Mirror

## Project Introduction

This project is an artificial intelligence-based virtual piano creation system that combines Markov chains and LSTM neural networks. Users can:

- Play the virtual piano using the keyboard
- Record notes and rhythms in real time
- Generate new melodies using Markov or LSTM models
- Play recorded or generated melodies

This system demonstrates the application of computational creativity in the field of music generation and provides an intuitive human-computer interaction experience.

## Project Features

- Virtual Piano Graphical Interface (Tkinter)
- Note recording and time tracking
- Markov Melody Generator
- LSTM Music Generation Model (implemented in PyTorch)
- Real-time playback function (supports playback of recorded melodies and AI-generated melodies)

### Dependency Installation

Please install the necessary Python libraries first:

```bash
pip install torch numpy markovify pygame
```

Make sure `pygame.midi` is natively supported (some systems require additional setup of MIDI support).

The program will open a window for the user to play and generate melodies.

## Quick Operation Instructions

| Function           | Operate                       |
|----------------|--------------------------------|
| Playing Notes       | Keyboard Input D~L（White Key），R~O（black Key） |
| Start recording     |  Start                  |
| Stop recording      |  Stop                   |
| Markov generate     |  Markov                 |
| LSTM generate       |  LSTM                   |
| Play Recording      |  Play                   |
| Play Markov         |  Play Markov            |
| Play LSTM           |  Play LSTM              |

## Project Structure

```
.
├── main.py                 # Main program entry
├── ui.py                   # Main interface and logic control
├── config.py               # Configuration of buttons, notes, colors, etc.
├── lstm.py                 # LSTM Model and generation logic
├── markov.py               # Markov Melody Generator
├── event_handler.py        # MIDI Playback and key processing
├── piano_display.py        # Display current note
├── piano_control.py        # Keyboard event handling
├── dataset.txt             # Musical note training dataset
├── data/
│   ├── notes.pkl           # Note Index Mapping
│   └── durations.pkl       # Duration Index Mapping
└── lstm_model.pth          # Trained LSTM model file
```

## Demo
Presentation Link: (https://youtu.be/bugTsqdgu1k)


## Acknowledgements

This project is supported by the IAT 460 course framework. Thanks to the teaching team for their wonderful teaching and resource support in the field of generative AI and computational creativity.
