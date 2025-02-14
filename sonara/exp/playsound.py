import simpleaudio as sa

def play_audio(file_path):
    """
    Plays the specified audio file using simpleaudio.
    :param file_path: Path to the audio file (e.g., "output.wav").
    """
    try:
        print(f"Playing audio file: {file_path}")
        # Load the audio file
        wave_obj = sa.WaveObject.from_wave_file(file_path)
        # Play the audio
        play_obj = wave_obj.play()
        # Wait until the audio finishes playing
        play_obj.wait_done()
    except Exception as e:
        print(f"Error playing audio: {e}")

if __name__ == "__main__":
    # Play the audio file
    play_audio("/tmp/output.wav")