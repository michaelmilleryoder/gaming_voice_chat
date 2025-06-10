# Gaming Voice Chat Post-Processing: Audio Post-Processing

This is a Python script for processing the **audio of stream videos** located in a specified input folder. The pipeline extracts and processes audio from each video and saves the results (processed audio, transcribed text) to an output folder.

For each file, this file:
- Diarizes the file using `pyannote.audio`, getting the audio segments of speech
- Transcribes the diarized file using Whisper, getting a "script" of the speech in the file -- saved to `.txt`
- Applies audio anonymization, de-identifying the speech in the audio -- processed audio is saved to `.wav` or `.mp4`

Any file that causes an error is skipped.

## Usage
Run the script from the command line with the following arguments:

```bash
python audio_pipeline.py <in_folder_path> <out_folder_path>
```

Where:
- <in_folder_path>: Path to the folder containing raw input stream video files.
- <out_folder_path>: Path to the folder where processed audio/text will be saved.

An example is:
```bash
python audio_pipeline.py "path/to/my/raw/videos/" "path/to/processed/videos/"
```
Which would process all videos in `path/to/my/raw/videos/`, saving the transcribed speech `.txt` and the anonymized speech `.wav`/`.mp4` for each video to `path/to/processed/videos/`.

## Additional Notes
Two versions of the script exist -- `audio_pipeline.py` and `audio_pipeline_windows.py`. If on a Windows machine, use `audio_pipeline_windows.py`. This version exists due to issues on Windows with accessing/deleting temporary files created within the script.
