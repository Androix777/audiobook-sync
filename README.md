Incomplete yet workable alternative to [SubsPlz](https://github.com/kanjieater/SubPlz) for audio books.

Accuracy is improved mainly through the following avenues:
- Noise detection instead of VAD
- Character level matching instead of line level matching
- Matching readings, not plain text

The matching algorithm is already fairly optimised and mostly produces results to the level the Whisper transcription allows. The post-processing heuristics for negotiating line borders and filling gaps often work, but are far from mature and will probably require a lot of trial and error to perfect. I will create issues for the problems I encounter and occasionally address some of them when I have the time. Feel free to do the same. Issues and PRs are welcome!

Workflow:
- Convert your e-book to TXT
-- I use Calibre with the following conversion rules (can be saved as a template):
  - Look & feel > Transform HTML > matches CSS selector: "ruby > rt" > Insert HTML at start: "\&#91;", Insert HTML at end: "\&#93;"
  - Look & feel > Transform HTML > is: "img" > Insert HTML before tag: "\&#x3000;"
- Convert the audio book into a compatible format. I usually choose MP3 with a constant bitrate of 192kbps which is also compatible with [https://reader.ttsu.app](https://reader.ttsu.app). You can do this with Audacity or ffmpeg.
- Move both TXT and audio file into the same folder as the script, change the file names in the script and run it.

Dependencies:
~~~
pip install numpy, rapidfuzz, sudachipy, sudachidict_core, librosa, faster_whisper
~~~
CPU (not recommended; change `whisper_device` to `"cpu"` and `whisper_compute_type` to `"default"`; use a `speech_timestamps_chunk_size` other than `1`):
~~~
pip install torch
~~~
GPU (recommended; get the appropriate [torch version](https://pytorch.org/get-started/locally/), e.g.):
~~~
pip install torch --index-url https://download.pytorch.org/whl/cu129
~~~