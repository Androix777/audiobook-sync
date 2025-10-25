import json
import os
import re
import time
import math
import numpy as np
from rapidfuzz import fuzz
from sudachipy import tokenizer, dictionary
tokenizer_obj = dictionary.Dictionary().create(mode=tokenizer.Tokenizer.SplitMode.A)

path_audio = "INSERT_AUDIO_FILE_NAME_HERE"
fn, fe = os.path.splitext(path_audio)
path_txt = "INSERT_TXT_FILE_NAME_HERE"

min_silence_duration_ms = 300
min_silence_duration = round(min_silence_duration_ms / 1000, 1)
pad = min(0.15, min_silence_duration / 2)
whisper_device = "cuda"
whisper_compute_type = "float16"
whisper_model = "medium" #"efwkjn/whisper-ja-anime-v0.1" for fewer hallucinations but worse timestamps
whisper_batch_size = 8
speech_timestamps_chunk_size = 1 # anything other than "1" will be much faster (up to 10x faster), but will risk imprecise timestamp borders 
noise_detection_threshold = 50
reading_window_size = 2500 # 2500 should be enough under most circumstances, 15000 needed for 告白 which includes a 25 minute audio book exclusive interview messing up the runtime estimation
furigana_mode = "ignore"

fn_segments = f"{fn}_segments_{min_silence_duration_ms}ms.json"
fn_whisper_results = f"{fn}_whisper_results_{min_silence_duration_ms}ms.json"
fn_whisper_resegmented = f"{fn}_whisper_resegmented_{min_silence_duration_ms}ms.json"
fn_whisper_srt = f"{fn}_whisper_{min_silence_duration_ms}ms.srt"


attach_left = "─／…、。！!）〉？?」』】\)・"
attach_right = "「『【｛（〈\("
ignore = "　\\s"
left_attaching_particles = "がはをへでにとの"

if os.path.exists(fn_segments):
    with open(fn_segments, "r", encoding="utf-8") as f:
        speech_timestamps = json.load(f)
    print("Found segments file. Skipping step 1...")
else:
    clock = time.time()
    print("Step 1: Noise detection")
    import librosa
    speech_timestamps = []
    for i in range(2000):
        y, sr = librosa.load(path_audio, sr=None, offset=max(0, i*1800-60), duration=1920)
        hop_size = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_size)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=0)
        noise_frames = np.where(rms_db > noise_detection_threshold)[0]
        times = librosa.frames_to_time(noise_frames, sr=sr, hop_length=hop_size)
        start_time = None
        for j, t in enumerate(times):
            if start_time is None:
                start_time = t
            if j == len(times) - 1 or times[j+1] - times[j] > min_silence_duration:
                if i == 0:
                    if t > 1800:
                        break
                    speech_timestamps.append({"start": max(0.0, round(float(start_time) - pad, 2)), "end": round(float(t) + pad, 2)})
                elif t > 60 and t <= 1860:
                    start = max(0.0, round(float(i*1800-60+start_time) - pad, 2))
                    end = round(float(i*1800-60+t) + pad, 2)
                    if not speech_timestamps or start - speech_timestamps[-1]["start"] > 0.1:
                        speech_timestamps.append({"start": start, "end": end})
                if t > 1860:
                    break
                start_time = None
        if librosa.get_duration(y=y, sr=sr) < 1860:
            break
    with open(fn_segments, "w", encoding="utf-8") as f:
        json.dump(speech_timestamps, f, separators=(",", ":"))

    print(f"Finished step 1 in {round(time.time() - clock, 1)}s.")

def whisper_transcribe(audio, timestamps, model):
    segments, _ = model.transcribe(
        audio=audio,
        language="ja",
        multilingual=False,
        word_timestamps=True,
        vad_filter=False,
        batch_size=whisper_batch_size,
        clip_timestamps=timestamps
    )
    return segments

def chunks(list, n):
    if n < 1:
        yield list
    else:
        for i in range(0, len(list), n):
            yield list[i:i + n]

batched_model = None
if os.path.exists(fn_whisper_results):
    with open(fn_whisper_results, "r", encoding="utf-8") as f:
        whisper_result = json.load(f)
    print("Found Whisper results. Skipping step 2.1...")
    words = [word for segment in whisper_result for word in segment["words"]]
else:
    clock = time.time()
    print("Step 2.1: Whisper")
    from faster_whisper import WhisperModel, BatchedInferencePipeline, audio

    model = WhisperModel(whisper_model, device=whisper_device, compute_type=whisper_compute_type)
    batched_model = BatchedInferencePipeline(model=model)
    audio = audio.decode_audio(path_audio)

    whisper_result = []
    words = []
    speech_timestamps_chunks = list(chunks(speech_timestamps, speech_timestamps_chunk_size))
    speech_timestamps_chunks_len = len(speech_timestamps_chunks)
    clock2 = time.time()
    for i, chunk in enumerate(speech_timestamps_chunks):
        segments = whisper_transcribe(audio, chunk, batched_model)
        for segment in segments:
            segment_dict = segment.__dict__
            segment_dict["words"] = [word.__dict__ for word in segment_dict["words"]]
            whisper_result.append(segment_dict)
            words.extend(segment_dict["words"])
        print(f"Transcribed chunk {i+1}/{speech_timestamps_chunks_len} ({round(((i+1)/speech_timestamps_chunks_len)*100, 1)}%). Time elapsed: {round(time.time() - clock2, 1)}s.", end="\r")
    print(f"Transcribed chunk {i+1}/{speech_timestamps_chunks_len} ({round(((i+1)/speech_timestamps_chunks_len)*100, 1)}%). Time elapsed: {round(time.time() - clock2, 1)}s.")


    with open(fn_whisper_results, "w", encoding="utf-8") as f:
        json.dump(whisper_result, f, separators=(",", ":"), ensure_ascii=False)

    print(f"Finished step 2.1 in {round(time.time() - clock, 1)}s.")

clock = time.time()
print("Step 2.2: Resegment")

if os.path.exists(fn_whisper_resegmented):
    with open(fn_whisper_resegmented, "r", encoding="utf-8") as f:
        whisper_resegmented = json.load(f)
    print("Found resegmented Whisper file. Skipping resegmentation...")
else:
    whisper_resegmented = []
    word_len = len(words)
    current_words = []
    text = ""

    j = 0
    for i in range(word_len):
        if not current_words:
            current_words.append(words[i])
            text += str(words[i]["word"]).strip()
            while speech_timestamps[j]["end"] < words[i]["start"]:
                if not batched_model:
                    from faster_whisper import WhisperModel, BatchedInferencePipeline, audio
                    model = WhisperModel(whisper_model, device=whisper_device, compute_type=whisper_compute_type)
                    batched_model = BatchedInferencePipeline(model=model)
                    audio = audio.decode_audio(path_audio)
                segments = whisper_transcribe(audio, [speech_timestamps[j]], batched_model)
                text = "".join(segment.text for segment in segments).strip()
                whisper_resegmented.append({
                    "text": text if text else "�" * int(np.ceil((speech_timestamps[j]["end"] - speech_timestamps[j]["start"]) / 0.2)),
                    "start": round(speech_timestamps[j]["start"], 2),
                    "end": round(speech_timestamps[j]["end"], 2)
                })
                j += 1
            start = speech_timestamps[j]["start"]
        if i+1 < word_len:
            if words[i+1]["start"] < speech_timestamps[j]["end"]:
                text += str(words[i+1]["word"]).strip()
                current_words.append(words[i+1])
            else:
                whisper_resegmented.append({
                    "text": text,
                    "start": speech_timestamps[j]["start"],
                    "end": speech_timestamps[j]["end"]
                })
                current_words = []
                text = ""
                j += 1
    if current_words:
        whisper_resegmented.append({
            "text": text,
            "start": speech_timestamps[j]["start"],
            "end": speech_timestamps[j]["end"]
        })
    
    if speech_timestamps_chunk_size != 1:
        for i in range(0, len(whisper_resegmented) - 1):
            tokens = tokenizer_obj.tokenize(whisper_resegmented[i]["text"] + whisper_resegmented[i+1]["text"])
            line_len = len(whisper_resegmented[i]["text"])
            for token in tokens:
                if token.end() < line_len:
                    continue
                if token.end() == line_len:
                    break
                if token.begin() < line_len:
                    whisper_resegmented[i+1]["text"] = whisper_resegmented[i]["text"][-1] + whisper_resegmented[i+1]["text"]
                    whisper_resegmented[i]["text"] = whisper_resegmented[i]["text"][:-1]
                elif token.begin() > line_len:
                    whisper_resegmented[i]["text"] += whisper_resegmented[i+1]["text"][0]
                    whisper_resegmented[i+1]["text"] = whisper_resegmented[i+1]["text"][1:]
                break

    with open(fn_whisper_resegmented, "w", encoding="utf-8") as f:
        json.dump(whisper_resegmented, f, separators=(",", ":"), ensure_ascii=False)

def seconds_to_srt_format(seconds):
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    min = (int(seconds) % 3600) // 60
    h = int(seconds) // 3600
    return f'{h:02}:{min:02}:{s:02},{ms:03}'

def get_srt(lines):
    srt = ""
    for i, line in enumerate(lines, 1):
        start_time = seconds_to_srt_format(line["start"])
        end_time = seconds_to_srt_format(line["end"])
        text = line["text"]
        srt += f"{i}\n"
        srt += f"{start_time} --> {end_time}\n"
        srt += f"{text}\n\n"
    return srt

with open(fn_whisper_srt, "w", encoding="utf-8") as f:
    f.write(get_srt(whisper_resegmented))

print(f"Finished step 2.2 in {round(time.time() - clock, 1)}s.")

clock = time.time()
print("Step 3: Match subtitles")

with open(path_txt, "r", encoding="utf-8") as f:
    txt = f.read()

with open(fn_whisper_resegmented, "r", encoding="utf-8") as f:
    subs = json.load(f)

txt = "　".join(txt.split("\n"))
txt = re.sub(r"　+", "　", txt)


txt_len = len(txt)
while True:
    txt = re.sub(r"\[([^\]]*)　([^\]]*)\]", r"[\g<1>\g<2>]", txt)
    new_txt_len = len(txt)
    if txt_len == new_txt_len:
        break
    else:
        txt_len = new_txt_len

def remove_brackets(text):
    return re.sub(r"\[([^\]]*)\]", r"\g<1>", text)

def merge_furigana(text):
    furigana_pattern = r"\[([^\]]*)\](.)\[([^\]]*)\]"
    furigana_replace = r"\g<2>[\g<1>\g<3>]"
    text_len = len(text)
    while True:
        text = re.sub(furigana_pattern, furigana_replace, text)
        new_text_len = len(text)
        if text_len == new_text_len:
            break
        else:
            text_len = new_text_len
    return text

def replace_with_reading(text):
    return re.sub(r".\[([^\]]*)\]", r"\g<1>", text)

match furigana_mode:
    case "keep":
        pass
    case "remove_brackets":
        txt = remove_brackets(txt)
    case "merge":
        txt = merge_furigana(txt)
    case "merge_remove_brackets":
        txt = merge_furigana(txt)
        txt = remove_brackets(txt)
    case "replace":
        txt = replace_with_reading(txt)
    case "ignore":
        txt = re.sub(r"\[[^\]]*\]", "", txt)

txt_len = len(txt)
t_min = subs[0]["start"]
t_max = subs[-1]["end"]
duration_total = round(t_max - t_min, 2)
duration_speech_total = 0.0

def get_reading(text):
    reading_aggr = ""
    reading_to_text_low = []
    reading_to_text_high = []
    txt_to_reading_low = []
    txt_to_reading_high = []
    len_text = len(text)
    batch_size = 10000
    pad_size = 500
    number_of_batches = math.ceil(len_text / batch_size)
    for batch in range(number_of_batches):
        i_min = batch * batch_size
        i_max = min(len_text - 1, (batch+1) * batch_size - 1)
        i_min_pad = max(0, i_min - pad_size)
        i_max_pad = min(len_text, i_max + pad_size)
        left_pad = min(i_min_pad, pad_size)

        for token in tokenizer_obj.tokenize(text[i_min_pad:i_max_pad]):
            # process overlaps in the left batch
            if token.begin() < left_pad or token.begin() >= left_pad + batch_size:
                continue
            token_len = len(token)
            reading = token.reading_form()
            if token_len == 1 and reading == "キゴウ":
                reading = token.surface()
            reading_len = len(reading)
            i_txt_low = i_min_pad + token.begin()
            i_txt_high = i_min_pad + token.end()
            i_reading_low = len(reading_aggr)
            i_reading_high = i_reading_low + reading_len
            reading_aggr += reading
            txt_to_reading_low.extend([i_reading_low] * token_len)
            txt_to_reading_high.extend([i_reading_high] * token_len)
            reading_to_text_low.extend([i_txt_low] * reading_len)
            reading_to_text_high.extend([i_txt_high] * reading_len)
    return (reading_aggr, reading_to_text_low, reading_to_text_high, txt_to_reading_low, txt_to_reading_high)

txt_reading, reading_to_txt_low, reading_to_txt_high, txt_to_reading_low, txt_to_reading_high = get_reading(txt)
txt_reading_len = len(txt_reading)

for i, sub in enumerate(subs):
    duration_speech_total = round(duration_speech_total + sub["end"] - sub["start"], 2)
    sub["running_total"] = duration_speech_total
    sub["reading"], _, _, _, _ = get_reading(sub["text"])
    sub["reading"] = re.sub(f"^[{attach_left + attach_right + ignore}]+", "", sub["reading"])
    sub["reading"] = re.sub(f"[{attach_left + attach_right + ignore}]+$", "", sub["reading"])
    sub["i"] = i

for sub in subs:
    sub["approx"] = int((sub["running_total"] / duration_speech_total) * txt_reading_len)

def maximize_score(subs, reading_min, reading_max):
    n = 2 * len(subs) + 1
    P = reading_max - reading_min
    if n == 1:
        return None, []

    num_vars = n - 1

    dp = [[-math.inf] * (P + 1) for _ in range(num_vars)]

    path = [[-1] * (P + 1) for _ in range(num_vars)]

    dp[0] = [0] * (P + 1)
    
    for i in range(1, num_vars):
        for j in range(P + 1):
            max_val = -math.inf
            best_k = -1
            for k in range(j + 1):
                current_val = dp[i-1][k] + f(subs, i, reading_min, k, j)
                if current_val > max_val:
                    max_val = current_val
                    best_k = k
            
            dp[i][j] = max_val
            path[i][j] = best_k

    max_total_sum = -math.inf
    optimal_last_i2 = -1
    
    for k in range(P + 1):
        current_sum = dp[num_vars - 1][k] + f(subs, n-1, reading_min, k, P)
        if current_sum > max_total_sum:
            max_total_sum = current_sum
            optimal_last_i2 = k

    if num_vars == 0:
        return max_total_sum, []
        
    i2_values = [-1] * num_vars
    i2_values[num_vars-1] = optimal_last_i2
    
    for i in range(num_vars - 1, 0, -1):
        i2_values[i-1] = path[i][i2_values[i]]
        if i % 2 != 0:
            sub = subs[i//2]
            sub["i1"] = reading_min + i2_values[i-1]
            sub["i2"] = reading_min + i2_values[i]
            sub["txt_reading"] = txt_reading[sub["i1"]:sub["i2"]]
            sub["score_partial"] = fuzz.partial_ratio(sub["reading"], sub["txt_reading"])
            sub["score"] = fuzz.ratio(sub["reading"], sub["txt_reading"])
            sub["txt_i1"] = reading_to_txt_low[sub["i1"]]
            sub["txt_i2"] = reading_to_txt_high[sub["i2"]-1] if sub["i2"] > sub["i1"] else sub["txt_i1"]
            sub["txt"] = txt[sub["txt_i1"]:sub["txt_i2"]]

def f(subs, i, reading_min, i1, i2):
    if i % 2 == 0 or i1 == i2:
        return 0
    sub = subs[i//2]
    sub_reading_len = len(sub["reading"])
    if i2-i1 < 0.3 * sub_reading_len or i2-i1 > 3 * sub_reading_len:
        return 0
    score_cutoff = min(90, max(40, 100 - 6 * min(i2-i1, sub_reading_len)))
    return (sub_reading_len / 8) * fuzz.ratio(sub["reading"], txt_reading[reading_min+i1:reading_min+i2], score_cutoff=score_cutoff)

subs_len = len(subs)
def recursive_align(subs_min, subs_max, reading_min, reading_max):
    current_streak = 1
    max_streak = 1
    max_streak_upper = subs_min + 1

    for i in range(subs_min, subs_max):
        sub = subs[i]
        if re.match(r"^�+$", sub["text"]):
            sub["i1"] = max(reading_min, subs[i-1]["i2"]) if i > 1 else 0
            sub["i2"] = min(reading_max, sub["i1"] + len(sub["reading"]))
            sub["score"] = 80
        else:
            i_low = max(reading_min, sub["approx"] - reading_window_size)
            i_high = min(reading_max, sub["approx"] + reading_window_size)

            score_partial, _, _, txt_reading_start, txt_reading_end = fuzz.partial_ratio_alignment(sub["reading"], txt_reading[i_low:i_high])
            i1 = i_low + txt_reading_start
            i2 = i_low + txt_reading_end

            if score_partial > 80 and current_streak > 1 and i1 < subs[i-1]["i2"] and i_high - subs[i-1]["i2"] > 0.8 * len(sub["reading"]):
                score_partial2, _, _, txt_reading_start2, txt_reading_end2 = fuzz.partial_ratio_alignment(sub["reading"], txt_reading[subs[i-1]["i2"]:i_high])
                if score_partial2 > 80:
                    score_partial = score_partial2
                    i1 = subs[i-1]["i2"] + txt_reading_start2
                    i2 = subs[i-1]["i2"] + txt_reading_end2
                    pass

            score = fuzz.ratio(sub["reading"], txt_reading[i1:i2])
            while i1 > i_low:
                score1 = fuzz.ratio(sub["reading"], txt_reading[i1-1:i2])
                if score1 > score:
                    i1 -= 1
                    score = score1
                else:
                    break
            while i2 < i_high:
                score1 = fuzz.ratio(sub["reading"], txt_reading[i1:i2+1])
                if score1 > score:
                    i2 += 1
                    score = score1
                else:
                    break
            while i2-i1 > 0:
                score1 = fuzz.ratio(sub["reading"], txt_reading[i1+1:i2])
                if score1 >= score:
                    i1 += 1
                    score = score1
                else:
                    break
            while i2-i1 > 0:
                score1 = fuzz.ratio(sub["reading"], txt_reading[i1:i2-1])
                if score1 >= score:
                    i2 -= 1
                    score = score1
                else:
                    break

            sub["i1"] = i1
            sub["i2"] = i2
            sub["score_partial"] = score_partial
            sub["score"] = score
            sub["txt_reading"] = txt_reading[i1:i2]
            sub["txt_i1"] = reading_to_txt_low[i1]
            sub["txt_i2"] = reading_to_txt_high[i2-1] if i1 < i2 else sub["txt_i1"]
            sub["txt"] = txt[sub["txt_i1"]:sub["txt_i2"]]
        
        if i > subs_min and sub["i1"] < sub["i2"] and subs[i-1]["i1"] < sub["i1"] and subs[i-1]["i2"] < sub["i2"] and sub["i1"] - subs[i-1]["i2"] < min(8, 3 * len(sub["reading"])):
            current_streak += 1
            if current_streak > max_streak:
                max_streak = current_streak
                max_streak_upper = i+1
        else:
            current_streak = 1
    max_streak_lower = max_streak_upper - max_streak
    left_border = max_streak_lower == subs_min
    right_border = max_streak_upper == subs_max

    avg_score = np.average([sub["score"] for sub in subs[max_streak_lower:max_streak_upper]])

    if left_border and right_border:
        return
    elif left_border and max_streak > 4 and avg_score > 80:
        offset = 4
        recursive_align(max_streak_upper - offset, subs_max, subs[max_streak_upper-1-offset]["i2"], reading_max)
    elif right_border and max_streak > 4 and avg_score > 80:
        offset = 4
        recursive_align(subs_min, max_streak_lower + offset, reading_min, subs[max_streak_lower+offset]["i1"])
    elif max_streak > 8 and avg_score > 80:
        offset = 4
        recursive_align(subs_min, max_streak_lower+offset, reading_min, subs[max_streak_lower+offset]["i1"])
        recursive_align(max_streak_upper-offset, subs_max, subs[max_streak_upper-1-offset]["i2"], reading_max)
    elif subs[subs_max-1]["score"] > 80 and subs[subs_max-1]["i2"] == max([sub["i2"] for sub in subs[subs_min:subs_max] if sub["score"] > 40], default= -1):
        recursive_align(subs_min, subs_max - 1, reading_min, subs[subs_max-1]["i1"])
    elif subs[subs_min]["score"] > 80 and subs[subs_min]["i1"] == min([sub["i1"] for sub in subs[subs_min:subs_max] if sub["score"] > 40], default= -1):
        recursive_align(subs_min + 1, subs_max, subs[subs_min]["i2"], reading_max)
    elif subs_max - subs_min > 30:
        print(f"WARNING: You are running a linear programming algorithm on {subs_max - subs_min} consecutive lines. There might be a problem with the source text or the Whisper transcription...")
        reading_min_chunk = reading_min
        for subs_min_chunk in range(subs_min, subs_max, 10):
            subs_max_chunk = min(subs_max, subs_min_chunk + 10)
            reading_combined = "".join([sub["reading"] for sub in subs[subs_min_chunk:subs_max_chunk]])
            _, _, _, txt_reading_start, txt_reading_end = fuzz.partial_ratio_alignment(reading_combined, txt_reading[reading_min_chunk:reading_max])
            maximize_score(subs[subs_min_chunk:subs_max_chunk], reading_min_chunk + txt_reading_start, reading_min_chunk + txt_reading_end)
            reading_min_chunk = subs[subs_max_chunk-1]["i2"]
    else:
        maximize_score(subs[subs_min:subs_max], reading_min, reading_max)

recursive_align(0, subs_len, 0, txt_reading_len)

def negotiate_border(left_sub, right_sub):
    if left_sub["txt_i2"] == right_sub["txt_i1"]:
        return
    if left_sub["txt_i2"] < right_sub["txt_i1"]:
        gap = txt[left_sub["txt_i2"]:right_sub["txt_i1"]]
        if re.match(f"^[{ignore}]+$", gap):
            return
        match = re.match(f"^([{attach_left}]*)[{ignore}]*([{attach_right}]*)$", gap)
        if not match:
            match = re.match(f"^([^{attach_left + attach_right + ignore}]{{,5}}[{attach_left}]+)[{ignore}]*([^{attach_left + attach_right + ignore}]*)$", gap)
        if not match:
            match = re.match(f"^([^{attach_left + attach_right + ignore}]*)[{ignore}]*([{attach_right}]+[^{attach_left + attach_right + ignore}]{{,5}})$", gap)
        if not match:
            match = re.match(f"^([^{attach_left + attach_right + ignore}]{{,5}}[{attach_left}]+)[{ignore}]*([{attach_right}]+[^{attach_left + attach_right + ignore}]{{,5}})$", gap)
        if not match:
            match = re.match(f"^([^{attach_left + attach_right + ignore}]{{,5}})[{ignore}]+([^{attach_left + attach_right + ignore}]{{,5}})$", gap)
        if not match:
            match = re.match(f"^([{attach_left + attach_right}]+)[{ignore}]+([{attach_left + attach_right}]+)$", gap)
        if not match:
            _, _, _, _, reading_align_end_left = fuzz.partial_ratio_alignment(left_sub["txt_reading"], left_sub["reading"])
            _, _, _, reading_align_start_right, _ = fuzz.partial_ratio_alignment(right_sub["txt_reading"], right_sub["reading"])
            if reading_align_end_left < len(left_sub["reading"]) and reading_align_start_right == 0:
                match = re.match(f"^(.*?[^{ignore}])[{ignore}]*([{attach_right}]*[{ignore}]*)$", gap)
            elif reading_align_end_left == len(left_sub["reading"]) and reading_align_start_right > 0:
                match = re.match(f"^([{ignore}]*[{attach_left}]*)[{ignore}]*([^{ignore}].*)$", gap)
            elif reading_align_start_right > 0 and left_sub["txt"][-1] in attach_left + left_attaching_particles:
                match = re.match(f"^()[{ignore}]*([^{attach_left + ignore}]+)$", gap)
        if not match:
            match = re.match(f"^([{left_attaching_particles}])()$", gap)
        if not match:
            match = re.match(f"^([^{ignore}]{{,8}})[{ignore}]+([^{ignore}]{{,8}})$", gap)
            if not match:
                match = re.match(f"^([{attach_left}]+)[{ignore}]*(.{{,8}})$", gap)
            if match:
                right_sub["shift_left"] = True
        if not match:
            match = re.match(f"^([^{attach_left + attach_right + ignore}]{{,5}}?[{attach_left + left_attaching_particles}]+)[{ignore}]*([^{attach_left + attach_right + ignore}]*)$", gap)
        if not match and len(gap) < 5:
            if left_sub["score"] > 90 and right_sub["score"] < 80 or left_sub["score"] > 80 and reading_align_start_right > 0:
                match = re.match(f"^()[{ignore}]*(.*)$", gap)
            elif left_sub["score"] < 80 and right_sub["score"] > 90:
                match = re.match(f"^(.*?)[{ignore}]*()$", gap)
        if not match and len(gap) < 10 and right_sub["start"] - left_sub["end"] < 5 * min_silence_duration and right_sub["end"] - left_sub["start"] < 60:
            left_sub["txt_i2"] = left_sub["txt_i1"]
            right_sub["start"] = left_sub["start"]
            right_sub["txt_i1"] = left_sub["txt_i1"]
            right_sub["txt"] = txt[right_sub["txt_i1"]:right_sub["txt_i2"]]
            return
        if not match:
            match = re.match(f"^([{attach_left}]*).*?([{attach_right}]*)$", gap)
        if match:
            left_sub["txt_i2"] += match.end(1)
            left_sub["txt"] = txt[left_sub["txt_i1"]:left_sub["txt_i2"]]
            right_sub["txt_i1"] -= match.end(2) - match.start(2)
            if "shift_left" in right_sub:
                right_sub["txt_i2"] =  right_sub["txt_i1"]
            right_sub["txt"] = txt[right_sub["txt_i1"]:right_sub["txt_i2"]]
            return
    else:
        overlap = txt[right_sub["txt_i1"]:left_sub["txt_i2"]]
        match = re.match(f"^({overlap}[{ignore}]*[{attach_left}]+)[{ignore}]*(.+)$", txt[right_sub["txt_i1"]:right_sub["txt_i2"]])
        if not match:
            match = re.match(f"^({overlap})[{ignore}]*([{attach_right}]+.*)$", txt[right_sub["txt_i1"]:right_sub["txt_i2"]])
        if match:
            left_sub["txt_i2"] = right_sub["txt_i1"] + match.end(1)
            left_sub["txt"] = txt[left_sub["txt_i1"]:left_sub["txt_i2"]]
            right_sub["txt_i1"] += match.start(2)
            right_sub["txt"] = txt[right_sub["txt_i1"]:right_sub["txt_i2"]]
            return
        match = re.match(f"^(.+?)[{ignore}]*([{attach_right}]+[{ignore}]*{overlap})$", txt[left_sub["txt_i1"]:left_sub["txt_i2"]])
        if not match:
            match = re.match(f"^(.*[{attach_left}]+)[{ignore}]*({overlap})$", txt[left_sub["txt_i1"]:left_sub["txt_i2"]])
        if match:
            left_sub["txt_i2"] = left_sub["txt_i1"] + match.end(1)
            left_sub["txt"] = txt[left_sub["txt_i1"]:left_sub["txt_i2"]]
            right_sub["txt_i1"] = left_sub["txt_i1"] + match.start(2)
            right_sub["txt"] = txt[right_sub["txt_i1"]:right_sub["txt_i2"]]
            return
        if ("っ" in overlap or "ッ" in overlap) and right_sub["start"] - left_sub["end"] < 5 * min_silence_duration and right_sub["end"] - left_sub["start"] < 60:
            left_sub["txt_i2"] = left_sub["txt_i1"]
            right_sub["start"] = left_sub["start"]
            right_sub["txt_i1"] = left_sub["txt_i1"]
            right_sub["txt"] = txt[right_sub["txt_i1"]:right_sub["txt_i2"]]
            return
        overlap_reading, _, _, _, _ = get_reading(overlap)
        score_left = fuzz.ratio(overlap_reading, txt_reading[max(0, left_sub["i2"]-len(overlap_reading)):left_sub["i2"]])
        score_right = fuzz.ratio(overlap_reading, txt_reading[right_sub["i1"]:min(txt_reading_len, right_sub["i1"]+len(overlap_reading))])
        if score_left > score_right:
            match = re.match(f"^({overlap})[{ignore}]*(.*)$", txt[right_sub["txt_i1"]:right_sub["txt_i2"]])
            left_sub["txt_i2"] = right_sub["txt_i1"] + match.end(1)
            left_sub["txt"] = txt[left_sub["txt_i1"]:left_sub["txt_i2"]]
            right_sub["txt_i1"] += match.start(2)
            right_sub["txt"] = txt[right_sub["txt_i1"]:right_sub["txt_i2"]]
            return
        else:
            match = re.match(f"^(.*)[{ignore}]*({overlap})$", txt[left_sub["txt_i1"]:left_sub["txt_i2"]])
            left_sub["txt_i2"] = left_sub["txt_i1"] + match.end(1)
            left_sub["txt"] = txt[left_sub["txt_i1"]:left_sub["txt_i2"]]
            right_sub["txt_i1"] = left_sub["txt_i1"] + match.start(2)
            right_sub["txt"] = txt[right_sub["txt_i1"]:right_sub["txt_i2"]]
            return

j = 0
k = -1
for i, sub in enumerate(subs):
    if sub["txt_i1"] == sub["txt_i2"]:
        if i > 0 and subs[i-1]["txt_i1"] != subs[i-1]["txt_i2"]:
            j = i
            k = i-1
        if i > 0 and i < subs_len - 1 and subs[i+1]["txt_i1"] != subs[i+1]["txt_i2"]:
            while j <= i:
                subs_debug = subs[j-1:i+2]
                txt_debug = txt[subs[j-1]["txt_i2"]:subs[i+1]["txt_i1"]]
                left_tail = f"[{attach_left}]*[{ignore}]*"
                fragment = f"[^{attach_left + ignore}]+[{attach_left}{left_attaching_particles}]+"
                match = re.match(f"^{left_tail}({fragment})[{ignore}]*({fragment}[{ignore}]*){{{i-j}}}[^{attach_left + ignore}]*$", txt[subs[j-1]["txt_i2"]:subs[i+1]["txt_i1"]])
                if match:
                    subs[j]["txt_i1"] = subs[j-1]["txt_i2"] + match.start(1)
                    subs[j]["txt_i2"] = subs[j-1]["txt_i2"] + match.end(1)
                    subs[j]["txt"] = txt[subs[j]["txt_i1"]:subs[j]["txt_i2"]]
                    negotiate_border(subs[k], subs[j])
                    k = j
                else:
                    subs[j]["txt_i1"] = subs[j-1]["txt_i2"]
                    subs[j]["txt_i2"] = subs[j-1]["txt_i2"]
                    if j < i:
                        subs[j+1]["start"] = subs[j]["start"]
                j += 1
        else:
            continue
    if i < subs_len - 1:
        if subs[i+1]["txt_i1"] == subs[i+1]["txt_i2"]:
            continue
        elif sub["txt_i1"] == sub["txt_i2"]:
            left_border = subs[k]["txt_i2"] if k >= 0 else 0
            match = re.match(f"^([{attach_left}]+).*$", txt[left_border:subs[i+1]["txt_i1"]])
            if match and k >= 0:
                subs[k]["txt_i2"] += match.end(1)
                subs[k]["txt"] = txt[subs[k]["txt_i1"]:subs[k]["txt_i2"]]
                left_border = subs[k]["txt_i2"]
                pass
            match = re.match(f"^.*?([{attach_right}]+)$", txt[left_border:subs[i+1]["txt_i1"]])
            if match:
                subs[i+1]["txt_i1"] = left_border + match.start(1)
                subs[i+1]["txt"] = txt[subs[i+1]["txt_i1"]:subs[i+1]["txt_i2"]]
                pass
            match = re.match(f"^[{ignore}]*(.{{,{max(0, 3 * len(sub['text']) - 1)}}}[^{ignore}])[{ignore}]*$", txt[left_border:subs[i+1]["txt_i1"]])
            if match:
                sub["txt_i1"] = left_border + match.start(1)
                sub["txt_i2"] = left_border + match.end(1)
                sub["txt"] = txt[sub["txt_i1"]:sub["txt_i2"]]
                pass
            if sub["txt_i1"] == sub["txt_i2"]:
                tail_left = txt[min(left_border, left_border - 5):left_border]
                head_right = txt[subs[i+1]["txt_i1"]:min(subs[i+1]["txt_i1"] + 5, subs[i+1]["txt_i2"])]
                if k >= 0 and ("っ" in tail_left or "ッ" in tail_left) and sub["start"] - subs[k]["end"] < 5 * min_silence_duration and sub["end"] - subs[k]["start"] < 60:
                    subs[k]["end"] = sub["end"]
                elif ("っ" in head_right or "ッ" in head_right) and subs[i+1]["start"] - sub["end"] < 5 * min_silence_duration and subs[i+1]["end"] - sub["start"] < 60:
                    subs[i+1]["start"] = sub["start"]
                else:
                    pass
        else:
            negotiate_border(sub, subs[i+1])

    left_border = subs[i-1]["txt_i2"] if i > 0 else 0
    right_border = subs[i+1]["txt_i1"] if i < subs_len - 1 else txt_len
    if sub["txt_i1"] != sub["txt_i2"] and sub["txt_i1"] - left_border > 10 and right_border - sub["txt_i2"] > 10 and sub["score"] < 80:
        sub["txt_i2"] = sub["txt_i1"]
        sub["txt"] = txt[sub["txt_i1"]:sub["txt_i2"]]

filtered_subs = [sub for sub in subs if sub["txt_i1"] != sub["txt_i2"]]

#match = re.match(f"^.*?([{attach_right}]+)$", txt[:filtered_subs[0]["txt_i1"]])
#if match:
#    filtered_subs[0]["txt_i1"] = match.start(1)
#match = re.match(f"^([{attach_left}]+).*$", txt[filtered_subs[-1]["txt_i2"]:])
#if match:
#    filtered_subs[-1]["txt_i2"] += match.end(1)

print("".join("■" if sub["txt_i1"] != sub["txt_i2"] else "□" for sub in subs))

def get_srt2(lines):
    srt = ""
    for i, line in enumerate(lines, 1):
        start_time = seconds_to_srt_format(line["start"])
        end_time = seconds_to_srt_format(line["end"])
        text = txt[line["txt_i1"]:line["txt_i2"]]
        srt += f"{i}\n"
        srt += f"{start_time} --> {end_time}\n"
        srt += f"{text}\n\n"
    return srt

with open(f"{fn}.srt", "w", encoding="utf-8") as f:
    f.write(get_srt2(filtered_subs))

print(f"Finished step 3 in {round(time.time() - clock, 1)}s.")