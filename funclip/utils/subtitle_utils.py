#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunClip). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)
import re

def time_convert(ms):
    ms = int(ms)
    tail = ms % 1000
    s = ms // 1000
    mi = s // 60
    s = s % 60
    h = mi // 60
    mi = mi % 60
    h = "00" if h == 0 else str(h)
    mi = "00" if mi == 0 else str(mi)
    s = "00" if s == 0 else str(s)
    tail = str(tail).zfill(3)
    if len(h) == 1: h = '0' + h
    if len(mi) == 1: mi = '0' + mi
    if len(s) == 1: s = '0' + s
    return "{}:{}:{},{}".format(h, mi, s, tail)

def str2list(text):
    pattern = re.compile(r'[\u4e00-\u9fff]|[\w-]+', re.UNICODE)
    elements = pattern.findall(text)
    return elements

class Text2SRT():
    def __init__(self, text, timestamp, offset=0):
        self.token_list = text
        self.timestamp = timestamp
        self.offset = offset
        start, end = timestamp[0][0] - offset, timestamp[-1][1] - offset
        self.start_sec, self.end_sec = start, end
        self.start_time = time_convert(start)
        self.end_time = time_convert(end)
    def text(self):
        if isinstance(self.token_list, str):
            return self.token_list.rstrip("、。，")
        else:
            res = ""
            for word in self.token_list:
                if '\u4e00' <= word <= '\u9fff':
                    res += word
                else:
                    res += " " + word
            return res.lstrip().rstrip("、。，")
    def srt(self, acc_ost=0.0):
        return "{} --> {}\n{}\n".format(
            time_convert(self.start_sec+acc_ost*1000),
            time_convert(self.end_sec+acc_ost*1000), 
            self.text())
    def time(self, acc_ost=0.0):
        return (self.start_sec/1000+acc_ost, self.end_sec/1000+acc_ost)
    def token_times(self, acc_ost=0.0):
        if isinstance(self.token_list, str) or len(self.token_list) != len(self.timestamp):
            return []
        return [
            {
                "text": str(token),
                "start": (ts[0] - self.offset) / 1000 + acc_ost,
                "end": (ts[1] - self.offset) / 1000 + acc_ost,
            }
            for token, ts in zip(self.token_list, self.timestamp)
        ]


def generate_srt(sentence_list):
    srt_total = ''
    for i, sent in enumerate(sentence_list):
        if not sent.get('timestamp'):
            continue
        t2s = Text2SRT(sent['text'], sent['timestamp'])
        if 'spk' in sent:
            srt_total += "{}  spk{}\n{}\n".format(i + 1, sent['spk'], t2s.srt())
        else:
            srt_total += "{}\n{}\n".format(i + 1, t2s.srt())
    return srt_total

def _subtitle_override_text(sentence, subtitle_overrides):
    if not subtitle_overrides:
        return None
    try:
        start_ms = int(sentence['timestamp'][0][0])
        end_ms = int(sentence['timestamp'][-1][1])
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    return subtitle_overrides.get(f"{start_ms}-{end_ms}")


def generate_srt_clip(sentence_list, start, end, begin_index=0, time_acc_ost=0.0, subtitle_overrides=None):
    start, end = int(start * 1000), int(end * 1000)
    srt_total = ''
    cc = 1 + begin_index
    subs = []
    for _, sent in enumerate(sentence_list):
        sentence_text = sent['text']
        sentence_tokens = str2list(sentence_text) if isinstance(sentence_text, str) else sentence_text
        override_text = _subtitle_override_text(sent, subtitle_overrides)
        if sent['timestamp'][-1][1] <= start:
            # print("CASE0")
            continue
        if sent['timestamp'][0][0] >= end:
            # print("CASE4")
            break
        # parts in between
        if (sent['timestamp'][-1][1] <= end and sent['timestamp'][0][0] > start) or (sent['timestamp'][-1][1] == end and sent['timestamp'][0][0] == start):
            # print("CASE1"); import pdb; pdb.set_trace()
            t2s = Text2SRT(override_text or sentence_text, sent['timestamp'], offset=start)
            srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
            subs.append((t2s.time(time_acc_ost), t2s.text(), t2s.token_times(time_acc_ost)))
            cc += 1
            continue
        if sent['timestamp'][0][0] <= start:
            # print("CASE2"); import pdb; pdb.set_trace()
            if not sent['timestamp'][-1][1] > end:
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > start:
                        break
                _text = override_text if j == 0 and override_text else sentence_tokens[j:]
                _ts = sent['timestamp'][j:]
            else:
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > start:
                        _start = j
                        break
                for j, ts in enumerate(sent['timestamp']):
                    if ts[1] > end:
                        _end = j
                        break
                # _text = " ".join(sent['text'][_start:_end])
                _text = sentence_tokens[_start:_end]
                _ts = sent['timestamp'][_start:_end]
            if len(ts):
                t2s = Text2SRT(_text, _ts, offset=start)
                srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
                subs.append((t2s.time(time_acc_ost), t2s.text(), t2s.token_times(time_acc_ost)))
                cc += 1
            continue
        if sent['timestamp'][-1][1] > end:
            # print("CASE3"); import pdb; pdb.set_trace()
            for j, ts in enumerate(sent['timestamp']):
                if ts[1] > end:
                    break
            _text = sentence_tokens[:j]
            _ts = sent['timestamp'][:j]
            if len(_ts):
                t2s = Text2SRT(_text, _ts, offset=start)
                srt_total += "{}\n{}".format(cc, t2s.srt(time_acc_ost))
                subs.append(
                    (t2s.time(time_acc_ost), t2s.text(), t2s.token_times(time_acc_ost))
                    )
                cc += 1
            continue
    return srt_total, subs, cc

