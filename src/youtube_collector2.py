"""
YouTube comment collector, second round.

Adds more Saudi data after the draft feedback, which said the Saudi result
relied on a small test set.

Changes from the first round:
1. Three new Saudi channels: @thmanyahPodcasts (AI, inflation, World Cup),
   @falsaif (AI) and @aleqtisadiah-news (inflation).
2. Topics are matched on the title first, using the same keywords as round
   1, so videos from round 1 keep their topic. If the title has no match,
   the description is checked with a shorter list of specific terms, and
   the match is only used if exactly one topic appears. This stops news
   roundups covering several stories from being assigned a topic.
3. The same rule is applied to the round 1 channels, so their video lists
   are rebuilt. Comments that were already collected are not fetched again.
4. The window stays 2024-06-01 to 2026-08-10, matching the Trends data.

On the first run yt_comments.csv and yt_collector_state.json are copied to
*_round1_backup files. New comments are appended to yt_comments.csv with the
same columns as before. The topic and match source for each video go in
yt_video_topics.csv.

Usage:
    $env:YOUTUBE_API_KEY="..."
    $env:PYTHONUTF8="1"
    python youtube_collector2.py              run or resume
    python youtube_collector2.py "--status"   progress, uses no quota

If it stops at the daily quota, run it again the next day and it resumes.
"""

import json
import os
import re
import shutil
import sys
import time
from datetime import datetime, timezone

import pandas as pd

COMMENTS_CSV = "yt_comments.csv"
STATE_JSON   = "yt_collector_state.json"
TOPICS_CSV   = "yt_video_topics.csv"
LOG_CSV      = "yt_collection_log_round2.csv"

API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

# same window as round 1
WINDOW_START = pd.Timestamp("2024-06-01", tz="UTC")
WINDOW_END   = pd.Timestamp("2026-08-10 23:59:59", tz="UTC")

RULE_VERSION = 2   # 1 = title only (round 1), 2 = title then description

CHANNELS = [
    # round 1 channels, relisted under rule 2
    ("SA", "@SaudiNewsTV",       "local"),
    ("US", "@NBCNews",           "local"),
    ("US", "@ABCNews",           "local"),
    # round 2 channels
    ("SA", "@thmanyahPodcasts",  "local"),
    ("SA", "@falsaif",           "local"),
    ("SA", "@aleqtisadiah-news", "local"),
]

MAX_VIDEOS_PER_CHANNEL = 60000
MAX_COMMENT_PAGES      = 20      # up to 2,000 comments per video, as round 1
DAILY_UNIT_BUDGET      = 9500

# usable weeks (20+ comments) per region-topic after round 1, only used
# to print a before and after comparison in --status
ROUND1_USABLE_WEEKS = {
    ("SA", "Ramadan"): 20, ("SA", "World Cup"): 20,
    ("SA", "artificial intelligence"): 8, ("SA", "inflation"): 9,
    ("SA", "weather"): 60,
    ("US", "ChatGPT"): 31, ("US", "Ramadan"): 1, ("US", "World Cup"): 31,
    ("US", "artificial intelligence"): 68, ("US", "bitcoin"): 32,
    ("US", "electric cars"): 39, ("US", "inflation"): 69, ("US", "weather"): 73,
}

# title keywords, same as round 1
TOPICS = {
    "ChatGPT": ["chatgpt", "chat gpt", "gpt", "gpt4", "gpt-4", "gpt5", "gpt-5",
                "openai", "open ai", "شات جي بي تي", "شات جيبيتي",
                "تشات جي بي تي", "جي بي تي", "شات بوت", "اوبن ايه اي",
                "روبوت المحادثة"],
    "artificial intelligence": ["artificial intelligence", "ai", "a.i.",
                "machine learning", "deep learning", "llm", "llms", "chatbot",
                "neural network", "ذكاء اصطناعي", "الذكاء الاصطناعي",
                "ذكاء إصطناعي", "تعلم الالة", "تعلم الآلة", "التعلم العميق",
                "نماذج لغوية", "الروبوتات", "خوارزميات"],
    "bitcoin": ["bitcoin", "btc", "crypto", "cryptocurrency", "cryptocurrencies",
                "blockchain", "ethereum", "eth", "altcoin", "mining", "binance",
                "بيتكوين", "بتكوين", "البيتكوين", "عملات رقمية", "عملة رقمية",
                "العملات الرقمية", "العملات المشفرة", "عملات مشفرة", "كريبتو",
                "بلوك تشين", "تعدين", "ايثيريوم", "منصات التداول"],
    "electric cars": ["electric car", "electric cars", "electric vehicle",
                "electric vehicles", "ev", "evs", "tesla", "hybrid car",
                "charging station", "battery car", "سيارات كهربائية",
                "سيارة كهربائية", "السيارات الكهربائية", "سيارات كهربائيه",
                "تسلا", "شحن كهربائي", "محطات شحن", "سيارة هجينة",
                "المركبات الكهربائية"],
    "Ramadan": ["ramadan", "ramadhan", "ramzan", "iftar", "suhoor", "sehri",
                "eid", "eid al-fitr", "taraweeh", "fasting", "رمضان",
                "شهر رمضان", "رمضان كريم", "افطار", "إفطار", "سحور",
                "عيد الفطر", "العيد", "تراويح", "التراويح", "صيام", "الصيام",
                "مسحراتي", "قيام الليل"],
    "inflation": ["inflation", "cost of living", "price rise", "rising prices",
                "cpi", "recession", "interest rate", "interest rates",
                "economy", "cost-of-living", "تضخم", "التضخم", "غلاء",
                "غلاء المعيشة", "الغلاء", "ارتفاع الاسعار", "ارتفاع الأسعار",
                "الاسعار", "الأسعار", "ركود", "الركود", "اسعار الفائدة",
                "أسعار الفائدة", "تكاليف المعيشة", "الاقتصاد"],
    "weather": ["weather", "forecast", "rain", "rainfall", "storm", "heatwave",
                "heat wave", "temperature", "temperatures", "flood", "fog",
                "sandstorm", "dust storm", "طقس", "الطقس", "حالة الطقس",
                "حالة الجو", "الارصاد", "الأرصاد", "امطار", "أمطار",
                "الامطار", "الأمطار", "عاصفة", "العاصفة", "موجة حر",
                "موجة حارة", "درجات الحرارة", "الحرارة", "غبار", "الغبار",
                "سيول", "ضباب", "منخفض جوي"],
    "World Cup": ["world cup", "worldcup", "fifa", "fifa world cup",
                "qatar 2022", "mondial", "wc2022", "world cup 2022",
                "world cup 2026", "كاس العالم", "كأس العالم", "مونديال",
                "المونديال", "مونديال قطر", "كاس العالم قطر", "فيفا",
                "الفيفا", "بطولة كاس العالم", "المنتخب"],
}

# description keywords, only longer and more specific terms
DESC_TOPICS = {
    "ChatGPT": ["chatgpt", "chat gpt", "openai", "شات جي بي تي",
                "شات جيبيتي", "تشات جي بي تي"],
    "artificial intelligence": ["artificial intelligence", "machine learning",
                "deep learning", "الذكاء الاصطناعي", "ذكاء اصطناعي",
                "تعلم الآلة", "تعلم الالة", "التعلم العميق"],
    "bitcoin": ["bitcoin", "cryptocurrency", "blockchain", "بيتكوين",
                "البيتكوين", "العملات الرقمية", "العملات المشفرة", "بلوك تشين"],
    "electric cars": ["electric car", "electric vehicle", "السيارات الكهربائية",
                "سيارات كهربائية", "المركبات الكهربائية"],
    "Ramadan": ["ramadan", "شهر رمضان", "عيد الفطر", "التراويح"],
    "inflation": ["inflation", "cost of living", "interest rates", "التضخم",
                "غلاء المعيشة", "ارتفاع الأسعار", "ارتفاع الاسعار",
                "أسعار الفائدة", "اسعار الفائدة", "تكاليف المعيشة"],
    "weather": ["weather forecast", "heatwave", "sandstorm", "حالة الطقس",
                "الأرصاد الجوية", "الارصاد الجوية", "موجة حر", "منخفض جوي"],
    "World Cup": ["world cup", "fifa", "كأس العالم", "كاس العالم",
                "المونديال"],
}

ARABIC_DIAC = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0640]")


def normalise_ar(s):
    s = ARABIC_DIAC.sub("", s)
    for a, b in (("\u0623", "\u0627"), ("\u0625", "\u0627"), ("\u0622", "\u0627"),
                 ("\u0649", "\u064a"), ("\u0629", "\u0647"), ("\u0624", "\u0648"),
                 ("\u0626", "\u064a")):
        s = s.replace(a, b)
    return s


def _is_arabic(t):
    return any("\u0600" <= ch <= "\u06ff" for ch in t)


def _compile(topic_map):
    out = {}
    for topic, keys in topic_map.items():
        lat = [re.escape(k.strip().lower()) for k in keys if not _is_arabic(k)]
        ar = [normalise_ar(k.strip()) for k in keys if _is_arabic(k)]
        rx = (re.compile(r"(?<![a-z0-9])(?:" + "|".join(lat) + r")(?![a-z0-9])")
              if lat else None)
        out[topic] = (rx, ar)
    return out


_TITLE = _compile(TOPICS)
_DESC = _compile(DESC_TOPICS)


def _matches(text, compiled):
    if not text:
        return set()
    low = text.lower()
    norm = normalise_ar(low)
    return {t for t, (rx, ar) in compiled.items()
            if (rx and rx.search(low)) or any(a in norm for a in ar)}


def assign_topic(title, description):
    """Returns (topic, source). Tries the title first, then the description
    if it matches exactly one topic. Returns (None, reason) otherwise."""
    t_hits = _matches(title, _TITLE)
    if t_hits:
        for topic in TOPICS:
            if topic in t_hits:
                return topic, "title"
    d_hits = _matches(description, _DESC)
    if len(d_hits) == 1:
        return next(iter(d_hits)), "description"
    return None, ("desc_multi_topic" if len(d_hits) > 1 else "none")


# state and output files

def backup_once():
    for f in (COMMENTS_CSV, STATE_JSON):
        b = f.replace(".", "_round1_backup.", 1)
        if os.path.exists(f) and not os.path.exists(b):
            shutil.copy2(f, b)
            print(f"  backup: {f} -> {b}")


def load_state():
    if not os.path.exists(STATE_JSON):
        sys.exit(f"{STATE_JSON} not found. The round 1 state file is needed "
                 f"so already collected videos are skipped.")
    with open(STATE_JSON, encoding="utf-8") as f:
        s = json.load(f)
    s["done_videos"] = set(s.get("done_videos", []))
    s.setdefault("done_channels", [])
    s.setdefault("video_lists", {})
    s.setdefault("list_rule", {})          # which rule each channel's list was built with
    s.setdefault("done_channels_r2", [])
    return s


def save_state(s):
    out = dict(s)
    out["done_videos"] = sorted(s["done_videos"])
    out["updated"] = datetime.now(timezone.utc).isoformat()
    tmp = STATE_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    os.replace(tmp, STATE_JSON)


COMMENT_COLS = ["comment_id", "region", "tier", "topic", "channel", "video_id",
                "video_published", "published", "week", "likes", "text"]


def append_comments(rows):
    """Append using the same columns and order as round 1."""
    if not rows:
        return
    df = pd.DataFrame(rows)[COMMENT_COLS]
    df.to_csv(COMMENTS_CSV, mode="a", header=not os.path.exists(COMMENTS_CSV),
              index=False, encoding="utf-8-sig")


def append_topics(rows):
    if not rows:
        return
    pd.DataFrame(rows).to_csv(TOPICS_CSV, mode="a",
                              header=not os.path.exists(TOPICS_CSV),
                              index=False, encoding="utf-8-sig")


# YouTube API calls

class QuotaStop(Exception):
    pass


class Budget:
    def __init__(self, limit):
        self.used, self.limit = 0, limit

    def spend(self, n=1):
        self.used += n
        if self.used >= self.limit:
            raise QuotaStop()


def get_service():
    if not API_KEY:
        sys.exit("Set YOUTUBE_API_KEY first.")
    from googleapiclient.discovery import build
    return build("youtube", "v3", developerKey=API_KEY, cache_discovery=False)


def _http_error_text(e):
    return str(e)


def resolve(yt, ref, budget):
    budget.spend(1)
    try:
        r = yt.channels().list(part="contentDetails,snippet",
                               forHandle=ref.lstrip("@")).execute()
    except Exception as e:
        if "quotaExceeded" in _http_error_text(e):
            raise QuotaStop()
        print(f"    resolve error: {e}")
        return None, None
    items = r.get("items", [])
    if not items:
        return None, None
    it = items[0]
    return it["contentDetails"]["relatedPlaylists"]["uploads"], it["snippet"]["title"]


def list_videos(yt, uploads, budget):
    vids, token, n, skipped_multi = [], None, 0, 0
    while n < MAX_VIDEOS_PER_CHANNEL:
        budget.spend(1)
        try:
            r = yt.playlistItems().list(part="snippet", playlistId=uploads,
                                        maxResults=50, pageToken=token).execute()
        except Exception as e:
            if "quotaExceeded" in _http_error_text(e):
                raise QuotaStop()
            print(f"    playlist error: {e}")
            break
        oldest = None
        for it in r.get("items", []):
            sn = it["snippet"]
            pub = pd.to_datetime(sn.get("publishedAt"), utc=True, errors="coerce")
            oldest = pub
            if pub is pd.NaT or not (WINDOW_START <= pub <= WINDOW_END):
                continue
            topic, src = assign_topic(sn.get("title", "") or "",
                                      sn.get("description", "") or "")
            if topic:
                vids.append({"video_id": sn["resourceId"]["videoId"],
                             "title": sn.get("title", "") or "",
                             "published": pub.isoformat(),
                             "topic": topic, "source": src})
            elif src == "desc_multi_topic":
                skipped_multi += 1
        n += len(r.get("items", []))
        if oldest is not None and oldest is not pd.NaT and oldest < WINDOW_START:
            break
        token = r.get("nextPageToken")
        if not token:
            break
    return vids, skipped_multi


def get_comments(yt, video_id, budget):
    out, token, pages = [], None, 0
    while pages < MAX_COMMENT_PAGES:
        budget.spend(1)
        try:
            r = yt.commentThreads().list(part="snippet", videoId=video_id,
                                         maxResults=100, textFormat="plainText",
                                         pageToken=token).execute()
        except Exception as e:
            msg = _http_error_text(e)
            if "quotaExceeded" in msg:
                raise QuotaStop()
            if ("commentsDisabled" in msg or "disabled comments" in msg
                    or "videoNotFound" in msg):
                return out
            print(f"    comments error {video_id}: {e}")
            return out
        for it in r.get("items", []):
            c = it["snippet"]["topLevelComment"]
            sn = c["snippet"]
            out.append({"comment_id": c["id"], "published": sn.get("publishedAt"),
                        "text": (sn.get("textDisplay") or "").replace("\n", " ").strip(),
                        "likes": sn.get("likeCount", 0)})
        token = r.get("nextPageToken")
        pages += 1
        if not token:
            break
    return out


def usable_weeks():
    if not os.path.exists(COMMENTS_CSV):
        return pd.Series(dtype=int)
    df = pd.read_csv(COMMENTS_CSV, encoding="utf-8-sig",
                     usecols=["region", "topic", "week", "comment_id"],
                     low_memory=False).drop_duplicates("comment_id")
    wk = df.groupby(["region", "topic", "week"]).size().rename("n").reset_index()
    return wk[wk["n"] >= 20].groupby(["region", "topic"]).size()


def show_status(state=None):
    state = state or load_state()
    uw = usable_weeks()
    print("\nUsable weeks per region-topic (20+ comments), round 1 vs now")
    print(f"  {'region':<7}{'topic':<26}{'round 1':>8}{'now':>6}{'change':>8}")
    keys = sorted(set(ROUND1_USABLE_WEEKS) | set(uw.index))
    for k in keys:
        before = ROUND1_USABLE_WEEKS.get(k, 0)
        now = int(uw.get(k, 0))
        mark = "  now 15+" if before < 15 <= now else ""
        print(f"  {k[0]:<7}{k[1]:<26}{before:>8}{now:>6}{now-before:>+8}{mark}")
    print(f"\nVideos completed: {len(state['done_videos']):,}")
    print(f"Channels finished under rule 2: {state.get('done_channels_r2', [])}")
    if os.path.exists(TOPICS_CSV):
        t = pd.read_csv(TOPICS_CSV, encoding="utf-8-sig")
        print("\nVideos matched in round 2, by source:")
        print(t.groupby(["channel", "source"]).size().unstack(fill_value=0).to_string())


def main():
    if "--status" in sys.argv:
        show_status()
        return

    backup_once()
    state = load_state()
    yt = get_service()
    budget = Budget(DAILY_UNIT_BUDGET)
    log = []
    print(f"Window: {WINDOW_START.date()} to {WINDOW_END.date()}")
    print(f"{len(state['done_videos']):,} videos already collected\n")

    try:
        for region, ref, tier in CHANNELS:
            key = f"{region}:{ref}"
            print(f"[{region}] {ref}")

            if (key in state["video_lists"]
                    and state["list_rule"].get(key) == RULE_VERSION):
                videos = state["video_lists"][key]
                print(f"    {len(videos)} topic videos (cached, rule {RULE_VERSION})")
            else:
                if key in state["video_lists"]:
                    print("    rebuilding list with the description rule")
                uploads, name = resolve(yt, ref, budget)
                if not uploads:
                    print("    could not resolve channel, skipping\n")
                    continue
                videos, n_multi = list_videos(yt, uploads, budget)
                state["video_lists"][key] = videos
                state["list_rule"][key] = RULE_VERSION
                save_state(state)
                n_desc = sum(1 for v in videos if v.get("source") == "description")
                print(f"    resolved: {name}")
                print(f"    {len(videos)} topic videos ({n_desc} via description, "
                      f"{n_multi} roundups skipped)")

            todo = [v for v in videos if v["video_id"] not in state["done_videos"]]
            print(f"    {len(todo)} not yet collected")

            got = 0
            for i, v in enumerate(todo, 1):
                comments = get_comments(yt, v["video_id"], budget)
                rows = []
                for c in comments:
                    pub = pd.to_datetime(c["published"], utc=True, errors="coerce")
                    if pub is pd.NaT:
                        continue
                    rows.append({
                        "comment_id": c["comment_id"], "region": region,
                        "tier": tier, "topic": v["topic"], "channel": ref,
                        "video_id": v["video_id"],
                        "video_published": v["published"],
                        "published": pub.isoformat(),
                        # same week definition as round 1
                        "week": pub.tz_convert(None).to_period("W-SUN").start_time,
                        "likes": c["likes"], "text": c["text"]})
                append_comments(rows)
                append_topics([{"video_id": v["video_id"], "channel": ref,
                                "region": region, "topic": v["topic"],
                                "source": v.get("source", "title"),
                                "video_published": v["published"],
                                "title": v.get("title", ""),
                                "n_comments": len(rows)}])
                got += len(rows)
                state["done_videos"].add(v["video_id"])
                if i % 10 == 0:
                    save_state(state)
                    print(f"      {i}/{len(todo)} videos, {got:,} comments, "
                          f"{budget.used:,} units")
                time.sleep(0.03)

            save_state(state)
            if key not in state["done_channels_r2"]:
                state["done_channels_r2"].append(key)
                save_state(state)
            log.append({"channel": key, "new_videos": len(todo), "new_comments": got})
            print(f"    {got:,} new comments\n")

        print("Collection finished")
    except QuotaStop:
        save_state(state)
        print("\nDaily quota reached, progress saved.")
        print("Quota resets at midnight Pacific time.")
    except KeyboardInterrupt:
        save_state(state)
        print("\nStopped, progress saved.")

    if log:
        pd.DataFrame(log).to_csv(LOG_CSV, index=False)
    print(f"Units used this run: {budget.used:,}")
    show_status(state)


if __name__ == "__main__":
    main()
