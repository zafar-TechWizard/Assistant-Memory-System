"""
Generate a realistic conversation.json for testing the consolidation pipeline.

The goal is conversations that READ like transcripts of actual exchanges, not
scripts. Real people type in fragments, lowercase, half-thoughts, with tangents
and callbacks. The assistant sounds helpful but not robotic — no perfectly-formatted
status reports, no over-articulated explanations.

Writes to <config.conversation_log_path> = <project>/BRAIN/memory/data/conversation.json

Usage:
    python -m memory.processing._generate_test_conversations
"""

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from memory.config import config


SESSIONS = [
    # ═══════════════════════════════════════════════════════════════════════
    # SESSION 1 — Monday morning, demo nerves
    # ═══════════════════════════════════════════════════════════════════════
    {
        "start": datetime(2026, 5, 14, 7, 45, 0, tzinfo=timezone.utc),
        "turns": [
            ("user",      "morning"),
            ("assistant", "morning. how'd you sleep"),
            ("user",      "not great. up at like 4 thinking about the demo"),
            ("assistant", "the marcus one?"),
            ("user",      "yeah"),
            ("assistant", "you've been on this for what, three weeks? you know it cold"),
            ("user",      "i know the system. i don't know what they're gonna push back on"),
            ("assistant", "want to brainstorm objections now while you have coffee?"),
            ("user",      "yeah actually that helps"),
            ("assistant", "ok. number one — cost vs the old approach. that's the obvious one"),
            ("user",      "right. and i don't have a clean answer for that yet"),
            ("assistant", "you have the latency numbers from last week's run though. that's the lever — they pay more but the p99 is half"),
            ("user",      "yeah ok. true."),
            ("assistant", "write that on a post-it before the call"),
            ("user",      "lol ok. also remind me to call john back, i missed him last night"),
            ("assistant", "got it. when?"),
            ("user",      "after standup. like 11ish"),
            ("assistant", "done"),
            ("user",      "ok i'm getting in the shower"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SESSION 2 — Monday late morning, Sarah feels off
    # ═══════════════════════════════════════════════════════════════════════
    {
        "start": datetime(2026, 5, 14, 11, 20, 0, tzinfo=timezone.utc),
        "turns": [
            ("user",      "standup just ended"),
            ("assistant", "how was it"),
            ("user",      "fine i guess. sarah was off though"),
            ("assistant", "off how"),
            ("user",      "idk. just clipped. like answering with one word when normally she rambles"),
            ("assistant", "could be a bad morning"),
            ("user",      "maybe. i have 1:1 with her at 11:30 anyway so"),
            ("assistant", "you'll see"),
            ("user",      "yeah"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SESSION 3 — Monday evening, the Sarah blowup
    # ═══════════════════════════════════════════════════════════════════════
    {
        "start": datetime(2026, 5, 14, 19, 15, 0, tzinfo=timezone.utc),
        "turns": [
            ("user",      "ok so"),
            ("assistant", "yeah"),
            ("user",      "the 1:1 went sideways"),
            ("assistant", "with sarah?"),
            ("user",      "yeah. she basically said i've been cutting her out of decisions on the project"),
            ("assistant", "what did you say"),
            ("user",      "i got really defensive. i said something like, i make the calls on my code and she makes the calls on hers"),
            ("assistant", "oof"),
            ("user",      "yeah. i know"),
            ("assistant", "do you actually feel that way"),
            ("user",      "no. not really. she's smart, i value her input. i was just... it felt like an attack and i swung back"),
            ("assistant", "is there anything to what she's saying though"),
            ("user",      "honestly? a little. i've been moving fast and i haven't been pulling her into stuff i probably should"),
            ("assistant", "ok so it's not nothing"),
            ("user",      "no. and also — she said something similar like a month ago and i kinda blew past it"),
            ("assistant", "so this is the second time"),
            ("user",      "yeah which is why she's pissed"),
            ("assistant", "what do you want to do"),
            ("user",      "apologize. for real this time. not over slack"),
            ("assistant", "do you want help drafting a message to set it up?"),
            ("user",      "yeah just something short. asking for coffee tomorrow morning"),
            ("assistant", "try this: 'hey, i owe you a real apology for how i reacted today. can we get coffee tomorrow morning? i want to actually talk.'"),
            ("user",      "yeah that's good. don't send yet, let me sit on it"),
            ("assistant", "saved as a draft"),
            ("user",      "ok. alex is gonna be home in like 20. can you put on the playlist she made"),
            ("assistant", "the sunday one?"),
            ("user",      "yeah"),
            ("assistant", "playing"),
            ("user",      "thanks. i'm gonna lay down for a few"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SESSION 4 — Tuesday morning, quick noise
    # ═══════════════════════════════════════════════════════════════════════
    {
        "start": datetime(2026, 5, 15, 8, 5, 0, tzinfo=timezone.utc),
        "turns": [
            ("user",      "morning"),
            ("assistant", "morning"),
            ("user",      "what time is it"),
            ("assistant", "8:05"),
            ("user",      "ok thanks"),
            ("assistant", "yep"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SESSION 5 — Tuesday afternoon, the quitting thought
    # ═══════════════════════════════════════════════════════════════════════
    {
        "start": datetime(2026, 5, 15, 15, 30, 0, tzinfo=timezone.utc),
        "turns": [
            ("user",      "ok i want to say something out loud"),
            ("assistant", "go"),
            ("user",      "i think i actually want to leave the job"),
            ("assistant", "leave to do what"),
            ("user",      "not sure exactly. something with ai and education. that's been the thing in my head for months"),
            ("assistant", "this isn't because of sarah right"),
            ("user",      "no. honestly. that's just a today thing. this has been there for like a year"),
            ("assistant", "ok"),
            ("user",      "i haven't told alex. not the q word anyway"),
            ("assistant", "what does she know"),
            ("user",      "that i've been restless. she's not stupid"),
            ("assistant", "what's the version of this you'd want to be doing"),
            ("user",      "tools for teachers maybe. or like, tutoring systems for kids. something where i'm not just shipping features for a company that doesn't need me to care"),
            ("assistant", "yeah"),
            ("user",      "the problem is money. we'd need like 8 months saved up to feel ok about it"),
            ("assistant", "how close are you"),
            ("user",      "maybe 5 right now. so not yet"),
            ("assistant", "want to start a side thing on the weekends? prototype something. see if the actual work feels like you think it does"),
            ("user",      "yeah that's smart"),
            ("assistant", "i'll start a doc. you can dump ideas as they come"),
            ("user",      "ok do it"),
            ("assistant", "done. also — mom called this afternoon, you missed it"),
            ("user",      "great"),
            ("assistant", "voicemail's about your visit in july"),
            ("user",      "and the career thing?"),
            ("assistant", "and the career thing"),
            ("user",      "of course. ok i'll call her tomorrow. set a reminder"),
            ("assistant", "morning? same as usual?"),
            ("user",      "8am yeah"),
            ("assistant", "set"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SESSION 6 — Tuesday late night, a thing clicked
    # ═══════════════════════════════════════════════════════════════════════
    {
        "start": datetime(2026, 5, 15, 23, 10, 0, tzinfo=timezone.utc),
        "turns": [
            ("user",      "ok it finally clicked"),
            ("assistant", "what did"),
            ("user",      "async context managers in python. i've been confused about this for months"),
            ("assistant", "the __aenter__ thing?"),
            ("user",      "yeah. i kept thinking the manager and the thing it gives you should be the same object. they don't have to be"),
            ("assistant", "right. it's whatever the consumer needs"),
            ("user",      "so for the resource pool in our code — the manager is the pool, what you get bound is a single connection from it"),
            ("assistant", "yeah that's the pattern"),
            ("user",      "i've been writing it wrong for weeks lol"),
            ("assistant", "want to refactor tomorrow?"),
            ("user",      "yeah. add it to my afternoon"),
            ("assistant", "done. and it's 11:15 by the way"),
            ("user",      "yeah yeah i know"),
            ("assistant", "you said 11 was the goal"),
            ("user",      "i'm going. lights"),
            ("assistant", "off. night"),
        ],
    },

    # ═══════════════════════════════════════════════════════════════════════
    # SESSION 7 — Wednesday morning, mom + sarah closure
    # ═══════════════════════════════════════════════════════════════════════
    {
        "start": datetime(2026, 5, 16, 8, 20, 0, tzinfo=timezone.utc),
        "turns": [
            ("user",      "alright calling mom"),
            ("assistant", "good luck"),
            ("user",      "ok done"),
            ("assistant", "how was it"),
            ("user",      "weird actually. in a good way?"),
            ("assistant", "yeah?"),
            ("user",      "the career stuff barely came up. i don't know why"),
            ("assistant", "did you bring it up"),
            ("user",      "kind of. i told her i was thinking about a change. like, exploring"),
            ("assistant", "and"),
            ("user",      "she got quiet. that's not her usual move"),
            ("assistant", "how do you feel about telling her"),
            ("user",      "honestly relieved. it's been sitting in me"),
            ("assistant", "yeah that makes sense"),
            ("user",      "what else is going on today"),
            ("assistant", "sarah replied to your draft btw"),
            ("user",      "oh"),
            ("assistant", "she wants coffee thursday morning"),
            ("user",      "ok yeah. tell her yes"),
            ("assistant", "sent. 8am thursday at roasters"),
            ("user",      "ok focus time. mute everything til lunch"),
            ("assistant", "done"),
            ("user",      "oh wait — alex texted, milk on the way home"),
            ("assistant", "reminder for 5:30"),
            ("user",      "thanks"),
            ("assistant", "yep"),
        ],
    },
]


def build_session(start_time: datetime, turns: list) -> dict:
    sid = f"session_{uuid4()}"
    conversations = []
    current = start_time
    for role, content in turns:
        conversations.append({
            "role": role,
            "content": content,
            "timestamp": current.isoformat().replace("+00:00", "") + "Z",
        })
        # Space turns by content length — short replies = fast, long replies = slower
        gap = 4 + min(len(content), 500) // 10
        current = current + timedelta(seconds=gap)
    return {
        "session_id": sid,
        "start_time": start_time.isoformat().replace("+00:00", "") + "Z",
        "conversations": conversations,
    }


def main() -> None:
    config.ensure_directories()

    sessions = [build_session(s["start"], s["turns"]) for s in SESSIONS]
    data = {f"user_{config.user_id}": sessions}

    path = config.conversation_log_path
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    total_turns = sum(len(s["conversations"]) for s in sessions)
    print(f"Wrote {len(sessions)} sessions to {path}")
    print(f"Total turns: {total_turns}")
    print(f"File size:   {path.stat().st_size:,} bytes")
    print()
    print("Sessions:")
    for s in sessions:
        date_part = s["start_time"][:19]
        print(f"  - {date_part}  {len(s['conversations']):>3} turns  {s['session_id']}")


if __name__ == "__main__":
    main()
