# Backlog

Things we have seen and decided not to change yet. Each entry says what was
observed, so the decision can be made later with evidence.

## Side-speech handling

**Observed (call of 2026-11-12, 5G):** the caller spoke quietly to a person
next to them while the bot was talking. The local voice detector paused the
bot each time, as designed - it cannot know who the words are for.

Loudness (RMS, on a scale where the detector's threshold is 500) at the four
moments the detector fired in that call:

| Moment | RMS | What speech recognition heard |
|---|---|---|
| most likely the talk to the friend | 1669 | nothing; the bot resumed after 2.5 s |
| 1.9 s before "What? Can you give me more information?" (side-talk or the start of that question) | 666 | the question to the bot |
| "Okay. Thank you." | 965 | to the bot |
| "Goodbye." | 6294 | to the bot |

Four readings, all taken only while the bot was speaking (the detector does
not run otherwise, and audio recording was off), so this is thin evidence.
Which reading belongs to the side-talk is inferred from the log, not known.
What it does show: the reading with no transcript (1669) was *louder* than
two of the readings that led to speech meant for the bot, so loudness alone
did not separate them here.

**Options to weigh:**

1. **Volume gate** - raise `BARGE_MIN_RMS`, or adapt it to the caller's own
   speaking level, so quiet side-talk does not pause the bot. Cheap, but on
   the numbers above it would also have ignored real speech to the bot.
2. **Smart resume** - keep pausing on any voice, but when the transcript
   arrives and is not addressed to the bot (no question, no answer to what was
   asked), resume the interrupted sentence where it stopped instead of
   dropping it. Needs a rule or a model call to judge "addressed to the bot".

**Before deciding:** log the loudness of every caller turn (not only while the
bot speaks), and repeat the call on Wi-Fi to rule out the phone's own noise
handling on a mobile connection.
