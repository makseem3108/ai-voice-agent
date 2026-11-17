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

## Mixed Arabic / Hebrew speech: which speech-to-text setting

**Observed (first real Arabic call, 2026-11-12, 5G):** with Nova-3 in
multi-language mode (hints: Arabic, Hebrew) the caller's short Arabic answers
came back as English and Persian-looking text ("Hi. I wanna", "اونجا راند",
"Alice Boy."), so no slot could be filled and the same question was asked
five times. The same sentences in a clean synthetic voice were transcribed
correctly in both modes, so the failure belongs to real phone audio.

**Decided for now:** the Arabic line uses Arabic-only recognition, which can
only write Arabic. Text that cannot be the line's language is not sent to the
model at all, and a loop guard stops a question from being asked a third time.

**Open:** callers do mix in Hebrew words (תור, בסדר, קופת חולים). Compare on a
recording of one real call - `scripts/stt_compare.py` - Arabic only, Arabic
with Hebrew keyterms, and multi-language with and without keyterms, then pick
the default.

**Compared (2026-11-17, one recorded call, 12 sentences, `scripts/stt_sentences.py`):**
Arabic with keyterms did best (average similarity 0.73, against 0.67 without
keyterms and 0.53 for Azure `ar-IL`) and stays the default for the Arabic line.
None of the three reliably wrote the Hebrew words.

## Next build: second speech-to-text for the phone-number turn

**Observed:** Nova-3 dropped a digit from the same ten-digit test number in
three calls out of three (2026-11-17; twice within one call it wrote nine
digits). On the one recorded sentence where all three were compared, Azure
Speech `ar-IL` wrote the number correctly and both Nova-3 settings did not.
That is thin evidence: compare four or five more recorded numbers first.

**Plan:** only while the bot is waiting for a phone number (the question, the
re-ask, a correction), keep that utterance's audio in memory and, at the end
of the turn, send it to Azure Speech (`ar-IL` / `he-IL`, short-audio
recognition) at the same time as the normal model request. If Azure's digits
form a valid Israeli number, use them; otherwise use Nova's, as today. The
read-back and the caller's "yes" stay as they are.

- **Delay:** in parallel with the model request; wait for Azure at most about
  1.2 s. Azure's response time is not measured yet.
- **Usage:** 10-25 s of audio per call against 5 free hours a month; add a
  monthly counter that stops before the limit, like the one for the voice.
- **If Azure fails** (timeout, error, 429, quota): the call continues with
  Nova's number and the log says why.
- **To weigh:** the caller's voice for that one turn goes to a second provider.

## Day words in the preferred time

**Observed (2026-11-17, Arabic line):** the caller said "الثلاثا أربعة ونص"
(Tuesday, half past four) and speech-to-text wrote "3 4 ونصف"; that text was
stored as the preferred time. In Arabic the weekday names are number words,
so the recogniser turns them into digits.

**Idea:** before storing `preferred_time`, turn a leading day number back into
the weekday (1 = الأحد ... 7 = السبت, likewise in Hebrew), or have the bot say
the day back in the confirmation sentence so the caller can correct it.

## After a false pause: restart from the last silence

**Now:** when the bot pauses on a sound that speech-to-text does not confirm,
it never continues mid-word: with under a second left the rest is dropped,
otherwise the sentence starts again. On the Hebrew and Arabic lines a whole
reply is one synthesis request, so "the sentence" is the whole reply - a false
pause eight seconds into a ten-second reply repeats all of it.

**Option:** remember where the silences are in the audio already sent and
restart from the last one (the start of the phrase that was cut).
