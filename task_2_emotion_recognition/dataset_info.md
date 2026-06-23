# Dataset Profile: RAVDESS Speech Dataset

This file documents the characteristics and coding mappings for the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) speech dataset.

## Overview
- **Dataset Name:** Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)
- **Modality:** Audio-Only Speech
- **Formats:** 16-bit, 48kHz uncompressed WAV audio
- **Number of Files:** 1,440
- **Target Classes:** 8 emotions (`neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`)
- **Actors:** 24 actors (12 male, 12 female). Odd actor IDs are male; even actor IDs are female.

---

## Filename Coding Scheme

Each audio file in the dataset follows a strict 7-part numerical identifier convention:
`modality-vocal_channel-emotion-intensity-statement-repetition-actor.wav`

Example filename: `03-01-06-01-02-01-02.wav`

| Identifier Part | Description | Values Map |
|:---|:---|:---|
| **01 (Modality)** | Modality type | 01: full AV, 02: video-only, 03: audio-only |
| **02 (Vocal Channel)** | Vocal channel type | 01: speech, 02: song |
| **03 (Emotion)** | Target emotional state | 01: neutral, 02: calm, 03: happy, 04: sad, 05: angry, 06: fearful, 07: disgust, 08: surprised |
| **04 (Intensity)** | Emotional intensity level | 01: normal, 02: strong |
| **05 (Statement)** | North American statement text | 01: "Kids are talking by the door"<br>02: "Dogs are sitting by the door" |
| **06 (Repetition)** | Repetition iteration number | 01: 1st repetition, 02: 2nd repetition |
| **07 (Actor)** | Actor identifier | 01 to 24 (Odd: Male, Even: Female) |

---

## Expected Data Distribution

1. **Total Speech Files:** 1,440 (60 files per actor)
2. **Actor Distribution:** 60 files per actor (balanced across 12 male and 12 female actors)
3. **Emotion Distribution:**
   - `neutral`: 96 files (only has normal intensity)
   - `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`: 192 files each (96 normal intensity + 96 strong intensity)
   - **Total:** $96 + (7 \times 192) = 1,440$
4. **Gender Balance:** 720 male vocal samples, 720 female vocal samples.
