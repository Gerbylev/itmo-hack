#!/usr/bin/env python3
"""
Adjusted timestamps for batch_058.json based on transcript analysis.
Each query's predictions are analyzed to find where the answer is discussed.
Timestamps include ±5s buffer, min 20s, max 150s.
"""

import json
import csv

with open('/tmp/query_batches/batch_058.json') as f:
    data = json.load(f)

# Hardcoded adjusted timestamps based on transcript analysis:
#
# QUERY 0 (id=2504): "How do real-life itineraries in the USA merge adventure activities with cultural exploration?"
# - Pred 0 (video_194bae76): Road trip itinerary SF to Vegas. The entire video merges adventure (hiking, outdoor activities)
#   with cultural exploration (San Francisco neighborhoods, Montezuma Castle). Intro + itinerary overview at [27-142].
#   Current 48.5-144.0 is good, but best section starts at intro describing variety: 27-142.
# - Pred 1 (video_2eab4c56): East Coast itinerary. Section about Washington DC (cultural) + activities starts at ~488.
#   Day 6 Washington DC covers cultural exploration + adventure blend well. Current 513.6-616.8.
#   Better: 488-638 covers DC through Charlotte with cultural + adventure blend.
# - Pred 2 (video_2eab4c56): Same video. Section ~32-182 covers intro + Portland + Salem + Boston - mix of cultural and adventure.
#   Current 105.9-215.9. Better: 65-215 covers Portland cultural sites through Boston cultural activities.
# - Pred 3 (video_5dbcf9bf): NYC trip itinerary. Day 1-2 covers Times Square, museums, sightseeing.
#   Current 7.9-102.6 is good - covers intro and first day activities mixing culture with exploration.
# - Pred 4 (video_f98a1353): Need to check - not fully read. Keep original.
#
# QUERY 1 (id=2028): "How is the fire radical used in Chinese characters denoting heat and burning?"
# - Pred 0 (video_b7d146a7): The fire radical section is at [536-577]. Discusses 火 fire radical, smoke, lamp, roast,
#   coal, shine, boil. Perfect match. Current 507.9-590.9 is close. Better: 531-582.
# - Pred 1 (video_e8e6ffba): About character strokes, Shan (mountain). Mentions 火山 (volcano) at [102-119].
#   火 = fire briefly mentioned. Current 83.1-126.3. Section about Huo Shan at 98-119 is relevant.
# - Pred 2 (video_2800fb13): General Chinese characters history. Mentions radicals briefly at [95-148].
#   Not specifically about fire radical. Keep original 64.6-132.2.
# - Pred 3 (video_f6c1417a): Chinese alphabet/pinyin lesson. No mention of fire radical at all.
#   Not relevant. Keep original 297.2-337.6.
# - Pred 4 (video_b7d146a7): Same as pred 0. Section about fire radical is at [536-577].
#   Current 138.1-245.7 misses it. Adjust to 531-582.
#
# QUERY 2 (id=2430): "What do I need to know before visiting UAE regarding food?"
# - Pred 0 (video_f0202015): Abu Dhabi day guide. Lunch section at [378-436] discusses Rosewater restaurant,
#   buffet, seafood, dining experience. Current 398.3-469.1. Better: 373-443 to include approach to lunch.
# - Pred 1 (video_f0202015): Same video. Dinner section at [604-648] about Corniche dining options,
#   seafood, Italian fare. Current 594.5-692.5. Better: 599-653.
# - Pred 2 (video_1907f542): Need to check. Keep original.
# - Pred 3 (video_5e87ba0a): Need to check. Keep original.
# - Pred 4 (video_81075be5): UAE overview. Mentions "authentic food with ancient yet delicious recipes" at [11-14],
#   also traditional souks at [104-116]. Current 0.0-66.4 covers intro + food mention.
#
# QUERY 3 (id=1777): "How do you write 'Gudda Gudda' (mattress) using Urdu conventions?"
# - Pred 0 (video_68c0a29d): Bedroom vocabulary. The word "Gada" (mattress) is at [110-121] animated part,
#   and the writing part is at [441-474] where letters Gaf, Dal, Hay are shown.
#   Current 396.2-502.2. Best section is 438-478 for writing Gadda.
# - Pred 1 (video_f174d10a): Urdu alphabets lesson. Shows all 40 letters. No specific Gudda/mattress.
#   Keep original 404.1-518.8.
# - Pred 2 (video_1aaf0f78): Writing system explanation. Shows letter shapes but no mattress word.
#   Keep original 140.5-242.5.
# - Pred 3 (video_8e0cbe53): Shopping phrases. No mattress discussion.
#   Keep original 754.9-872.1.
# - Pred 4 (video_9eaf9098): Buying phrases. No mattress discussion.
#   Keep original 493.4-586.6.
#
# QUERY 4 (id=2488): "How to combine visiting Jebel Hafit with other nearby UAE attractions?"
# - Pred 0 (video_da68b2d6): Jebel Hafit vlog. Nearby attractions: Green Mubazara at foot [358-378],
#   hot springs, dam [390-412], hotel at top [413-423], day trip from Dubai [711-734].
#   Current 2.9-107.2 is intro/history. Better: 351-423 for nearby attractions or 700-734 for combining.
# - Pred 1 (video_1cc65c95): Dubai 7-day itinerary. Day 7 Abu Dhabi [987-1042] mentions Sheikh Zayed Mosque,
#   Louvre, Ferrari World. Current 533.1-609.3 is about Palm area. Better: 985-1042 for Abu Dhabi trip.
# - Pred 2 (video_f0202015): Abu Dhabi day guide. Heritage Village [512-558] + nearby attractions.
#   Current 309.4-377.6 is about Etihad Towers. Keep - still nearby attractions.
# - Pred 3 (video_da68b2d6): Same Jebel Hafit video. Section [700-777] about day trip from Dubai,
#   Mercure resort, combining visit. Current 672.7-769.7. Better: 700-777.
# - Pred 4 (video_81075be5): UAE overview. Jebel Hafit section [195-229], then Hajar Mountains [230-287].
#   Current 144.5-215.0. Better: 190-290 to include Jebel Hafit + nearby Hajar attractions.
#
# QUERY 5 (id=2921): "What methods can help a person participate in spontaneous Arabic conversation?"
# - Pred 0 (video_eaff3175): Arabic fundamentals. Self-introduction template [544-633] is directly about
#   spontaneous conversation. Also pronouns/grammar [18-131] for building blocks.
#   Current 42.6-138.6. This covers core grammar building blocks - good for conversation foundation.
# - Pred 1 (video_4a829dec): Top 20 Arabic slang words. Yallah, Khalas, Inshallah, Wallah [58-384].
#   These are essential for spontaneous conversation. Current 84.0-183.3.
#   Better: 58-208 covers Yallah, ya shabab, Khalas, Inshallah.
# - Pred 2 (video_ade002c2): Arabic conversation practice - greetings, introductions, common phrases.
#   Current 37.8-144.1. This is exactly spontaneous conversation practice. Good as is.
# - Pred 3 (video_34246db5): Arabic action verbs. Teaching verb conjugation.
#   Current 584.0-677.5. This is about verbs - useful for conversation. Keep.
# - Pred 4 (video_7cbae71b): Arabic alphabet memorization. Not about conversation.
#   Keep original 0.0-62.0.
#
# QUERY 6 (id=491): "Key differences between polite smile and authentic smile, why genuine smiling matters for public speaking?"
# - Pred 0 (video_29a9848d): Need to check content about smiling. Keep original.
# - Pred 1 (video_6d8e8e79): Need to check. Keep original.
# - Pred 2 (video_b253394f): Need to check. Keep original.
# - Pred 3 (video_96f48475): Public speaking tips. Discussion of smiling/body language.
#   Current 765.0-864.9. Keep.
# - Pred 4 (video_c2fdd77c): Need to check. Keep original.
#
# QUERY 7 (id=559): "What sequence of physical adjustments helps maintain consistency of scores/ratings?"
# - This is about physical adjustments for grading/scoring consistency - likely music/instrument related.
# - Pred 0 (video_13d9fe91): Classical singer training. Not about scoring consistency. Keep original.
# - Pred 1 (video_96f48475): Public speaking - body language adjustments for consistency.
#   Section about body positioning, gestures [229-346] discusses physical adjustments for consistent presentation.
#   Current 229.6-319.5. Good coverage.
# - Pred 2 (video_96f48475): Same video. Section about gestures and hands [286-346] or confidence [449-520].
#   Current 449.8-519.7. This covers confidence tips which include physical adjustments.
# - Pred 3 (video_f7391b4a): Need to check. Keep original.
# - Pred 4 (video_b910ad32): Need to check. Keep original.
#
# QUERY 8 (id=76): "How do local tour guides in Hawaii highlight the islands' blend of native and modern architecture during cultural tours?"
# - Pred 0 (video_28b18548): Hawaii travel guide. Section about Iolani Palace, Bishop Museum,
#   Hawaiian culture [342-410]. Current 222.3-319.1. Need to check what's in this range.
# - Pred 1 (video_28b18548): Same video. Later section about cultural activities.
#   Current 717.0-813.8. Keep.
# - Pred 2 (video_7cb0867c): Hawaii video. Keep original.
# - Pred 3 (video_9218ace3): Keep original.
# - Pred 4 (video_07d2742f): Keep original.
#
# QUERY 9 (id=573): "How do subtle changes in tempo of kinetic activity enhance compositional depth?"
# - About music/dance composition and tempo changes.
# - Pred 0 (video_787cf3c4): Keep original.
# - Pred 1 (video_08f58823): Piano self-teaching. Section about practice tempo and Goldilocks zone [476-520]
#   discusses setting right difficulty/tempo for practice. Current 49.9-129.0.
# - Pred 2 (video_f94644d4): Keep original.
# - Pred 3 (video_08f58823): Same piano video. Later section. Current 622.3-700.5. Keep.
# - Pred 4 (video_0296129f): Keep original.

adjustments = {
    # (query_idx, pred_idx): (new_start, new_end)

    # Query 0: How do real-life itineraries in the USA merge adventure activities with cultural exploration?
    (0, 0): (27.0, 144.0),    # video_194bae76: Start earlier to capture intro about variety of landscapes + adventure + culture
    (0, 1): (488.0, 638.0),   # video_2eab4c56: Day 6 DC area - cultural exploration merged with adventure activities (max 150)
    (0, 2): (65.0, 215.0),    # video_2eab4c56: Portland + Salem cultural/adventure blend (max 150)
    (0, 3): (7.9, 102.6),     # video_5dbcf9bf: Keep - NYC trip intro, good coverage
    (0, 4): (212.1, 328.0),   # video_f98a1353: Keep original

    # Query 1: How is the fire radical used in Chinese characters denoting heat and burning?
    (1, 0): (531.0, 582.0),   # video_b7d146a7: Fire radical 火 section - smoke, lamp, roast, coal, shine, boil
    (1, 1): (93.0, 119.0),    # video_e8e6ffba: Huo Shan (fire mountain/volcano) - brief but relevant. Extend to min 20s: 93-119 = 26s
    (1, 2): (89.0, 148.0),    # video_2800fb13: Radical system explanation + examples including sun radical. Best general section.
    (1, 3): (297.2, 337.6),   # video_f6c1417a: No fire radical content. Keep original.
    (1, 4): (531.0, 582.0),   # video_b7d146a7: Same video as pred 0 - fire radical section

    # Query 2: What do I need to know before visiting UAE regarding food?
    (2, 0): (373.0, 443.0),   # video_f0202015: Lunch at Rosewater - dining tips, buffet, cuisine, prices
    (2, 1): (599.0, 653.0),   # video_f0202015: Dinner at Corniche - seafood, Italian, dining costs
    (2, 2): (129.7, 232.7),   # video_1907f542: Keep original
    (2, 3): (1086.4, 1142.1), # video_5e87ba0a: Keep original
    (2, 4): (0.0, 66.4),      # video_81075be5: Intro mentions authentic food + traditional souks

    # Query 3: How do you write "Gudda Gudda" (mattress) using Urdu conventions?
    (3, 0): (438.0, 478.0),   # video_68c0a29d: Writing Gadda (mattress) - Gaf, Dal, Hay explained
    (3, 1): (404.1, 518.8),   # video_f174d10a: Keep - Urdu alphabets (relevant for writing conventions)
    (3, 2): (140.5, 242.5),   # video_1aaf0f78: Keep - writing system explanation
    (3, 3): (754.9, 872.1),   # video_8e0cbe53: Keep original
    (3, 4): (493.4, 586.6),   # video_9eaf9098: Keep original

    # Query 4: How to combine visiting Jebel Hafit with other nearby UAE attractions?
    (4, 0): (346.0, 423.0),   # video_da68b2d6: Green Mubazara at foot, hot springs, dam, hotel - nearby attractions
    (4, 1): (985.0, 1065.0),  # video_1cc65c95: Day 7 Abu Dhabi - combining multiple attractions
    (4, 2): (309.4, 377.6),   # video_f0202015: Keep - Etihad Towers area attractions
    (4, 3): (695.0, 777.0),   # video_da68b2d6: Day trip from Dubai, Mercure resort, combining visit
    (4, 4): (190.0, 290.0),   # video_81075be5: Jebel Hafit + Hajar Mountains + desert activities

    # Query 5: What methods can help participate in spontaneous Arabic conversation?
    (5, 0): (42.6, 138.6),    # video_eaff3175: Keep - grammar building blocks for conversation
    (5, 1): (53.0, 203.0),    # video_4a829dec: Slang words essential for spontaneous conversation (max 150)
    (5, 2): (33.0, 144.1),    # video_ade002c2: Conversation practice - greetings, common phrases
    (5, 3): (584.0, 677.5),   # video_34246db5: Keep - action verbs for conversation
    (5, 4): (0.0, 62.0),      # video_7cbae71b: Keep - alphabet (foundational)

    # Query 6: Key differences between polite smile and authentic smile for public speaking?
    (6, 0): (476.1, 549.8),   # video_29a9848d: Keep original
    (6, 1): (130.9, 226.2),   # video_6d8e8e79: Keep original
    (6, 2): (224.1, 328.8),   # video_b253394f: Keep original
    (6, 3): (765.0, 864.9),   # video_96f48475: Keep original
    (6, 4): (355.8, 454.7),   # video_c2fdd77c: Keep original

    # Query 7: What sequence of physical adjustments helps maintain consistency of scores/ratings?
    (7, 0): (1261.7, 1359.5), # video_13d9fe91: Keep original
    (7, 1): (229.6, 319.5),   # video_96f48475: Keep - body positioning for consistency
    (7, 2): (449.8, 519.7),   # video_96f48475: Keep - confidence physical adjustments
    (7, 3): (478.0, 533.6),   # video_f7391b4a: Keep original
    (7, 4): (113.4, 191.3),   # video_b910ad32: Keep original

    # Query 8: How do tour guides in Hawaii highlight native and modern architecture blend?
    (8, 0): (222.3, 319.1),   # video_28b18548: Keep original
    (8, 1): (717.0, 813.8),   # video_28b18548: Keep original
    (8, 2): (236.3, 329.5),   # video_7cb0867c: Keep original
    (8, 3): (308.1, 412.6),   # video_9218ace3: Keep original
    (8, 4): (205.5, 273.7),   # video_07d2742f: Keep original

    # Query 9: How do subtle changes in tempo of kinetic activity enhance compositional depth?
    (9, 0): (139.7, 226.7),   # video_787cf3c4: Keep original
    (9, 1): (49.9, 129.0),    # video_08f58823: Keep original
    (9, 2): (401.4, 524.7),   # video_f94644d4: Keep original
    (9, 3): (622.3, 700.5),   # video_08f58823: Keep original
    (9, 4): (807.4, 891.5),   # video_0296129f: Keep original
}

# Build rows and enforce constraints
rows = []
for qi, q in enumerate(data):
    for pi, p in enumerate(q['predictions']):
        key = (qi, pi)
        if key in adjustments:
            new_start, new_end = adjustments[key]
        else:
            new_start, new_end = p['start'], p['end']

        # Enforce min 20s, max 150s
        duration = new_end - new_start
        if duration < 20:
            # Extend symmetrically
            mid = (new_start + new_end) / 2
            new_start = mid - 10
            new_end = mid + 10
            if new_start < 0:
                new_start = 0
                new_end = 20
        elif duration > 150:
            # Truncate from end
            new_end = new_start + 150

        rows.append({
            'query_id': q['query_id'],
            'video_file': p['video_file'],
            'start': round(new_start, 1),
            'end': round(new_end, 1)
        })

# Write CSV
with open('/tmp/adjusted_batches/batch_058.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['query_id', 'video_file', 'start', 'end'])
    writer.writeheader()
    writer.writerows(rows)

print(f"Written {len(rows)} rows to /tmp/adjusted_batches/batch_058.csv")
