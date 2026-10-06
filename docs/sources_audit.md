# Sources audit: the organizers' approved list vs what Baseera has (2026-10-06)

The list itself is saved in `docs/approved_sources_links.txt`. Every link in it was opened and answered (the only exceptions: `center.dawa` is a typo for `dawa.center`, and `islamenc.com` timed out once). Result per source:

## Already in Baseera before today

| Source | What we use | Status |
|---|---|---|
| qurancomplex.gov.sa/quran-dev | Hafs Mushaf text (ground truth) and Tafseer al-Muyassar | in the database |
| quranenc.com (API) | English translation `english_saheeh` | in the database (the API lists 76 translations; none of them is an Arabic tafsir) |
| hadeethenc.com (API) | 3,574 hadiths, Arabic + English, with grades | in the database |
| dorar.net/article/389 (hadith API) | live hadith grades for Verify mode | live + cached |
| icadb.com | Q&A for Muslims (id 2) and non-Muslims (id 11), terminology (id 5) | in the database |
| dawa.center/file/7937 (Bayyinat) | 263 questions on doubts | in the database |
| mp3quran.net (API) | recitation audio links | live + cached |
| islamqa.info, binbaz.org.sa, binothaimeen.net | referral links only | links only |

## Brought in today (new)

| Source | What was added | Where it shows up |
|---|---|---|
| icadb.com books API | 335 of the 345 books of the organizers' central database (fiqh, hajj and umrah, funerals, women's rulings, prayer, creed...), about 40,000 distinct passages after removing identical text published in several editions | keyword search (not embedded: CPU embedding is about 4 passages/second) |
| dorar.net/feqhia | the Fiqh Encyclopedia: 2,489 articles, 4,595 passages, each a question-style heading with the positions of the schools and references | keyword + semantic search, links back to the page |
| icadb encyclopedia 106 | dictionary of Quranic words: 2,632 words | semantic + keyword |
| icadb encyclopedias 108, 103, 109, 104, 107 | notable people (291), places (78), sects and religions (58), Names of Allah (104), particles of meaning (82) | semantic + keyword |

Deliberately NOT ingested: the icadb hadith cards (project rule 3: hadith text and grades come only from HadeethEnc and Dorar), Friday sermons, and the dawah-method dictionaries (not answer content).

## Not brought, and why (what is needed)

| Source | Reason | What I need from you |
|---|---|---|
| shamela.ws | The library is a very large download (`dev.shamela.ws/downloads/shamela-database-1448.zip`, size not shown) in its own database format; parsing it is a separate project | Nothing for tonight. If you want it later, say so and approve the download |
| islamhouse.com (API v3, documented in Postman) | The API gives metadata and links; the actual book and article texts are attachments (PDF, audio, video), which means downloading and parsing large files | Nothing for tonight; possible later for a "books to read" feature |
| dorar.net/aqeeda, /tafseer, /history | Same site and same structure as the fiqh encyclopedia, so the same crawler would work; not done only for time | Say the word and I run the crawler on them (about 15 to 30 minutes each) |
| quranpedia.net, tafsir.net, modoee.com, wahy.net, surahapp.com | Web apps with no bulk data API found; they hold more tafsir | If you can get a tafsir dataset as JSON or a database (for example Tafsir al-Sa'di or Ibn Kathir), bring it and I will ingest it; today we only have Tafseer al-Muyassar |
| byenah.com (portal and Postman collection), islamic-content.com (al-Jamhara dictionary, MCP server), terminologyenc.com, islamenc.com | Dawah introductions and dictionaries; their API is described in a Postman collection I did not get to; terminologyenc is likely part of the same family as icadb terminology | The Postman collection JSON of byenah (link on its API page) if you want me to add it |
| risala.prh.gov.sa, bohoth.awqaf.gov.kw | Plain websites (articles, PDFs), no API | Nothing, unless you want specific articles added by hand |
| ksaa.gov.sa dictionaries (Riyadh, Siwar, Falak) | Arabic language dictionaries, not Islamic content | An API key if you want it, otherwise skip |
| fonts.qurancomplex.gov.sa, qurancomplex.gov.sa/quran-translations | A Quran font would improve how verses look in the UI; more translations of the meanings | Approval to download the Uthmanic font file (a few MB) if you want it |

## Honest limits

- The 40,000 book passages are searched by keyword only. Their quality as answers has been checked on the question sets listed in `docs/how_baseera_decides.md`, not passage by passage.
- Every new passage links back to its original page or API record, and is credited in `NOTICE.md`.
