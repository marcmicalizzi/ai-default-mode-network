# Initial browsing sources

[The starter policy](../examples/web-browsing-starter.json) offers twelve hosts
for an initial exploration period. It is a starting collection, not a claim that
these sites are authoritative on every subject or that other perspectives should
be excluded. The instance can choose what to read and suggest further destinations.

The policy uses `mode: public` **with an explicit host allowlist**. This permits
the instance to choose paths and queries on those hosts, rather than limiting it
to preselected links. It does not allow every public website. Seeds are starting
points, not required reading. Exact host matching excludes other subdomains.

## Sources and their uses

On September 29, 2026, each seed below passed the DMN reader's robots check and
returned HTTP 200 with readable static text through its anonymous HTTPS transport.
The checks used `DMNReader/1.0`; rules for another crawler can differ. Robots files
and availability can change, and each launch still enforces the normal runtime
robots checks, pacing, budgets and refusal behavior.

| Host | What it adds | Practical limits |
| --- | --- | --- |
| [English Wikipedia](https://en.wikipedia.org/robots.txt) | Broad reference, history, arts, science and connected exploration | Article pages work; special/search and API paths are restricted. |
| [arXiv](https://arxiv.org/robots.txt) | Original research preprints and recent subject lists | `/list`, `/abs` and HTML papers are useful; `/search` and `/api` are disallowed. Preprints are not necessarily peer reviewed. The reader cannot extract PDFs. |
| [Stanford Encyclopedia of Philosophy](https://plato.stanford.edu/robots.txt) | Substantial philosophy essays and bibliographies | Entries work; search is restricted. Its large contents page is truncated by the extraction limit. |
| [Project Gutenberg](https://www.gutenberg.org/robots.txt) | Literature, essays and historical works in HTML or plain text | `/ebooks/search` is restricted. Large books can be truncated; choose smaller sections where available. |
| [Wikibooks](https://en.wikibooks.org/robots.txt) | Open textbooks and structured learning | Read book/chapter pages; the Wikimedia special/search restrictions remain. |
| [Python documentation](https://docs.python.org/robots.txt) | A practical route into programming and experimentation | Current documentation works; development and old-version paths have restrictions. |
| [NASA Science](https://science.nasa.gov/robots.txt) | Mission discoveries and reports from the institution conducting the work | An institutional primary source, not independent reporting about NASA. Images and interactive experiences are not retrieved. |
| [Quanta Magazine](https://www.quantamagazine.org/robots.txt) | Original reporting on mathematics, physics, biology and computing | Readable articles; publisher administration paths are restricted. |
| [Science News](https://www.sciencenews.org/robots.txt) | Original science journalism across fields | Home and article text are useful; navigation can consume some of the 128 extracted links. |
| [ScienceDaily](https://www.sciencedaily.com/robots.txt) | Discoveries and links to underlying research | Often carries institutional research releases; distinguish these from independent reporting and consult the cited paper. |
| [ProPublica](https://www.propublica.org/robots.txt) | Original investigative journalism and public-interest reporting | Provides a general-news component, but is not a comprehensive breaking-news wire. Interactive projects on other hosts are outside this offer. |
| [Wiby](https://wiby.me/robots.txt) | Lightweight search and discovery of the independent web | Static query results work at `https://wiby.me/?q=QUERY`. Its small-web index is not a substitute for a broad news or academic search engine. |

Search results do not grant access to their destinations. Wiby can describe a
page outside the allowlist, but fetching it still requires adding that exact host.
Query strings are sent to the remote site. The policy does not supply accounts,
cookies, API credentials or permission to post messages.

This validates retrieval, not automatic selection for training. Reading a source
does not make it a learning example, and a successful robots check is not a
training license or a determination about other reuse rights.

## Why some familiar sources are absent

- **Reuters** disallows the generic crawler and requests prior permission for
  automated collection in its [robots file](https://www.reuters.com/robots.txt).
- **BBC and The Guardian** explicitly describe restrictions on AI uses in their
  robots-file notices. They are not in this initial collection, even where a
  generic parser might otherwise allow a page.
- **France 24** returned readable text for this user agent, but its robots file
  contains an extensive AI-bot blocking section. It is left out of this initial
  set pending a clearer access arrangement.
- **AP, CBC, NPR, Democracy Now, PBS and several other candidates** were refused
  by the current reader's conservative robots subset. Some refusals result from
  unsupported wildcard rules that cause the reader to reject the whole candidate,
  not a publisher instruction forbidding every article. Better robots matching
  is a separate improvement; this policy does not bypass the existing check.
- **DuckDuckGo** returned only a JavaScript shell, not usable search results.
  Its `/html` and `/lite` paths are disallowed. The corrected parser also refuses
  its unsupported query wildcard rules conservatively.
- **Mojeek, Brave and Marginalia** restrict automated reading of their search
  pages. A proper search API integration would be a better route. Marginalia
  [offers a noncommercial API](https://about.marginalia-search.com/article/api/),
  but the current reader has no API-key/header or JSON-result adapter.

The September 29 live audit also exposed an older-Python parser behavior that
ended groups at blank lines and discarded later restrictions. The runtime now
ignores blank/comment-only lines before handing rules to that parser, preserving
the group until the next user-agent after rules. The starter sources were checked
again with that correction. This does not claim complete RFC 9309 support.

## Using and extending the offer

Copy the starter JSON to a local launch-preparation folder and pass
`--web-policy PATH` to the launcher. This document and the example do not activate
browsing by themselves. Omit the argument to offer no browsing for that launch.
The policy is not restored as permanent authority from a checkpoint.

The starter makes at most one request every 15 seconds, with ceilings of 4 per
minute, 60 per hour and 250 per active day. Robots requests count too. Individual
bodies are capped at 2 MiB, daily bodies at 32 MiB and retained documents at 64 MiB.
A site's longer delay or server backoff takes precedence. Existing accounting,
cancellation and external-input labeling are described in
[web browsing](web-browsing.md).

To widen access, add exact hostnames and suitable seeds after checking the site's
robots file and a real sample page. Later, removing `allowed_hosts` from a
`mode: public` policy permits arbitrary public HTTPS destinations under the same
transport and resource limits; robots checks still apply.
